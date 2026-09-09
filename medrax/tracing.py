"""Query-scoped execution tracing for MedRAX."""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


_current_trace: contextvars.ContextVar[Optional["TraceRecorder"]] = contextvars.ContextVar(
    "medrax_current_trace", default=None
)
_SECRET_KEY = re.compile(r"api[_-]?key|authorization|access[_-]?token|secret", re.I)
_DATA_IMAGE = re.compile(r"^data:(image/[^;]+);base64,(.*)$", re.S)


def safe_serialize(value: Any, *, _depth: int = 0) -> Any:
    """Convert runtime values to bounded JSON-safe summaries."""
    if _depth > 8:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        match = _DATA_IMAGE.match(value)
        if match:
            return {
                "storage": "omitted",
                "mime_type": match.group(1),
                "encoded_length": len(match.group(2)),
            }
        return value if len(value) <= 20_000 else value[:20_000] + "...[TRUNCATED]"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if _SECRET_KEY.search(str(key))
                else safe_serialize(item, _depth=_depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [safe_serialize(item, _depth=_depth + 1) for item in list(value)[:200]]

    # Avoid importing torch solely for tracing.
    if value.__class__.__module__.startswith("torch") and hasattr(value, "shape"):
        return {
            "type": value.__class__.__name__,
            "shape": list(value.shape),
            "dtype": str(getattr(value, "dtype", "unknown")),
            "device": str(getattr(value, "device", "unknown")),
        }
    if hasattr(value, "model_dump"):
        try:
            return safe_serialize(value.model_dump(), _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return safe_serialize(value.dict(), _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "content"):
        result = {
            "type": value.__class__.__name__,
            "content": safe_serialize(value.content, _depth=_depth + 1),
        }
        for attribute in ("name", "tool_call_id", "tool_calls", "response_metadata", "usage_metadata"):
            item = getattr(value, attribute, None)
            if item:
                result[attribute] = safe_serialize(item, _depth=_depth + 1)
        return result
    text = repr(value)
    return text if len(text) <= 20_000 else text[:20_000] + "...[TRUNCATED]"


class TraceRecorder:
    def __init__(
        self,
        thread_id: str,
        query: str,
        trace_dir: str | Path = "logs/traces",
        image_path: str | Path | None = None,
        display_path: str | Path | None = None,
    ) -> None:
        now = datetime.now().astimezone()
        self.trace_id = f"{now:%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}"
        self.thread_id = thread_id
        self.query = query
        self.trace_dir = Path(trace_dir)
        self.started_at = now
        self.started_clock = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self.image = self._image_metadata(image_path, display_path)

    @classmethod
    def start(cls, thread_id: str, query: str, **kwargs: Any) -> "TraceRecorder":
        return cls(thread_id, query, **kwargs)

    @staticmethod
    def _image_metadata(image_path: Any, display_path: Any) -> Optional[dict[str, Any]]:
        if not image_path:
            return None
        path = Path(image_path)
        metadata: dict[str, Any] = {
            "path": str(path),
            "display_path": str(display_path) if display_path else None,
            "format": path.suffix.lower(),
        }
        try:
            content = path.read_bytes()
            metadata.update(size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        except OSError as exc:
            metadata["metadata_error"] = str(exc)
        return metadata

    def emit(
        self,
        event: str,
        message: str,
        output: Any = None,
        *,
        level: str = "INFO",
        duration_ms: Optional[float] = None,
        stacklevel: int = 1,
    ) -> None:
        frame = inspect.currentframe()
        for _ in range(stacklevel):
            frame = frame.f_back if frame else None
        module = inspect.getmodule(frame) if frame else None
        filename = frame.f_code.co_filename if frame else "unknown"
        try:
            relative_file = Path(filename).resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            relative_file = Path(filename).name
        item = {
            "sequence": len(self.events) + 1,
            "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
            "logger": module.__name__ if module else "medrax",
            "level": level,
            "source": {
                "file": relative_file,
                "line": frame.f_lineno if frame else 0,
                "function": frame.f_code.co_name if frame else "unknown",
            },
            "event": event,
            "message": message,
            "output": safe_serialize(output),
        }
        if duration_ms is not None:
            item["duration_ms"] = round(duration_ms, 3)
        self.events.append(item)

    def finish(
        self,
        final_output: Any = None,
        error: Optional[BaseException] = None,
        error_stage: Optional[str] = None,
    ) -> tuple[Optional[Path], Optional[Path]]:
        completed_at = datetime.now().astimezone()
        document = {
            "schema_version": "1.0",
            "trace_id": self.trace_id,
            "thread_id": self.thread_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "status": "failed" if error else "completed",
            "input": {"query": self.query, "image": self.image},
            "events": self.events,
            "final_output": safe_serialize(final_output),
            "error": (
                {
                    "type": error.__class__.__name__,
                    "message": str(error),
                    "stage": error_stage,
                }
                if error
                else None
            ),
            "total_duration_ms": round((time.perf_counter() - self.started_clock) * 1000, 3),
        }
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            json_path = self.trace_dir / f"trace_{self.trace_id}.json"
            log_path = self.trace_dir / f"trace_{self.trace_id}.log"
            self._atomic_write(json_path, json.dumps(document, ensure_ascii=False, indent=2))
            lines = [self._format_event(item) for item in self.events]
            self._atomic_write(log_path, "\n".join(lines) + ("\n" if lines else ""))
            return json_path, log_path
        except Exception:
            logging.getLogger(__name__).warning("Failed to persist query trace", exc_info=True)
            return None, None

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _format_event(item: dict[str, Any]) -> str:
        source = item["source"]
        output = json.dumps(item["output"], ensure_ascii=False, separators=(",", ":"))
        duration = f" - duration_ms={item['duration_ms']}" if "duration_ms" in item else ""
        return (
            f"{item['timestamp']} - {item['logger']} - {item['level']} - "
            f"{Path(source['file']).name}:{source['line']} - "
            f"{item['event']} - {item['message']}: {output}{duration}"
        )


def bind_trace(recorder: TraceRecorder) -> contextvars.Token:
    return _current_trace.set(recorder)


def reset_trace(token: contextvars.Token) -> None:
    _current_trace.reset(token)


def get_current_trace() -> Optional[TraceRecorder]:
    return _current_trace.get()


def trace_event(event: str, message: str, output: Any = None, **kwargs: Any) -> None:
    recorder = get_current_trace()
    if recorder is not None:
        recorder.emit(event, message, output, stacklevel=2, **kwargs)

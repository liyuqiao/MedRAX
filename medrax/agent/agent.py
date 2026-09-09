import json
import operator
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from medrax.tracing import trace_event

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
        """
        self.system_prompt = system_prompt
        self.log_tools = log_tools

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        trace_event(
            "model_input",
            "Sending conversation state to the medical reasoning model",
            {"message_count": len(messages), "messages": messages},
        )
        started = time.perf_counter()
        try:
            response = self.model.invoke(messages)
        except Exception as exc:
            trace_event(
                "model_error",
                "Medical reasoning model invocation failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
                level="ERROR",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        trace_event(
            "model_output",
            "Medical reasoning model returned a response",
            {
                "content": response.content,
                "tool_calls": response.tool_calls,
                "response_metadata": getattr(response, "response_metadata", None),
                "usage_metadata": getattr(response, "usage_metadata", None),
            },
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        has_calls = len(response.tool_calls) > 0
        trace_event(
            "workflow_route",
            "Routing to tool execution" if has_calls else "No tool calls; ending workflow",
            {"has_tool_calls": has_calls, "tool_call_count": len(response.tool_calls)},
        )
        return has_calls

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the model's response.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for call in tool_calls:
            print(f"Executing tool: {call}")
            trace_event(
                "tool_input",
                f"Starting tool {call['name']}",
                {"tool_call_id": call["id"], "name": call["name"], "args": call["args"]},
            )
            started = time.perf_counter()
            if call["name"] not in self.tools:
                print("\n....invalid tool....")
                result = "invalid tool, please retry"
                level = "WARNING"
            else:
                level = "INFO"
                try:
                    result = self.tools[call["name"]].invoke(call["args"])
                except Exception as exc:
                    trace_event(
                        "tool_error",
                        f"Tool {call['name']} failed",
                        {
                            "tool_call_id": call["id"],
                            "name": call["name"],
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        level="ERROR",
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
                    raise

            trace_event(
                "tool_output",
                f"Tool {call['name']} returned a result",
                {"tool_call_id": call["id"], "name": call["name"], "result": result},
                level=level,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

            tool_message = ToolMessage(
                tool_call_id=call["id"],
                name=call["name"],
                args=call["args"],
                content=str(result),
            )
            results.append(tool_message)
            trace_event(
                "tool_message_created",
                "Tool result was converted to a ToolMessage for the next model turn",
                tool_message,
            )

        self._save_tool_calls(results)
        print("Returning to model processing!")

        return {"messages": results}

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)

import os
import warnings
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/root/autodl-tmp/medrax-models",
    temp_dir="/root/autodl-tmp/medrax-temp",
    retrieval_report_path=None,
    retrieval_combined_path=None,
    retrieval_independent_path=None,
    device="cuda",
    model="chatgpt-4o-latest",
    temperature=0.7,
    top_p=0.95,
    openai_kwargs={}
):
    """Initialize the MedRAX agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    all_tools = {
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True),
        "XRayVQATool": lambda: XRayVQATool(cache_dir=model_dir, device=device),
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(
            cache_dir=model_dir, device=device
        ),
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(
            cache_dir=model_dir, temp_dir=temp_dir, load_in_8bit=True, device=device
        ),
        "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool(
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device
        ),
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
    }

    retrieval_paths = (
        retrieval_report_path, retrieval_combined_path, retrieval_independent_path
    )
    if all(retrieval_paths):
        all_tools["CXRClinicalRetrievalTool"] = lambda: CXRClinicalRetrievalTool(
            report_path=retrieval_report_path,
            combined_path=retrieval_combined_path,
            independent_path=retrieval_independent_path,
        )
    elif tools_to_use and "CXRClinicalRetrievalTool" in tools_to_use:
        raise ValueError(
            "CXRClinicalRetrievalTool requires report, combined-phrase, and "
            "independent-phrase result paths"
        )

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    checkpointer = MemorySaver()
    model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs)
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the MedRAX application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # Example: initialize with only specific tools
    # Here three tools are commented out, you can uncomment them to use them
    selected_tools = [
        "ImageVisualizerTool",
        "DicomProcessorTool",
        "ChestXRayClassifierTool",
        "ChestXRaySegmentationTool",
        "ChestXRayReportGeneratorTool",
        "CXRClinicalRetrievalTool",
        # "XRayVQATool",
        # "LlavaMedTool",
        # "XRayPhraseGroundingTool",
        # "ChestXRayGeneratorTool",
    ]

    # Collect the ENV variables
    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key

    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir="/root/autodl-tmp/medrax-models",  # Change this to the path of the model weights
        temp_dir="/root/autodl-tmp/medrax-temp",  # Change this to the path of the temporary directory
        retrieval_report_path=(
            "/root/autodl-tmp/cxr-clip/outputs/badcase/base-report.jsonl"
        ),
        retrieval_combined_path=(
            "/root/autodl-tmp/cxr-clip/outputs/badcase/base-triple.jsonl"
        ),
        retrieval_independent_path=(
            "/root/autodl-tmp/cxr-clip/outputs/phrase-retrieval-badcase/"
            "baseline/retrieval-details.jsonl"
        ),
        device="cuda",  # Change this to the device you want to use
        model="qwen3-vl-flash",  # Change this to the model you want to use, e.g. gpt-4o-mini
        temperature=0.7,
        top_p=0.95,
        openai_kwargs=openai_kwargs
    )
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)

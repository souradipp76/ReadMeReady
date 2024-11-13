"""
Types
"""

from enum import Enum
from typing import Callable, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel


class LLMModels(str, Enum):
    """LLM Models"""

    GPT3 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT432k = "gpt-4-32k"
    TINYLLAMA_1p1B_CHAT_GGUF = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    LLAMA2_7B_CHAT_GPTQ = "TheBloke/Llama-2-7B-Chat-GPTQ"
    LLAMA2_13B_CHAT_GPTQ = "TheBloke/Llama-2-13B-Chat-GPTQ"
    CODELLAMA_7B_INSTRUCT_GPTQ = "TheBloke/CodeLlama-7B-Instruct-GPTQ"
    CODELLAMA_13B_INSTRUCT_GPTQ = "TheBloke/CodeLlama-13B-Instruct-GPTQ"
    LLAMA2_7B_CHAT_HF = "meta-llama/Llama-2-7b-chat-hf"
    LLAMA2_13B_CHAT_HF = "meta-llama/Llama-2-13b-chat-hf"
    CODELLAMA_7B_INSTRUCT_HF = "meta-llama/CodeLlama-7b-Instruct-hf"
    CODELLAMA_13B_INSTRUCT_HF = "meta-llama/CodeLlama-13b-Instruct-hf"
    GOOGLE_GEMMA_2B_INSTRUCT = "google/gemma-2b-it"
    GOOGLE_GEMMA_7B_INSTRUCT = "google/gemma-7b-it"
    GOOGLE_CODEGEMMA_2B_INSTRUCT = "google/codegemma-2b-it"
    GOOGLE_CODEGEMMA_7B_INSTRUCT = "google/codegemma-7b-it"
    GOOGLE_GEMMA_2B_INSTRUCT_GGUF = "bartowski/gemma-2-2b-it-GGUF"


class Priority(str, Enum):
    """Priority"""

    COST = "cost"
    PERFORMANCE = "performance"


class AutodocReadmeConfig:
    """AutodocReadmeConfig"""

    def __init__(self, headings: str):
        self.headings = [heading.strip() for heading in headings.split(",")]


class AutodocUserConfig:
    """AutodocUserConfig"""

    def __init__(self, llms: List[LLMModels], streaming: bool = False):
        self.llms = llms
        self.streaming = streaming


class AutodocRepoConfig:
    """AutodocRepoConfig"""

    def __init__(
        self,
        name: str,
        repository_url: str,
        root: str,
        output: str,
        llms: List[LLMModels],
        priority: Priority,
        max_concurrent_calls: int,
        add_questions: bool,
        ignore: List[str],
        file_prompt: str,
        folder_prompt: str,
        chat_prompt: str,
        content_type: str,
        target_audience: str,
        link_hosted: bool,
        peft_model_path: str | None,
        device: str,
    ):
        self.name = name
        self.repository_url = repository_url
        self.root = root
        self.output = output
        self.llms = llms
        self.peft_model_path = peft_model_path
        self.priority = priority
        self.max_concurrent_calls = max_concurrent_calls
        self.add_questions = add_questions
        self.ignore = ignore
        self.file_prompt = file_prompt
        self.folder_prompt = folder_prompt
        self.chat_prompt = chat_prompt
        self.content_type = content_type
        self.target_audience = target_audience
        self.link_hosted = link_hosted
        self.device = device


class FileSummary:
    """FileSummary"""

    def __init__(
        self,
        file_name: str,
        file_path: str,
        url: str,
        summary: str,
        questions: Optional[str],
        checksum: str,
    ):
        self.file_name = file_name
        self.file_path = file_path
        self.url = url
        self.summary = summary
        self.questions = questions
        self.checksum = checksum


class ProcessFileParams:
    """ProcessFileParams"""

    def __init__(
        self,
        file_name: str,
        file_path: str,
        project_name: str,
        content_type: str,
        file_prompt: str,
        target_audience: str,
        link_hosted: bool,
    ):
        self.file_name = file_name
        self.file_path = file_path
        self.project_name = project_name
        self.content_type = content_type
        self.file_prompt = file_prompt
        self.target_audience = target_audience
        self.link_hosted = link_hosted


class FolderSummary:
    """FolderSummary"""

    def __init__(
        self,
        folder_name: str,
        folder_path: str,
        url: str,
        files: List[FileSummary],
        folders: List["FolderSummary"],
        summary: str,
        questions: str,
        checksum: str,
    ):
        self.folder_name = folder_name
        self.folder_path = folder_path
        self.url = url
        self.files = files
        self.folders = folders
        self.summary = summary
        self.questions = questions
        self.checksum = checksum


class ProcessFolderParams:
    """ProcessFolderParams"""

    def __init__(
        self,
        input_path: str,
        folder_name: str,
        folder_path: str,
        project_name: str,
        content_type: str,
        folder_prompt: str,
        target_audience: str,
        link_hosted: bool,
        should_ignore: Callable[[str], bool],
    ):
        self.input_path = input_path
        self.folder_name = folder_name
        self.folder_path = folder_path
        self.project_name = project_name
        self.content_type = content_type
        self.folder_prompt = folder_prompt
        self.target_audience = target_audience
        self.link_hosted = link_hosted
        self.should_ignore = should_ignore


class TraverseFileSystemParams:
    """TraverseFileSystemParams"""

    def __init__(
        self,
        input_path: str,
        project_name: str,
        process_file: Optional[Callable[[ProcessFileParams], None]],
        process_folder: Optional[Callable[[ProcessFolderParams], None]],
        ignore: List[str],
        file_prompt: str,
        folder_prompt: str,
        content_type: str,
        target_audience: str,
        link_hosted: bool,
    ):
        self.input_path = input_path
        self.project_name = project_name
        self.process_file = process_file
        self.process_folder = process_folder
        self.ignore = ignore
        self.file_prompt = file_prompt
        self.folder_prompt = folder_prompt
        self.content_type = content_type
        self.target_audience = target_audience
        self.link_hosted = link_hosted


class LLMModelDetails:
    """LLMModelDetails"""

    def __init__(
        self,
        name: LLMModels,
        input_cost_per_1k_tokens: float,
        output_cost_per_1k_tokens: float,
        max_length: int,
        llm: BaseChatModel | None,
        input_tokens: int,
        output_tokens: int,
        succeeded: int,
        failed: int,
        total: int,
        gguf_file=None,
    ):
        self.name = name
        self.input_cost_per_1k_tokens = input_cost_per_1k_tokens
        self.output_cost_per_1k_tokens = output_cost_per_1k_tokens
        self.max_length = max_length
        self.llm = llm
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.succeeded = succeeded
        self.failed = failed
        self.total = total
        self.gguf_file = gguf_file

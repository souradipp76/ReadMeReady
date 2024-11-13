import pytest
from unittest.mock import MagicMock

from readme_ready.types import (
    LLMModels,
    Priority,
    AutodocReadmeConfig,
    AutodocUserConfig,
    AutodocRepoConfig,
    FileSummary,
    ProcessFileParams,
    FolderSummary,
    ProcessFolderParams,
    TraverseFileSystemParams,
    LLMModelDetails,
)


def test_llm_models():
    # Test that all enum members are accessible and correct
    assert LLMModels.GPT3 == "gpt-3.5-turbo"
    assert LLMModels.GPT4 == "gpt-4"
    assert LLMModels.GPT432k == "gpt-4-32k"
    assert (
        LLMModels.TINYLLAMA_1p1B_CHAT_GGUF
        == "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    )
    assert LLMModels.LLAMA2_7B_CHAT_GPTQ == "TheBloke/Llama-2-7B-Chat-GPTQ"
    assert LLMModels.LLAMA2_13B_CHAT_GPTQ == "TheBloke/Llama-2-13B-Chat-GPTQ"
    assert (
        LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ
        == "TheBloke/CodeLlama-7B-Instruct-GPTQ"
    )
    assert (
        LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ
        == "TheBloke/CodeLlama-13B-Instruct-GPTQ"
    )
    assert LLMModels.LLAMA2_7B_CHAT_HF == "meta-llama/Llama-2-7b-chat-hf"
    assert LLMModels.LLAMA2_13B_CHAT_HF == "meta-llama/Llama-2-13b-chat-hf"
    assert (
        LLMModels.CODELLAMA_7B_INSTRUCT_HF
        == "meta-llama/CodeLlama-7b-Instruct-hf"
    )
    assert (
        LLMModels.CODELLAMA_13B_INSTRUCT_HF
        == "meta-llama/CodeLlama-13b-Instruct-hf"
    )
    assert LLMModels.GOOGLE_GEMMA_2B_INSTRUCT == "google/gemma-2b-it"
    assert LLMModels.GOOGLE_GEMMA_7B_INSTRUCT == "google/gemma-7b-it"
    assert LLMModels.GOOGLE_CODEGEMMA_2B_INSTRUCT == "google/codegemma-2b-it"
    assert LLMModels.GOOGLE_CODEGEMMA_7B_INSTRUCT == "google/codegemma-7b-it"
    assert (
        LLMModels.GOOGLE_GEMMA_2B_INSTRUCT_GGUF
        == "bartowski/gemma-2-2b-it-GGUF"
    )


def test_priority():
    # Test that all enum members are accessible and correct
    assert Priority.COST == "cost"
    assert Priority.PERFORMANCE == "performance"


def test_autodoc_readme_config():
    headings_input = "# Introduction, ## Usage, ### Installation"
    config = AutodocReadmeConfig(headings=headings_input)
    assert config.headings == [
        "# Introduction",
        "## Usage",
        "### Installation",
    ]


def test_autodoc_user_config():
    llms = [LLMModels.GPT3, LLMModels.GPT4]
    config = AutodocUserConfig(llms=llms, streaming=True)
    assert config.llms == llms
    assert config.streaming is True


def test_autodoc_repo_config():
    config = AutodocRepoConfig(
        name="MyProject",
        repository_url="https://github.com/user/MyProject",
        root="./MyProject",
        output="./output",
        llms=[LLMModels.GPT3],
        priority=Priority.PERFORMANCE,
        max_concurrent_calls=5,
        add_questions=False,
        ignore=["*.md", "*.txt"],
        file_prompt="Explain this file.",
        folder_prompt="Explain this folder.",
        chat_prompt="Chat prompt here.",
        content_type="code",
        target_audience="developers",
        link_hosted=True,
        peft_model_path=None,
        device="cpu",
    )
    assert config.name == "MyProject"
    assert config.repository_url == "https://github.com/user/MyProject"
    assert config.root == "./MyProject"
    assert config.output == "./output"
    assert config.llms == [LLMModels.GPT3]
    assert config.priority == Priority.PERFORMANCE
    assert config.max_concurrent_calls == 5
    assert config.add_questions is False
    assert config.ignore == ["*.md", "*.txt"]
    assert config.file_prompt == "Explain this file."
    assert config.folder_prompt == "Explain this folder."
    assert config.chat_prompt == "Chat prompt here."
    assert config.content_type == "code"
    assert config.target_audience == "developers"
    assert config.link_hosted is True
    assert config.peft_model_path is None
    assert config.device == "cpu"


def test_file_summary():
    summary = FileSummary(
        file_name="test.py",
        file_path="./test.py",
        url="https://github.com/user/MyProject/test.py",
        summary="This is a test file.",
        questions="What does this file do?",
        checksum="abc123",
    )
    assert summary.file_name == "test.py"
    assert summary.file_path == "./test.py"
    assert summary.url == "https://github.com/user/MyProject/test.py"
    assert summary.summary == "This is a test file."
    assert summary.questions == "What does this file do?"
    assert summary.checksum == "abc123"


def test_process_file_params():
    params = ProcessFileParams(
        file_name="test.py",
        file_path="./test.py",
        project_name="MyProject",
        content_type="code",
        file_prompt="Explain this file.",
        target_audience="developers",
        link_hosted=False,
    )
    assert params.file_name == "test.py"
    assert params.file_path == "./test.py"
    assert params.project_name == "MyProject"
    assert params.content_type == "code"
    assert params.file_prompt == "Explain this file."
    assert params.target_audience == "developers"
    assert params.link_hosted is False


def test_folder_summary():
    folder = FolderSummary(
        folder_name="src",
        folder_path="./src",
        url="https://github.com/user/MyProject/src",
        files=[],
        folders=[],
        summary="This folder contains source code.",
        questions="What is in this folder?",
        checksum="def456",
    )
    assert folder.folder_name == "src"
    assert folder.folder_path == "./src"
    assert folder.url == "https://github.com/user/MyProject/src"
    assert folder.files == []
    assert folder.folders == []
    assert folder.summary == "This folder contains source code."
    assert folder.questions == "What is in this folder?"
    assert folder.checksum == "def456"


def test_process_folder_params():
    should_ignore_func = MagicMock(return_value=False)
    params = ProcessFolderParams(
        input_path="./",
        folder_name="src",
        folder_path="./src",
        project_name="MyProject",
        content_type="code",
        folder_prompt="Explain this folder.",
        target_audience="developers",
        link_hosted=True,
        should_ignore=should_ignore_func,
    )
    assert params.input_path == "./"
    assert params.folder_name == "src"
    assert params.folder_path == "./src"
    assert params.project_name == "MyProject"
    assert params.content_type == "code"
    assert params.folder_prompt == "Explain this folder."
    assert params.target_audience == "developers"
    assert params.link_hosted is True
    assert params.should_ignore("test") is False
    should_ignore_func.assert_called_with("test")


def test_traverse_file_system_params():
    process_file_func = MagicMock()
    process_folder_func = MagicMock()
    params = TraverseFileSystemParams(
        input_path="./",
        project_name="MyProject",
        process_file=process_file_func,
        process_folder=process_folder_func,
        ignore=["*.md"],
        file_prompt="Explain this file.",
        folder_prompt="Explain this folder.",
        content_type="code",
        target_audience="developers",
        link_hosted=False,
    )
    assert params.input_path == "./"
    assert params.project_name == "MyProject"
    assert params.process_file == process_file_func
    assert params.process_folder == process_folder_func
    assert params.ignore == ["*.md"]
    assert params.file_prompt == "Explain this file."
    assert params.folder_prompt == "Explain this folder."
    assert params.content_type == "code"
    assert params.target_audience == "developers"
    assert params.link_hosted is False


def test_llm_model_details():
    llm_mock = MagicMock()
    details = LLMModelDetails(
        name=LLMModels.GPT4,
        input_cost_per_1k_tokens=0.03,
        output_cost_per_1k_tokens=0.06,
        max_length=8192,
        llm=llm_mock,
        input_tokens=1000,
        output_tokens=500,
        succeeded=10,
        failed=0,
        total=10,
        gguf_file="path/to/gguf",
    )
    assert details.name == LLMModels.GPT4
    assert details.input_cost_per_1k_tokens == 0.03
    assert details.output_cost_per_1k_tokens == 0.06
    assert details.max_length == 8192
    assert details.llm == llm_mock
    assert details.input_tokens == 1000
    assert details.output_tokens == 500
    assert details.succeeded == 10
    assert details.failed == 0
    assert details.total == 10
    assert details.gguf_file == "path/to/gguf"

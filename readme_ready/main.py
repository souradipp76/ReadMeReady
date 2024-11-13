"""CLI interface for readme_ready project.
"""

from urllib.parse import urlparse

import questionary

from readme_ready.index import index
from readme_ready.query import query
from readme_ready.types import (
    AutodocReadmeConfig,
    AutodocRepoConfig,
    AutodocUserConfig,
    LLMModels,
)


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m readme_ready` and `$ readme_ready `.
    """
    print("Initializing Auto Documentation...")

    def url_validator(x):
        try:
            result = urlparse(x)
            return all([result.scheme, result.netloc])
        except AttributeError:
            return False

    name = questionary.text(
        message="Project Name?[Example: readme_ready]"
    ).ask()
    project_root = questionary.path(
        message="Project Root?[Example: ./readme_ready/]",
        only_directories=True,
        default=f"./{name}/",
    ).ask()
    project_url = questionary.text(
        message="Project URL?[Example: "
        + "https://github.com/username/readme_ready]",
        validate=url_validator,
    ).ask()
    output_dir = questionary.path(
        message="Output Directory?[Example: ./output/readme_ready/]",
        only_directories=True,
        default=f"./output/{name}/",
    ).ask()
    mode = questionary.select(
        message="Documentation Mode?",
        choices=["Readme", "Query"],
        default="Readme",
    ).ask()

    if mode.lower() == "readme":
        headings = questionary.text(
            message="List of Readme Headings?(comma separated)"
            + "[Example: # Introduction,## Usage]"
        ).ask()

    model_name = questionary.select(
        message="Which model?",
        choices=[
            LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,
            LLMModels.GOOGLE_GEMMA_2B_INSTRUCT_GGUF.value,
            LLMModels.LLAMA2_7B_CHAT_GPTQ.value,
            LLMModels.LLAMA2_13B_CHAT_GPTQ.value,
            LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ.value,
            LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ.value,
            LLMModels.LLAMA2_7B_CHAT_HF.value,
            LLMModels.LLAMA2_13B_CHAT_HF.value,
            LLMModels.CODELLAMA_7B_INSTRUCT_HF.value,
            LLMModels.CODELLAMA_13B_INSTRUCT_HF.value,
            LLMModels.GOOGLE_GEMMA_2B_INSTRUCT.value,
            LLMModels.GOOGLE_GEMMA_7B_INSTRUCT.value,
        ],
        default=LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,
    ).ask()
    peft = questionary.confirm(message="Is finetuned?", default=False).ask()

    peft_model_path = None
    if peft:
        peft_model_path = questionary.path(
            message="Finetuned Model Path?[Example: ./output/model/]",
            only_directories=True,
        ).ask()

    device = questionary.select(
        message="Device?", choices=["cpu", "gpu"], default="cpu"
    ).ask()

    match model_name:
        case LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value:
            model = LLMModels.TINYLLAMA_1p1B_CHAT_GGUF
        case LLMModels.GOOGLE_GEMMA_2B_INSTRUCT_GGUF.value:
            model = LLMModels.GOOGLE_GEMMA_2B_INSTRUCT_GGUF
        case LLMModels.LLAMA2_7B_CHAT_GPTQ.value:
            model = LLMModels.LLAMA2_7B_CHAT_GPTQ
        case LLMModels.LLAMA2_13B_CHAT_GPTQ.value:
            model = LLMModels.LLAMA2_13B_CHAT_GPTQ
        case LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ.value:
            model = LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ
        case LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ.value:
            model = LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ
        case LLMModels.LLAMA2_13B_CHAT_HF.value:
            model = LLMModels.LLAMA2_13B_CHAT_HF
        case LLMModels.CODELLAMA_7B_INSTRUCT_HF.value:
            model = LLMModels.CODELLAMA_7B_INSTRUCT_HF
        case LLMModels.CODELLAMA_13B_INSTRUCT_HF.value:
            model = LLMModels.CODELLAMA_13B_INSTRUCT_HF
        case LLMModels.GOOGLE_GEMMA_2B_INSTRUCT.value:
            model = LLMModels.GOOGLE_GEMMA_2B_INSTRUCT
        case LLMModels.GOOGLE_GEMMA_7B_INSTRUCT.value:
            model = LLMModels.GOOGLE_GEMMA_7B_INSTRUCT
        case _:
            model = LLMModels.LLAMA2_7B_CHAT_HF
    print("Initialization Complete.\n")

    repo_config = {
        "name": name,
        "root": project_root,
        "repository_url": project_url,
        "output": output_dir,
        "llms": [model],
        "peft_model_path": peft_model_path,
        "ignore": [
            ".*",
            "*package-lock.json",
            "*package.json",
            "node_modules",
            "*dist*",
            "*build*",
            "*test*",
            "*.svg",
            "*.md",
            "*.mdx",
            "*.toml",
        ],
        "file_prompt": """Write a detailed technical explanation of
            what this code does. Focus on the high-level
            purpose of the code and how it may be used in the
            larger project.\nInclude code examples where
            appropriate. Keep you response between 100 and 300
            words. DO NOT RETURN MORE THAN 300 WORDS.
            Output should be in markdown format.
            Do not just list the methods and classes in this file.""",
        "folder_prompt": """Write a technical explanation of what the
            code in this file does and how it might fit into the
            larger project or work with other parts of the project.
            Give examples of how this code might be used. Include code
            examples where appropriate. Be concise. Include any
            information that may be relevant to a developer who is
            curious about this code. Keep you response under
            400 words. Output should be in markdown format.
            Do not just list the files and folders in this folder.""",
        "chat_prompt": "",
        "content_type": "docs",
        "target_audience": "smart developer",
        "link_hosted": True,
        "priority": None,
        "max_concurrent_calls": 50,
        "add_questions": False,
        "device": device,
    }
    user_config = {"llms": [model]}

    repo_conf = AutodocRepoConfig(
        name=repo_config["name"],
        repository_url=repo_config["repository_url"],
        root=repo_config["root"],
        output=repo_config["output"],
        llms=repo_config["llms"],
        peft_model_path=repo_config["peft_model_path"],
        priority=repo_config["priority"],
        max_concurrent_calls=repo_config["max_concurrent_calls"],
        add_questions=repo_config["add_questions"],
        ignore=repo_config["ignore"],
        file_prompt=repo_config["file_prompt"],
        folder_prompt=repo_config["folder_prompt"],
        chat_prompt=repo_config["chat_prompt"],
        content_type=repo_config["content_type"],
        target_audience=repo_config["target_audience"],
        link_hosted=repo_config["link_hosted"],
        device=repo_config["device"],
    )

    usr_conf = AutodocUserConfig(llms=user_config["llms"])

    index.index(repo_conf)
    print("Done Indexing!!")

    if mode.lower() == "query":
        query.query(repo_conf, usr_conf)
    else:
        readme_conf = AutodocReadmeConfig(headings=headings)
        query.generate_readme(repo_conf, usr_conf, readme_conf)

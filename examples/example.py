"""
Example
"""
from readme_ready.query import query
from readme_ready.index import index
from readme_ready.types import (
    AutodocReadmeConfig,
    AutodocRepoConfig,
    AutodocUserConfig,
    LLMModels,
)

model = LLMModels.TINYLLAMA_1p1B_CHAT_GGUF # Choose model from supported models

repo_config = AutodocRepoConfig (
    name = "readmy_ready",
    root = "./readme_ready",
    repository_url = "https://github.com/souradipp76/ReadMeReady",
    output = "./output/readmy_ready",
    llms = [model],
    peft_model_path = None,
    ignore = [
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
        "*.toml"
    ],
    file_prompt = "",
    folder_prompt = "",
    chat_prompt = "",
    content_type = "docs",
    target_audience = "smart developer",
    link_hosted = True,
    priority = None,
    max_concurrent_calls = 50,
    add_questions = False,
    device = "cpu",
)

user_config = AutodocUserConfig(
    llms = [model]
)

readme_config = AutodocReadmeConfig(
    headings = "# Description, # Requirements, # Installation, # Usage, # Contributing, # License"
)

index.index(repo_config)
query.generate_readme(repo_config, user_config, readme_config)
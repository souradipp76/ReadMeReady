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

# Choose model from supported models
model = LLMModels.LLAMA2_7B_CHAT_GPTQ

# Initialize the repository configuration. `root` refers to the path to the
# code repository for which you want to generate a README for. Please download
# any code repository from GitHub and use that or if you have your own
# repository downloaded (say 'MyRepo') you can use that as well.
# Set `name` to the 'MyRepo'.
# Set `root` as <path to 'MyRepo'>.
# Set `repository_url` to the GitHub URL of 'MyRepo' (if any) else leave blank.
# Set `output` as the path to the directory where the README and other metadata
# will be generated and saved.
# Set other parameters accordingly (or leave as default).

repo_config = AutodocRepoConfig (
    name = "readmy_ready", # Set repository name
    root = "./readme_ready", # Set path to root directory of the repository
    repository_url = "https://github.com/souradipp76/ReadMeReady", # Set url
    output = "./output/readmy_ready", # Set path to output directory to save
    llms = [model],
    peft_model_path = None, # Set path to PEFT model
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
    device = "auto", # Select device "cpu" or "auto"
)

user_config = AutodocUserConfig(
    llms = [model]
)

readme_config = AutodocReadmeConfig(
    # Set comma separated list of README headings
    headings = "Description,Requirements,Installation,Usage,Contributing,License"
)

index.index(repo_config)
query.generate_readme(repo_config, user_config, readme_config)
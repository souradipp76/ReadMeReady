# ReadmeReady

[![codecov](https://codecov.io/gh/souradipp76/ReadMeReady/branch/main/graph/badge.svg?token=49620380-3fe7-4eb1-8dbb-3457febc6f78)](https://codecov.io/gh/souradipp76/ReadMeReady)
[![CI](https://github.com/souradipp76/ReadMeReady/actions/workflows/main.yml/badge.svg)](https://github.com/souradipp76/ReadMeReady/actions/workflows/main.yml)

Auto-generate code documentation in Markdown format in seconds.

## What is ReadmeReady?

Automated documentation of programming source code is a challenging task with significant practical and scientific implications for the developer community. ReadmeReady is a large language model (LLM)-based application that developers can use as a support tool to generate basic documentation for any publicly available or custom repository. Over the last decade, several research have been done on generating documentation for source code using neural network architectures. With the recent advancements in LLM technology, some open-source applications have been developed to address this problem. However, these applications typically rely on the OpenAI APIs, which incur substantial financial costs, particularly for large repositories. Moreover, none of these open-source applications offer a fine-tuned model or features to enable users to fine-tune custom LLMs. Additionally, finding suitable data for fine-tuning is often challenging. Our application addresses these issues.

## Installation

ReadmeReady is available only on Linux/Windows.

### Dependencies

Please follow the installation guide [here](https://pypi.org/project/python-magic/) to install `python-magic`.

### Install it from PyPI

The simplest way to install ReadmeReady and its dependencies is from PyPI with pip, Python's preferred package installer.

```bash
pip install readme_ready
```

In order to upgrade ReadmeReady to the latest version, use pip as follows.

```bash
$ pip install -U readme_ready
```

### Install it from source

You can also install ReadmeReady from source as follows.

```bash
$ git clone https://github.com/souradipp76/ReadMeReady.git
$ cd ReadMeReady
$ make install
```

To create a virtual environment before installing ReadmeReady, you can use the command:
```bash
$ make virtualenv
$ source .venv/bin/activate
```

## Usage

### Initialize
```bash
$ export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
$ export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

Set `OPENAI_API_KEY=dummy` to use only open-source models.

### Command-Line

```bash
$ python -m readme_ready
#or
$ readme_ready
```

### In Code

```py
from readme_ready.query import query
from readme_ready.index import index
from readme_ready.types import (
    AutodocReadmeConfig,
    AutodocRepoConfig,
    AutodocUserConfig,
    LLMModels,
)

model = LLMModels.LLAMA2_7B_CHAT_GPTQ # Choose model from supported models

repo_config = AutodocRepoConfig (
    name = "<REPOSITORY_NAME>", # Replace <REPOSITORY_NAME>
    root = "<REPOSITORY_ROOT_DIR_PATH>", # Replace <REPOSITORY_ROOT_DIR_PATH>
    repository_url = "<REPOSITORY_URL>", # Replace <REPOSITORY_URL>
    output = "<OUTPUT_DIR_PATH>", # Replace <OUTPUT_DIR_PATH>
    llms = [model],
    peft_model_path = "<PEFT_MODEL_NAME_OR_PATH>", # Replace <PEFT_MODEL_NAME_OR_PATH>
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
```

Run the sample script in the `examples/example.py` to see a typical code usage. See example on Google Colab: <a src="https://colab.research.google.com/assets/colab-badge.svg" href="https://colab.research.google.com/github/souradipp76/ReadMeReady/blob/main/examples/example.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>

See detailed API references [here](https://souradipp76.github.io/ReadMeReady/reference/).

### Finetuning

For finetuning on custom datasets, follow the instructions below.

- Run the notebook file `scripts/data.ipynb` and follow the instructions in the file to generate custom dataset from open-source repositories.
- Run the notebook file `scripts/fine-tuning-with-llama2-qlora.ipynb` and follow the instructions in the file to finetune custom LLMs.

### Validation

Run the script `scripts/run_validate.sh` to generate BLEU and BERT scores for 5 sample repositories comparing the actual README file with the generated ones. Note that to reproduce the scores, a GPU with 16GB or more is required.

```bash
$ chmod +x scripts/run_validate.sh
$ scripts/run_validate.sh
```

### Supported models
- TINYLLAMA_1p1B_CHAT_GGUF (`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`)
- GOOGLE_GEMMA_2B_INSTRUCT_GGUF (`bartowski/gemma-2-2b-it-GGUF`)
- LLAMA2_7B_CHAT_GPTQ (`TheBloke/Llama-2-7B-Chat-GPTQ`)
- LLAMA2_13B_CHAT_GPTQ (`TheBloke/Llama-2-13B-Chat-GPTQ`)
- CODELLAMA_7B_INSTRUCT_GPTQ (`TheBloke/CodeLlama-7B-Instruct-GPTQ`)
- CODELLAMA_13B_INSTRUCT_GPTQ (`TheBloke/CodeLlama-13B-Instruct-GPTQ`)
- LLAMA2_7B_CHAT_HF (`meta-llama/Llama-2-7b-chat-hf`)
- LLAMA2_13B_CHAT_HF (`meta-llama/Llama-2-13b-chat-hf`)
- CODELLAMA_7B_INSTRUCT_HF (`meta-llama/CodeLlama-7b-Instruct-hf`)
- CODELLAMA_13B_INSTRUCT_HF (`meta-llama/CodeLlama-13b-Instruct-hf`)
- GOOGLE_GEMMA_2B_INSTRUCT (`google/gemma-2b-it`)
- GOOGLE_GEMMA_7B_INSTRUCT (`google/gemma-7b-it`)
- GOOGLE_CODEGEMMA_2B (`google/codegemma-2b`)
- GOOGLE_CODEGEMMA_7B_INSTRUCT (`google/codegemma-7b-it`)

## Contributing

ReadmeReady is an open-source project that is supported by a community who will gratefully and humbly accept any contributions you might make to the project.

If you are interested in contributing, read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

- Submit a bug report or feature request on [GitHub Issues](https://github.com/souradipp76/ReadMeReady/issues).
- Add to the documentation or help with our website.
- Write unit or integration tests for our project under the `tests` directory.
- Answer questions on our issues, mailing list, Stack Overflow, and elsewhere.
- Write a blog post, tweet, or share our project with others.

As you can see, there are lots of ways to get involved, and we would be very happy for you to join us!

## License

Read the [LICENSE](LICENSE) file.

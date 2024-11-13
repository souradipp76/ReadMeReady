# ReadmeReady

[![codecov](https://codecov.io/gh/souradipp76/ReadMeReady/branch/main/graph/badge.svg?token=49620380-3fe7-4eb1-8dbb-3457febc6f78)](https://codecov.io/gh/souradipp76/ReadMeReady)
[![CI](https://github.com/souradipp76/ReadMeReady/actions/workflows/main.yml/badge.svg)](https://github.com/souradipp76/ReadMeReady/actions/workflows/main.yml)

Auto-generate code documentation in Markdown format in seconds.

## What is ReadmeReady?

Automated documentation of programming source code is a challenging task with significant practical and scientific implications for the developer community. ReadmeReady is a large language model (LLM)-based application that developers can use as a support tool to generate basic documentation for any publicly available or custom repository. Over the last decade, several research have been done on generating documentation for source code using neural network architectures. With the recent advancements in LLM technology, some open-source applications have been developed to address this problem. However, these applications typically rely on the OpenAI APIs, which incur substantial financial costs, particularly for large repositories. Moreover, none of these open-source applications offer a fine-tuned model or features to enable users to fine-tune custom LLMs. Additionally, finding suitable data for fine-tuning is often challenging. Our application addresses these issues.

## Installation
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

### Command-Line

```bash
$ python -m readme_ready
#or
$ readme_ready
```

### In Code

```py
from readme_ready import query
from readme_ready import index

repo_config = {
    "name": <NAME>, # Replace <NAME>
    "root": <PROJECT_ROOT>, # Replace <PROJECT_ROOT>
    "repository_url": <PROJECT_URL>, # Replace <PROJECT_URL>
    "output": <OUTPUT_DIR>, # Replace <OUTPUT_DIR>
    "llms": [<MODEL_NAME_OR_PATH>], # Replace <MODEL_NAME_OR_PATH>
    "peft_model_path": <PEFT_MODEL_NAME_OR_PATH>, # Replace <PEFT_MODEL_NAME_OR_PATH>
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
        "*.toml"
    ],
    "file_prompt": "",
    "folder_prompt": "",
    "chat_prompt": "",
    "content_type": "docs",
    "target_audience": "smart developer",
    "link_hosted": True,
    "priority": None,
    "max_concurrent_calls": 50,
    "add_questions": False
}

user_config = {
    "llms": [model]
}
index.index(**repo_config)
query.generate_readme(**repo_config, **user_config)
```

### Finetuning

For finetuning on custom datasets, follow the instructions below.

- Run the notebook file `scripts/data.ipynb` and follow the instructions in the file to generate custom dataset from open-source repositories.
- Run the notebook file `scripts/fine-tuning-with-llama2-qlora.ipynb` and follow the instructions in the file to finetune custom LLMs.

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

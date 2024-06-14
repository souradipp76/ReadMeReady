<h1 align="center"> ReadMeReady</h1>

<p align="center">
  Generate code documentation in seconds.
</p>

[![codecov](https://codecov.io/gh/souradipp76/doc_generator/branch/main/graph/badge.svg?token=doc_generator_token_here)](https://codecov.io/gh/souradipp76/doc_generator)
[![CI](https://github.com/souradipp76/doc_generator/actions/workflows/main.yml/badge.svg)](https://github.com/souradipp76/doc_generator/actions/workflows/main.yml)

## Installation
### Install it from PyPI

```bash
pip install doc_generator
```

### Install it from source

```bash
$ git clone https://github.com/souradipp76/ReadMeReady.git
$ cd ReadMeReady
$ make install
```

## Usage

### Initialize
```bash
$ export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
$ export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

### Command-Line

```bash
$ python -m doc_generator
#or
$ doc_generator
```

### In Code

```py
from doc_generator import query
from doc_generator import index

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

## Contributing

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

Read the [LICENSE.md](LICENSE.md) file.
# doc_generator

[![codecov](https://codecov.io/gh/souradipp76/doc_generator/branch/main/graph/badge.svg?token=doc_generator_token_here)](https://codecov.io/gh/souradipp76/doc_generator)
[![CI](https://github.com/souradipp76/doc_generator/actions/workflows/main.yml/badge.svg)](https://github.com/souradipp76/doc_generator/actions/workflows/main.yml)

Awesome doc_generator created by souradipp76

## Install it from PyPI

```bash
pip install doc_generator
```

## Usage

```py
from doc_generator import query
from doc_generator import index

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
        "*.toml"
    ],
    "file_prompt": "Write a detailed technical explanation of \
        what this code does. \n      Focus on the high-level \
        purpose of the code and how it may be used in the \
        larger project.\n      Include code examples where \
        appropriate. Keep you response between 100 and 300 \
        words. \n      DO NOT RETURN MORE THAN 300 WORDS.\n \
        Output should be in markdown format.\n \
        Do not just list the methods and classes in this file.",
    "folder_prompt": "Write a technical explanation of what the \
        code in this file does\n      and how it might fit into the \
        larger project or work with other parts of the project.\n      \
        Give examples of how this code might be used. Include code \
        examples where appropriate.\n      Be concise. Include any \
        information that may be relevant to a developer who is \
        curious about this code.\n      Keep you response under \
        400 words. Output should be in markdown format.\n      \
        Do not just list the files and folders in this folder.",
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
index.index(AutodocRe**repo_config)
query.generate_readme(**repo_config, **user_config)
```

```bash
$ python -m doc_generator
#or
$ doc_generator
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

"""CLI interface for doc_generator project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""
from doc_generator.query import query


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m doc_generator` and `$ doc_generator `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    Examples:
        * Run a test suite
        * Run a server
        * Do some other stuff
        * Run a command line application (Click, Typer, ArgParse)
        * List all available tasks
        * Run an application (Flask, FastAPI, Django, etc.)
    """
    # Example config objects, these need to be defined or imported properly
    repo_config = {
        "name": "autodoc",
        "repository_url": "https://github.com/context-labs/autodoc",
        "output": "doc_generator/autodoc",
        "content_type": "docs",
        "chat_prompt": "Additional instructions here",
        "target_audience": "developers"
    }
    user_config = {
        "llms": ["gpt-3.5-turbo"]
    }
    query(**repo_config, **user_config)

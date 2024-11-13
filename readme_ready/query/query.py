"""
Query
"""

import os
import traceback

from markdown2 import markdown
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import clear

from readme_ready.query.create_chat_chain import (
    make_qa_chain,
    make_readme_chain,
)
from readme_ready.types import (
    AutodocReadmeConfig,
    AutodocRepoConfig,
    AutodocUserConfig,
)
from readme_ready.utils.HNSWLib import HNSWLib
from readme_ready.utils.llm_utils import get_embeddings

chat_history: list[tuple[str, str]] = []


def display_welcome_message(project_name):
    """Display Welcome Message"""
    print(f"Welcome to the {project_name} chatbot.")
    print(
        f"Ask any questions related to the {project_name} codebase, "
        + "and I'll try to help. Type 'exit' to quit.\n"
    )


def init_qa_chain(
    repo_config: AutodocRepoConfig, user_config: AutodocUserConfig
):
    data_path = os.path.join(repo_config.output, "docs", "data")
    embeddings = get_embeddings(repo_config.llms[0].value, repo_config.device)
    vector_store = HNSWLib.load(data_path, embeddings)
    chain = make_qa_chain(
        repo_config.name,
        repo_config.repository_url,
        repo_config.content_type,
        repo_config.chat_prompt,
        repo_config.target_audience,
        vector_store,
        user_config.llms,
        on_token_stream=user_config.streaming,
    )

    return chain


def init_readme_chain(
    repo_config: AutodocRepoConfig, user_config: AutodocUserConfig
):
    data_path = os.path.join(repo_config.output, "docs", "data")
    embeddings = get_embeddings(repo_config.llms[0].value, repo_config.device)
    vector_store = HNSWLib.load(data_path, embeddings)
    chain = make_readme_chain(
        repo_config.name,
        repo_config.repository_url,
        repo_config.content_type,
        repo_config.chat_prompt,
        repo_config.target_audience,
        vector_store,
        user_config.llms,
        repo_config.peft_model_path,
        on_token_stream=user_config.streaming,
    )

    return chain


def query(repo_config: AutodocRepoConfig, user_confg: AutodocUserConfig):
    """Query"""
    chain = init_qa_chain(repo_config, user_confg)

    clear()
    display_welcome_message(repo_config.name)

    while True:
        question = prompt(f"How can I help with {repo_config.name}?\n")
        if question.strip().lower() == "exit":
            break

        print("Thinking...")
        try:
            response = chain.invoke(
                {"input": question, "chat_history": chat_history}
            )
            chat_history.append((question, response["answer"]))
            print("\n\nMarkdown:\n")
            print(markdown(response["answer"]))
        except RuntimeError as error:
            print(f"Something went wrong: {error}")
            traceback.print_exc()


def generate_readme(
    repo_config: AutodocRepoConfig,
    user_config: AutodocUserConfig,
    readme_config: AutodocReadmeConfig,
):
    """Generate README"""
    chain = init_readme_chain(repo_config, user_config)

    clear()

    print("Generating README...")
    data_path = os.path.join(repo_config.output, "docs", "data")
    readme_path = os.path.join(
        data_path, f"README_{repo_config.llms[0].name}.md"
    )
    with open(readme_path, "w", encoding="utf-8") as file:
        file.write(f"# {repo_config.name}")

    with open(readme_path, "a", encoding="utf-8") as file:
        headings = readme_config.headings
        for heading in headings:
            question = f"{heading}"
            try:
                response = chain.invoke({"input": question})
                print("\n\nMarkdown:\n")
                print(markdown(response["answer"]))
                file.write(markdown(response["answer"]))
            except RuntimeError as error:
                print(f"Something went wrong: {error}")
                traceback.print_exc()

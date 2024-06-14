"""
Query
"""
import os
import traceback

from markdown2 import markdown
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import clear

from doc_generator.query.create_chat_chain import (make_qa_chain,
                                                   make_readme_chain)
from doc_generator.types import AutodocRepoConfig, AutodocUserConfig
from doc_generator.utils.HNSWLib import HNSWLib
from doc_generator.utils.llm_utils import get_embeddings

chat_history: list[tuple[str, str]] = []


def display_welcome_message(project_name):
    """Display Welcome Message"""
    print(f"Welcome to the {project_name} chatbot.")
    print(f"Ask any questions related to the {project_name} codebase, \
          and I'll try to help. Type 'exit' to quit.\n")


def query(repo_config: AutodocRepoConfig, user_confg: AutodocUserConfig):
    """Query"""
    data_path = os.path.join(repo_config.output, 'docs', 'data')
    embeddings = get_embeddings(repo_config.llms[0].value)
    vector_store = HNSWLib.load(data_path, embeddings)
    chain = make_qa_chain(repo_config.name,
                          repo_config.repository_url,
                          repo_config.content_type,
                          repo_config.chat_prompt,
                          repo_config.target_audience,
                          vector_store,
                          user_confg.llms)

    clear()
    display_welcome_message(repo_config.name)

    while True:
        question = prompt(f"How can I help with {repo_config.name}?\n")
        if question.strip().lower() == 'exit':
            break

        print('Thinking...')
        try:
            response = chain.invoke(
                {
                    'question': question,
                    'chat_history': chat_history
                })
            chat_history.append((question, response['answer']))
            print('\n\nMarkdown:\n')
            print(markdown(response['answer']))
        except RuntimeError as error:
            print(f"Something went wrong: {error}")
            traceback.print_exc()


def generate_readme(repo_config: AutodocRepoConfig,
                    user_confg: AutodocUserConfig):
    """Generate README"""
    data_path = os.path.join(repo_config.output, 'docs', 'data')
    embeddings = get_embeddings(repo_config.llms[0].value)
    vector_store = HNSWLib.load(data_path, embeddings)
    chain = make_readme_chain(repo_config.name,
                              repo_config.repository_url,
                              repo_config.content_type,
                              repo_config.chat_prompt,
                              repo_config.target_audience,
                              vector_store,
                              user_confg.llms,
                              repo_config.peft_model_path)

    clear()

    print('Generating README...')
    readme_path = os.path.join(data_path,
                               f"README_{repo_config.llms[0].name}.md")
    with open(readme_path, "w", encoding='utf-8') as file:
        file.write(f"# {repo_config.name}")

    with open(readme_path, "a", encoding='utf-8') as file:
        headings = [
            "## NAME",
            "## DESCRIPTION",
            "## USAGE",
            "## INSTALLATION",
            "### REQUIREMENTS",
            "### MANUAL",
            "### AUTOMATIC",
            "#### Linux",
            "#### OS X",
            "## Windows",
            "## KNOWN ISSUES",
            "## REPORTING BUGS",
            "## AUTHORS",
            "## COPYRIGHT"
            ]
        for heading in headings:
            question = f"{heading}"
            try:
                response = chain.invoke({'input': question})
                print('\n\nMarkdown:\n')
                print(markdown(response["answer"]))
                file.write(markdown(response["answer"]))
            except RuntimeError as error:
                print(f"Something went wrong: {error}")
                traceback.print_exc()

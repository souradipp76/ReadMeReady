"""
In the updated code, I've added the type annotation chat_history: list[tuple[str, str]] = [] on line 13. 
This annotation specifies that chat_history is a list of tuples, where each tuple contains two strings. 
The first string represents the question, and the second string represents the answer.
"""

import os
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import clear
from markdown2 import markdown

from doc_generator.types import AutodocRepoConfig, AutodocUserConfig
from doc_generator.utils.HNSWLib import HNSWLib
from doc_generator.query.createChatChain import make_qa_chain, make_readme_chain
from doc_generator.utils.LLMUtils import get_embeddings

import traceback

chat_history: list[tuple[str, str]] = []  # Type annotation for chat_history


def display_welcome_message(project_name):
    print(f"Welcome to the {project_name} chatbot.")
    print(f"Ask any questions related to the {project_name} codebase, and I'll try to help. Type 'exit' to quit.\n")


def query(repo_config: AutodocRepoConfig, user_confg: AutodocUserConfig):
    data_path = os.path.join(repo_config.output, 'docs', 'data')
    embeddings = get_embeddings(repo_config.llms[0])
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
            response = chain.invoke({'question': question, 'chat_history': chat_history})
            chat_history.append((question, response['answer']))
            print('\n\nMarkdown:\n')
            print(markdown(response['answer']))
        except Exception as error:
            print(f"Something went wrong: {error}")
            traceback.print_exc()


def generate_readme(repo_config: AutodocRepoConfig, user_confg: AutodocUserConfig):
    data_path = os.path.join(repo_config.output, 'docs', 'data')
    embeddings = get_embeddings(repo_config.llms[0])
    vector_store = HNSWLib.load(data_path, embeddings)
    chain = make_readme_chain(repo_config.name,
                              repo_config.repository_url,
                              repo_config.content_type,
                              repo_config.chat_prompt,
                              repo_config.target_audience,
                              vector_store,
                              user_confg.llms)

    clear()

    print('Generating README...')
    with open(os.path.join(data_path, "README.md"), "w", encoding='utf-8') as file:
        file.write(f"# {repo_config.name}")
    
    with open(os.path.join(data_path, "README.md"), "a", encoding='utf-8') as file:
        headings = ["Description", "Requirements", "Installation", "Usage", "Contributing", "License"]
        for heading in headings:
            question = f"Provide the README content for the section with heading \"{heading}\" starting with ## {heading}."
            try:
                response = chain.invoke({'input': question})
                print('\n\nMarkdown:\n')
                print(response["answer"])
                file.write(markdown(response["answer"]))
            except Exception as error:
                print(f"Something went wrong: {error}")
                traceback.print_exc()
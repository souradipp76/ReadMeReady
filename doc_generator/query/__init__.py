import os
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import clear
from markdown2 import markdown

from doc_generator.types import AutodocRepoConfig, AutodocUserConfig
from doc_generator.utils.HNSWLib import HNSWLib
from doc_generator.utils.createChatChain import make_chain
from doc_generator.utils.LLMUtils import get_embeddings

import traceback

chat_history = []

def display_welcome_message(project_name):
    print(f"Welcome to the {project_name} chatbot.")
    print(f"Ask any questions related to the {project_name} codebase, and I'll try to help. Type 'exit' to quit.\n")


def query(repo_config: AutodocRepoConfig, user_confg: AutodocUserConfig):
    data_path = os.path.join(repo_config.output, 'docs', 'data')
    embeddings = get_embeddings(repo_config.llms[0])
    vector_store = HNSWLib.load(data_path, embeddings)
    chain = make_chain(repo_config.name,
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
            chat_history.append((question, response['text']))
            print('\n\nMarkdown:\n')
            print(markdown(response['text']))
        except Exception as error:
            print(f"Something went wrong: {error}")
            traceback.print_exc()

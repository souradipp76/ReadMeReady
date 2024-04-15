from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import clear
import os
from langchain_community.embeddings import OpenAIEmbeddings
from HNSWLib import HNSWLib
from markdown2 import markdown
from createChatChain import make_chain

chat_history = []

def display_welcome_message(project_name):
    print(f"Welcome to the {project_name} chatbot.")
    print(f"Ask any questions related to the {project_name} codebase, and I'll try to help. Type 'exit' to quit.\n")

def query(name, repository_url, output, content_type, chat_prompt, target_audience, llms):
    data_path = os.path.join(output, 'docs', 'data')
    embeddings = OpenAIEmbeddings()
    vector_store = HNSWLib.load(data_path, embeddings)
    chain = make_chain(name, repository_url, content_type, chat_prompt, target_audience, vector_store, llms)

    clear()
    display_welcome_message(name)

    while True:
        question = prompt(f"How can I help with {name}?\n")
        if question.strip().lower() == 'exit':
            break

        print('Thinking...')
        try:
            response = chain.call({'question': question, 'chat_history': chat_history})
            chat_history.append((question, response['text']))
            print('\n\nMarkdown:\n')
            print(markdown(response['text']))
        except Exception as error:
            print(f"Something went wrong: {error}")

if __name__ == "__main__":
    repo_config = {
        "name": "autodoc",
        "repository_url": "https://github.com/context-labs/autodoc",
        "output": "./.autodoc",
        "content_type": "docs",
        "chat_prompt": "Additional instructions here",
        "target_audience": "developers"
    }
    user_config = {
        "llms": ["gpt-3.5-turbo"]
    }
    query(**repo_config, **user_config)

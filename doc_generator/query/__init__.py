import os
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import clear
from langchain_openai import OpenAIEmbeddings
from markdown2 import markdown

from doc_generator.utils.HNSWLib import HNSWLib
from doc_generator.utils.createChatChain import make_chain

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
            response = chain({'question': question, 'chat_history': chat_history})
            chat_history.append((question, response['text']))
            print('\n\nMarkdown:\n')
            print(markdown(response['text']))
        except Exception as error:
            print(f"Something went wrong: {error}")

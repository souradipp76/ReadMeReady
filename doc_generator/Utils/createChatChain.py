from langchain.chains import LLMChain, ChatVectorDBChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


# Define the prompt template for condensing the follow-up question
condense_prompt = PromptTemplate.from_template(
    "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\n"
             "Chat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
)

def make_qa_prompt(project_name, repository_url, content_type, chat_prompt, target_audience):
    additional_instructions = f"\nHere are some additional instructions for answering questions about {content_type}:\n{chat_prompt}" if chat_prompt else ""
    return PromptTemplate.from_template(
        f"You are an AI assistant for a software project called {project_name}. You are trained on all the {content_type} that makes up this project.\n"
                 f"The {content_type} for the project is located at {repository_url}.\n"
                 "You are given the following extracted parts of a technical summary of files in a {content_type} and a question. "
                 "Provide a conversational answer with hyperlinks back to GitHub.\n"
                 "You should only use hyperlinks that are explicitly listed in the context. Do NOT make up a hyperlink that is not listed.\n"
                 "Include lots of {content_type} examples and links to the {content_type} examples, where appropriate.\n"
                 "Assume the reader is a {target_audience} but is not deeply familiar with {project_name}.\n"
                 "Assume the reader does not know anything about how the project is structured or which folders/files are provided in the context.\n"
                 "Do not reference the context in your answer. Instead use the context to inform your answer.\n"
                 "If you don't know the answer, just say \"Hmm, I'm not sure.\" Don't try to make up an answer.\n"
                 "If the question is not about the {project_name}, politely inform them that you are tuned to only answer questions about the {project_name}.\n"
                 "Your answer should be at least 100 words and no more than 300 words.\n"
                 "Do not include information that is not directly relevant to the question, even if the context includes it.\n"
                 "Always include a list of reference links to GitHub from the context. Links should ONLY come from the context.\n"
                 f"{additional_instructions}\n"
                 "Question: {question}\n\n"
                 "Context:\n{context}\n\n"
                 "Answer in Markdown:"
    )

def make_chain(project_name, repository_url, content_type, chat_prompt, target_audience, vectorstore, llms, on_token_stream=None):
    llm = llms[1] if len(llms) > 1 else llms[0]
    question_generator = LLMChain(
        llm=ChatOpenAI(temperature=0.1, model_name=llm),
        prompt=condense_prompt
    )

    qa_prompt = make_qa_prompt(project_name, repository_url, content_type, chat_prompt, target_audience)
    doc_chain = load_qa_chain(
        llm=ChatOpenAI(temperature=0.2,
                       model_name=llm,
                       streaming=bool(on_token_stream),
                       model_kwargs={
                            "frequency_penalty": 0.0,
                            "presence_penalty": 0.0,
                        }),
        prompt=qa_prompt
    )

    return ChatVectorDBChain(
    vectorstore=vectorstore,
    combine_docs_chain=doc_chain,
    question_generator=question_generator
)

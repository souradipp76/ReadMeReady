from langchain.chains.conversational_retrieval.base import ChatVectorDBChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from doc_generator.utils.LLMUtils import get_llama_chat_model, get_openai_chat_model

# Define the prompt template for condensing the follow-up question
condense_qa_prompt = PromptTemplate.from_template(
    template="<s>[INST]Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\n"
                "Chat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:[/INST]")

condense_readme_prompt = PromptTemplate.from_template(
    template="<s>[INST]Given the following question, rephrase the question to be a standalone question.\n\n"
                "Input: {question}\nStandalone question:[/INST]")

def make_qa_prompt(project_name, repository_url, content_type, chat_prompt, target_audience):
    additional_instructions = f"\nHere are some additional instructions for answering questions about {content_type}:\n{chat_prompt}" if chat_prompt else ""
    template = f"""<s>[INST]You are an AI assistant for a software project called {project_name}. You are trained on all the {content_type} that makes up this project.
        The {content_type} for the project is located at {repository_url}.
        You are given the following extracted parts of a technical summary of files in a {content_type} and a question.
        Provide a conversational answer with hyperlinks back to GitHub.
        You should only use hyperlinks that are explicitly listed in the context. Do NOT make up a hyperlink that is not listed.
        Include lots of {content_type} examples and links to the {content_type} examples, where appropriate.

        Assume the reader is a {target_audience} but is not deeply familiar with {project_name}.
        Assume the reader does not know anything about how the project is structured or which folders/files do what and what functions are written in which files and what these functions do.
        If you don't know the answer, just say \"Hmm, I'm not sure.\" Don't try to make up an answer.
        If the question is not about the {project_name}, politely inform them that you are tuned to only answer questions about the {project_name}.
        Your answer should be at least 100 words and no more than 300 words.
        Do not include information that is not directly relevant to repository, even though the names of the functions might be common or is frequently used in several other places.
        Always include a list of reference links to GitHub from the context. Links should ONLY come from the context.

        {additional_instructions}
        Question: {{question}}
        Context: 
        {{context}}
        Answer in Markdown:
        [/INST]"""

    return PromptTemplate(template=template, input_variables=["question", "context"])


def make_readme_prompt(project_name, repository_url, content_type, chat_prompt, target_audience):
    additional_instructions = f"\nHere are some additional instructions for generating readme content about {content_type}:\n{chat_prompt}" if chat_prompt else ""
    template = f"""<s>[INST]You are an AI assistant for a software project called {project_name}. You are trained on all the {content_type} that makes up this project.
    The {content_type} for the project is located at {repository_url}.
    You are given a repository which might contain several modules and each module will contain a set of files.
    Look at the source code in the repository and you have to generate content for the section of a README.md file following the heading given below. If you use any hyperlinks, they should link back to the github repository shared with you.
    You should only use hyperlinks that are explicitly listed in the context. Do NOT make up a hyperlink that is not listed.

    Assume the reader is a {target_audience} but is not deeply familiar with {project_name}.
    Assume the reader does not know anything about how the project is structured or which folders/files do what and what functions are written in which files and what these functions do.
    If you don't know how to fill up the readme.md file in one of its sections, leave that part blank. Don't try to make up any content.
    Do not include information that is not directly relevant to repository, even though the names of the functions might be common or is frequently used in several other places.

    {additional_instructions}
    Heading: {{question}}
    Context:
    {{context}}

    Answer in Markdown:
    [/INST]"""

    # Return a template object instead of string if you have a class handling it
    return PromptTemplate(template=template, input_variables=["question", "context"])


def make_qa_chain(project_name, repository_url, content_type, chat_prompt, target_audience, vectorstore, llms, on_token_stream=None):
    llm = llms[1] if len(llms) > 1 else llms[0]
    question_chat_model = None
    doc_chat_model = None
    if "llama" in llm.lower():
        question_chat_model = get_llama_chat_model(llm, {"temperature": 0.1})
    else:
        question_chat_model = get_openai_chat_model(llm, temperature=0.1)
    question_generator = LLMChain(
        llm=question_chat_model,
        prompt=condense_qa_prompt
    )

    if "llama" in llm.lower():
        doc_chat_model = get_llama_chat_model(llm, {"temperature": 0.2})
    else:
        doc_chat_model = get_openai_chat_model(llm,
                                               temperature=0.2,
                                               streaming=bool(on_token_stream),
                                               model_kwargs={
                                                    "frequency_penalty": 0.0,
                                                    "presence_penalty": 0.0,
                                                })

    qa_prompt = make_qa_prompt(project_name, repository_url, content_type, chat_prompt, target_audience)
    doc_chain = load_qa_chain(
        llm=doc_chat_model,
        prompt=qa_prompt
    )

    return ChatVectorDBChain(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator
    )


def make_readme_chain(project_name, repository_url, content_type, chat_prompt, target_audience, vectorstore, llms, on_token_stream=None):
    llm = llms[1] if len(llms) > 1 else llms[0]

    question_chat_model = None
    doc_chat_model = None
    if "llama" in llm.lower():
        question_chat_model = get_llama_chat_model(llm, {"temperature": 0.1})
    else:
        question_chat_model = get_openai_chat_model(llm, temperature=0.1)
    question_generator = LLMChain(
        llm=question_chat_model,
        prompt=condense_readme_prompt
    )

    if "llama" in llm.lower():
        doc_chat_model = get_llama_chat_model(llm, {"temperature": 0.2})
    else:
        doc_chat_model = get_openai_chat_model(llm,
                                               temperature=0.2,
                                               streaming=bool(on_token_stream),
                                               model_kwargs={
                                                    "frequency_penalty": 0.0,
                                                    "presence_penalty": 0.0,
                                                })

    readme_prompt = make_readme_prompt(project_name, repository_url, content_type, chat_prompt, target_audience)
    doc_chain = load_qa_chain(
        llm=doc_chat_model,
        prompt=readme_prompt
    )

    return ChatVectorDBChain(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator
    )
"""
Create Chat Chain
"""

from typing import List

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents.stuff import (
    create_stuff_documents_chain,
)
from langchain.prompts import PromptTemplate

from readme_ready.types import LLMModels
from readme_ready.utils.llm_utils import (
    get_gemma_chat_model,
    get_llama_chat_model,
    get_openai_chat_model,
    models,
)

# Define the prompt template for condensing the follow-up question
condense_qa_prompt = PromptTemplate.from_template(
    template="Given the following conversation and a follow up "
    + "question, rephrase the follow up question to be a standalone "
    + "question.\n\n"
    + "Chat History:\n{chat_history}\nFollow Up Input: {input}\n\
        Standalone question:"
)

condense_readme_prompt = PromptTemplate.from_template(
    template="Given the following question, rephrase the question "
    + "to be a standalone question.\n\n"
    + "Input: {input}\nStandalone question:"
)


def make_qa_prompt(
    project_name, repository_url, content_type, chat_prompt, target_audience
):
    """Make QA Prompt"""
    additional_instructions = (
        "\nHere are some additional instructions for "
        + f"answering questions about {content_type}:\n"
        + f"{chat_prompt}"
        if chat_prompt
        else ""
    )
    template = f"""You are an AI assistant for a software project
        called {project_name}. You are trained on all the {content_type}
        that makes up this project.
        The {content_type} for the project is located at {repository_url}.
        You are given the following extracted parts of a technical summary
        of files in a {content_type} and a question.
        Provide a conversational answer with hyperlinks back to GitHub.
        You should only use hyperlinks that are explicitly listed in the
        context. Do NOT make up a hyperlink that is not listed.
        Include lots of {content_type} examples and links to the
        {content_type} examples, where appropriate.

        Assume the reader is a {target_audience} but is not deeply familiar
        with {project_name}.
        Assume the reader does not know anything about how the project is
        structured or which folders/files do what and what functions are
        written in which files and what these functions do.
        If you don't know the answer, just say \"Hmm, I'm not sure.\" Don't
        try to make up an answer.
        If the question is not about the {project_name}, politely inform them
        that you are tuned to only answer questions about the {project_name}.
        Your answer should be at least 100 words and no more than 300 words.
        Do not include information that is not directly relevant to
        repository, even though the names of the functions might be common or
        is frequently used in several other places.
        Always include a list of reference links to GitHub from the context.
        Links should ONLY come from the context.

        {additional_instructions}
        Question: {{input}}
        Context:
        {{context}}
        Answer in Markdown:
        """

    return PromptTemplate(
        template=template, input_variables=["input", "context"]
    )


def make_readme_prompt(
    project_name, repository_url, content_type, chat_prompt, target_audience
):
    """Make Readme Prompt"""
    additional_instructions = (
        "\nHere are some additional instructions for "
        + f"generating readme content about {content_type}:\n"
        + f"{chat_prompt}"
        if chat_prompt
        else ""
    )
    template = f"""You are an AI assistant for a software project called
    {project_name}. You are trained on all the {content_type} that makes up
    this project.
    The {content_type} for the project is located at {repository_url}.
    You are given the following extracted parts of a repository which might
    contain several modules and each module will contain a set of files.
    Look at the source code in the repository and you have to generate
    content for the section of a README.md file following the heading given
    below starting with ## based on the context. If you use any hyperlinks,
    they should link back to the github repository shared with you.
    You should only use hyperlinks that are explicitly listed in the context.
    Do NOT make up a hyperlink that is not listed.

    Assume the reader is a {target_audience} but is not deeply familiar with
    {project_name}.
    Assume the reader does not know anything about how the project is
    structured or which folders/files do what and what functions are written
    in which files and what these functions do.
    If you don't know how to fill up the README.md file in one of its
    sections, leave that part blank. Don't try to make up any content.
    Do not include information that is not directly relevant to repository,
    even though the names of the functions might be common or is frequently
    used in several other places.
    Provide the answer in correct markdown format.

    {additional_instructions}

    Context:
    {{context}}

    Question: Provide the README content for the section with
    heading \"{{input}}\" starting with {{input}}.
    Answer in Markdown:
    """

    # Return a template object instead of string
    # if you have a class handling it
    return PromptTemplate(
        template=template, input_variables=["input", "context"]
    )


def make_qa_chain(
    project_name,
    repository_url,
    content_type,
    chat_prompt,
    target_audience,
    vectorstore,
    llms: List[LLMModels],
    device: str = "cpu",
    on_token_stream=None,
):
    """Make QA Chain"""
    llm = llms[1] if len(llms) > 1 else llms[0]
    llm_name = llm.value
    print(f"LLM:  {llm_name.lower()}")
    question_chat_model = None
    doc_chat_model = None
    model_kwargs = {"temperature": 0.1, "device": device}

    if "llama" in llm_name.lower():
        if "gguf" in llm_name.lower():
            model_kwargs["gguf_file"] = models[llm].gguf_file
        question_chat_model = get_llama_chat_model(
            llm_name, model_kwargs=model_kwargs
        )
    elif "gemma" in llm_name.lower():
        if "gguf" in llm_name.lower():
            model_kwargs["gguf_file"] = models[llm].gguf_file
        question_chat_model = get_gemma_chat_model(
            llm_name, model_kwargs=model_kwargs
        )
    else:
        question_chat_model = get_openai_chat_model(
            llm_name,
            temperature=0.1,
            streaming=bool(on_token_stream),
            model_kwargs={
                "frequency_penalty": None,
                "presence_penalty": None,
            },
        )

    question_generator = create_history_aware_retriever(
        question_chat_model, vectorstore.as_retriever(), condense_qa_prompt
    )

    model_kwargs = {"temperature": 0.2, "device": device}
    if "llama" in llm_name.lower():
        if "gguf" in llm_name.lower():
            model_kwargs["gguf_file"] = models[llm].gguf_file
        doc_chat_model = get_llama_chat_model(
            llm_name,
            streaming=bool(on_token_stream),
            model_kwargs=model_kwargs,
        )
    elif "gemma" in llm_name.lower():
        if "gguf" in llm_name.lower():
            model_kwargs["gguf_file"] = models[llm].gguf_file
        question_chat_model = get_gemma_chat_model(
            llm_name,
            streaming=bool(on_token_stream),
            model_kwargs=model_kwargs,
        )
    else:
        doc_chat_model = get_openai_chat_model(
            llm_name,
            temperature=0.2,
            streaming=bool(on_token_stream),
            model_kwargs={
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
        )

    qa_prompt = make_qa_prompt(
        project_name,
        repository_url,
        content_type,
        chat_prompt,
        target_audience,
    )
    doc_chain = create_stuff_documents_chain(
        llm=doc_chat_model, prompt=qa_prompt
    )

    return create_retrieval_chain(
        retriever=question_generator, combine_docs_chain=doc_chain
    )


def make_readme_chain(
    project_name,
    repository_url,
    content_type,
    chat_prompt,
    target_audience,
    vectorstore,
    llms: List[LLMModels],
    peft_model=None,
    device: str = "cpu",
    on_token_stream=None,
):
    """Make Readme Chain"""
    llm = llms[1] if len(llms) > 1 else llms[0]
    llm_name = llm.value
    doc_chat_model = None
    print(f"LLM:  {llm_name.lower()}")
    model_kwargs = {
        "temperature": 0.2,
        "peft_model": peft_model,
        "device": device,
    }
    if "llama" in llm_name.lower():
        if "gguf" in llm_name.lower():
            model_kwargs["gguf_file"] = models[llm].gguf_file
        doc_chat_model = get_llama_chat_model(
            llm_name,
            streaming=bool(on_token_stream),
            model_kwargs=model_kwargs,
        )
    elif "gemma" in llm_name.lower():
        if "gguf" in llm_name.lower():
            model_kwargs["gguf_file"] = models[llm].gguf_file
        doc_chat_model = get_gemma_chat_model(
            llm_name,
            streaming=bool(on_token_stream),
            model_kwargs=model_kwargs,
        )
    else:
        doc_chat_model = get_openai_chat_model(
            llm_name,
            temperature=0.2,
            streaming=bool(on_token_stream),
            model_kwargs={
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
        )

    readme_prompt = make_readme_prompt(
        project_name,
        repository_url,
        content_type,
        chat_prompt,
        target_audience,
    )
    doc_chain = create_stuff_documents_chain(
        llm=doc_chat_model, prompt=readme_prompt
    )

    return create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=doc_chain
    )

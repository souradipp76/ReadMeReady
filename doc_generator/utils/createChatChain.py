from langchain.chains.conversational_retrieval.base import ChatVectorDBChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from doc_generator.utils.LLMUtils import get_chat_model

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
        "You are given a repository which might contain several modules and each module will contain a set of files.\n"
        "Look at the source code in the repository and you have to geneate a readme.md file following some template given below. If you use any hyperlinks, they should link back to the github repository shared with you.\n"
        "You should only use hyperlinks that are explicitly listed in the context. Do NOT make up a hyperlink that is not listed.\n"
        
        "Assume the reader is a {target_audience} but is not deeply familiar with {project_name}.\n"
        "Assume the reader does not know anything about how the project is structured or which folders/files do what and what functions are written in which files and what these functions do.\n"
        "If you don't know how to fill up the readme.md file in one of its sections, leave that part blank. Don't try to make up an answer.\n"
        "Do not include information that is not directly relevant to repository, even though the names of the functions might be common or is frequently used in several other places.\n"
        "Now lets start describing how the readme.md file will be structured.\n"
        "The first section will be Installation. Here provide a list of packages from the requirements.txt folder in the repository that needs to be installed. Mention what versions of those packages need to be installed. Also add the commands that need to be put in the terminal to install those packages. For instance, for installing a py-package, provide a command pip install py-package. If there is no requirements.txt or similar folder in the repository, then find out what frameworks and packages have been imported in all the files after going through the code provide their names and the required versions that need to be installed. Remind the user that it is usually best-practice in Python projects to install into a sandboxed virtual environment, This will be locked to a specific Python version and contain only the Python libraries that you install into it, so that your Python projects do not get affected.\n"
        "The second section will be Usage. Here provide a list of commands that need to be run in the terminal to use various features of the project. For instance, if the project has a command called run, then provide a command to run that command. Go through various files and various modules and after reading each function, provide an example usage of that function. Write two lines about each function, what the functionality of that function is, what paramters it take as input, what is outputs, if it is dependent on the output of any other function in the whole repository, what other functions call/use this function, and finally provide a toy example of the usage of the function. Do this for every function in all files that exist in the repository. Structure them in the same way the repository has been structured. Finally provide some run commands on how to run the main function in the terminal.\n"
        "The third section will be Development. Here provide a list of commands that need to be run in the terminal to develop. Comment here on how to make format, make lint, check types and generate tests for the code that has been written in the files of the repository. Try to create unit tests and write tests for functions that are called/ used by many other functions, to test that they work correctly. Write some commands on how to run those tests.\n"
        "The fourth section will be Conclusion. Here provide a general idea of what the project does and how it works. Here you can also provide how the user can geenrate a github.io page for the repository documentation using the four sections that you generated above in the readme.md file. Tell how to create a workflow and yml file and github pages can be used to create a documentation for the project. Put the readme.md file in the repository.\n"
        
        f"{additional_instructions}\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Answer in Markdown:"
    )

def make_chain(project_name, repository_url, content_type, chat_prompt, target_audience, vectorstore, llms, on_token_stream=None):
    llm = llms[1] if len(llms) > 1 else llms[0]
    question_generator = LLMChain(
        # llm=ChatOpenAI(temperature=0.1, model_name=llm),
        llm=get_chat_model(llm, {"temperature": 0.1}),
        prompt=condense_prompt
    )

    qa_prompt = make_qa_prompt(project_name, repository_url, content_type, chat_prompt, target_audience)
    doc_chain = load_qa_chain(
        # llm=ChatOpenAI(temperature=0.2,
        #                model_name=llm,
        #                streaming=bool(on_token_stream),
        #                model_kwargs={
        #                 "frequency_penalty": 0.0,
        #                 "presence_penalty": 0.0,
        #                }),
        llm=get_chat_model(llm, {
            "temperature": 0.2
        }),
        prompt=qa_prompt
    )

    return ChatVectorDBChain(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator
    )

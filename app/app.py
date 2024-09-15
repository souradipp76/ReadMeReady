import streamlit as st
import os
import shutil
import git
from streamlit_monaco import st_monaco
from pymarkdown.api import PyMarkdownApi

from doc_generator.query import query
from doc_generator.index import index
from doc_generator.types import AutodocRepoConfig, AutodocUserConfig, LLMModels

# App title
st.set_page_config(page_title="Readme Generator")

# Replicate Credentials
with st.sidebar:
    st.title('Readme Generator')
    st.write('This document generator is created using the open-source LLM models.')
    # if 'REPLICATE_API_TOKEN' in st.secrets:
    #     st.success('API key already provided!', icon='✅')
    #     replicate_api = st.secrets['REPLICATE_API_TOKEN']
    # else:
    #     replicate_api = st.text_input('Enter Replicate API token:', type='password')
    #     if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
    #         st.warning('Please enter your credentials!', icon='⚠️')
    #     else:
    #         st.success('Proceed to entering your prompt message!', icon='👉')
    # os.environ['REPLICATE_API_TOKEN'] = replicate_api

    openai_api_key = st.text_input('Enter OpenAI API Key:', type='password')
    os.environ['OPENAI_API_KEY'] = openai_api_key

    with st.form("my_form"):
        st.subheader('Models and parameters')
        options = [
            LLMModels.LLAMA2_7B_CHAT_GPTQ.value,
            LLMModels.LLAMA2_13B_CHAT_GPTQ.value,
            LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ.value,
            LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ.value,
            LLMModels.LLAMA2_7B_CHAT_HF.value,
            LLMModels.LLAMA2_13B_CHAT_HF.value,
            LLMModels.CODELLAMA_7B_INSTRUCT_HF.value,
            LLMModels.CODELLAMA_13B_INSTRUCT_HF.value,
            LLMModels.GOOGLE_GEMMA_2B_INSTRUCT.value,
            LLMModels.GOOGLE_GEMMA_7B_INSTRUCT.value
        ]
        llm = st.selectbox('Choose a model', options, key='llm')
        temperature = st.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.slider('max_length', min_value=32, max_value=128, value=120, step=8)

        st.subheader('Repo Config')
        name = st.text_input(label='Project Name', placeholder="repo")
        project_url = st.text_input(label='Project URL', placeholder = "https://github.com/username/repo")
        project_root = "../data/"
        output_dir = os.path.join("./output", name)
        # is_peft = st.checkbox(label="Is finetuned?")
        # peft_model_path = st.text_input(label='Finetuned Model Path', placeholder="./output/model/")
        submitted = st.form_submit_button("Submit")
        if submitted:
            shutil.rmtree(project_root)
            repo = git.Repo.clone_from(project_url, os.path.join(project_root, name))
            repo_config = {
                "name": name,
                "root": project_root,
                "repository_url": project_url,
                "output": output_dir,
                "llms": [llm],
                "peft_model_path": None,
                "ignore": [
                    ".*",
                    "*package-lock.json",
                    "*package.json",
                    "node_modules",
                    "*dist*",
                    "*build*",
                    "*test*",
                    "*.svg",
                    "*.md",
                    "*.mdx",
                    "*.toml"
                ],
                "file_prompt": "Write a detailed technical explanation of \
                    what this code does. \n      Focus on the high-level \
                    purpose of the code and how it may be used in the \
                    larger project.\n      Include code examples where \
                    appropriate. Keep you response between 100 and 300 \
                    words. \n      DO NOT RETURN MORE THAN 300 WORDS.\n \
                    Output should be in markdown format.\n \
                    Do not just list the methods and classes in this file.",
                "folder_prompt": "Write a technical explanation of what the \
                    code in this file does\n      and how it might fit into the \
                    larger project or work with other parts of the project.\n      \
                    Give examples of how this code might be used. Include code \
                    examples where appropriate.\n      Be concise. Include any \
                    information that may be relevant to a developer who is \
                    curious about this code.\n      Keep you response under \
                    400 words. Output should be in markdown format.\n      \
                    Do not just list the files and folders in this folder.",
                "chat_prompt": "",
                "content_type": "docs",
                "target_audience": "smart developer",
                "link_hosted": True,
                "priority": None,
                "max_concurrent_calls": 50,
                "add_questions": False
            }
            user_config = {
                "llms": [llm]
            }

            repo_conf = AutodocRepoConfig(
                name=repo_config["name"],
                repository_url=repo_config["repository_url"],
                root=repo_config["root"],
                output=repo_config["output"],
                llms=repo_config["llms"],
                peft_model_path=repo_config["peft_model_path"],
                priority=repo_config["priority"],
                max_concurrent_calls=repo_config["max_concurrent_calls"],
                add_questions=repo_config["add_questions"],
                ignore=repo_config["ignore"],
                file_prompt=repo_config["file_prompt"],
                folder_prompt=repo_config["folder_prompt"],
                chat_prompt=repo_config["chat_prompt"],
                content_type=repo_config["content_type"],
                target_audience=repo_config["target_audience"],
                link_hosted=repo_config["link_hosted"],
            )
            usr_conf = AutodocUserConfig(llms=user_config['llms'])
            index.index(**repo_config)
            
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')


# Markdown editor
st.title("Markdown Editor")
default_readme_content = "# Hello world"
if "readme_content" not in st.session_state.keys():
    st.session_state.readme_content = default_readme_content

content = st_monaco(
    value=st.session_state.readme_content, 
    height="600px", 
    language="markdown",
    lineNumbers=True,
    minimap=False,
    theme="vs-dark",
)

if st.button(label="Save"):
    st.session_state.readme_content = content
    st.success("Saved")

def validate_markdown():
    error_str = ""
    errors = PyMarkdownApi().scan_string(st.session_state.readme_content)
    if len(errors.scan_failures) > 0:
        print(errors.scan_failures)
        error_str = "\n".join([f'Line {failure.line_number}: Col {failure.column_number}: {failure.rule_id}: {failure.rule_description} {failure.extra_error_information} ({failure.rule_name})' for failure in errors.scan_failures])
    return error_str  

if st.button("Validate"):
    error_str = validate_markdown()
    if not error_str:
        error_str = "No error"
    validate_container = st.empty()
    validate_container.text_area(
        "Validation Results",
        value=error_str,
        height=150,
    )

if st.download_button(
    label="Download",
    data=st.session_state.readme_content,
    file_name="README.md",
    mime="text/markdown",
):
    st.success("Downloaded")


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Provide a heading to generate README section?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Provide a heading to generate README section?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input(disabled=not openai_api_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query.generate_readme_section(prompt, **repo_config, **user_config)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
    st.session_state.readme_content += full_response
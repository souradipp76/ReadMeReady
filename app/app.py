import streamlit as st
import os
import shutil
import git
from streamlit_ace import st_ace
from pymarkdown.api import PyMarkdownApi

# App title
st.set_page_config(page_title="Readme Generator", layout="wide",page_icon=":material/graphic_eq:")

# Credentials and COnfiguration
with st.sidebar:
    st.title(':rainbow[Readme Generator]')
    st.write('This is a Readme generator app which uses open-source large language models.')

    openai_api_key = "dummy"
    # openai_api_key = st.text_input('Enter OpenAI API Key:', type='password')
    os.environ['OPENAI_API_KEY'] = openai_api_key

    hf_token = st.text_input('Enter HuggingFace token:', type='password')
    os.environ['HF_TOKEN'] = hf_token

    from doc_generator.query import query
    from doc_generator.index import index
    from doc_generator.types import AutodocRepoConfig, AutodocUserConfig, LLMModels

    with st.form("my_form"):
        st.subheader('Model')
        options = [
            LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,
            LLMModels.LLAMA2_7B_CHAT_GPTQ.value,
            LLMModels.LLAMA2_13B_CHAT_GPTQ.value,
            LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ.value,
            LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ.value,
            LLMModels.LLAMA2_7B_CHAT_HF.value,
            LLMModels.LLAMA2_13B_CHAT_HF.value,
            LLMModels.CODELLAMA_7B_INSTRUCT_HF.value,
            LLMModels.CODELLAMA_13B_INSTRUCT_HF.value,
            LLMModels.GOOGLE_GEMMA_2B_INSTRUCT.value,
            LLMModels.GOOGLE_GEMMA_7B_INSTRUCT.value,
            LLMModels.GOOGLE_GEMMA_2B_INSTRUCT_GGUF.value
        ]
        llm = st.selectbox('Choose a model', options, key='llm')
        device = st.selectbox('Choose a device', ["cpu", "gpu"], key='device')
        st.subheader('Parameters')
        temperature = st.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.slider('max_length', min_value=512, max_value=4096, value=2048, step=512)

        st.subheader('Repository Configuration')
        name = st.text_input(label='Name', placeholder="example_repo")
        project_url = st.text_input(label='GitHub Link', placeholder = "https://github.com/username/example_repo")
        project_root = os.path.join(".", name)
        output_dir = os.path.join("output", name)
        # is_peft = st.checkbox(label="Is finetuned?")
        # peft_model_path = st.text_input(label='Finetuned Model Path', placeholder="./output/model/")
        submitted = st.form_submit_button("Setup")
        if submitted:
            st.toast('Indexing repository...')
            try:
                repo = git.Repo.clone_from(project_url, project_root)
            except:
                print('Project already exists.')

            match llm:
                case LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value:
                    model = LLMModels.TINYLLAMA_1p1B_CHAT_GGUF
                case LLMModels.LLAMA2_7B_CHAT_GPTQ.value:
                    model = LLMModels.LLAMA2_7B_CHAT_GPTQ
                case LLMModels.LLAMA2_13B_CHAT_GPTQ.value:
                    model = LLMModels.LLAMA2_13B_CHAT_GPTQ
                case LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ.value:
                    model = LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ
                case LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ.value:
                    model = LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ
                case LLMModels.LLAMA2_13B_CHAT_HF.value:
                    model = LLMModels.LLAMA2_13B_CHAT_HF
                case LLMModels.CODELLAMA_7B_INSTRUCT_HF.value:
                    model = LLMModels.CODELLAMA_7B_INSTRUCT_HF
                case LLMModels.CODELLAMA_13B_INSTRUCT_HF.value:
                    model = LLMModels.CODELLAMA_13B_INSTRUCT_HF
                case LLMModels.GOOGLE_GEMMA_2B_INSTRUCT.value:
                    model = LLMModels.GOOGLE_GEMMA_2B_INSTRUCT
                case LLMModels.GOOGLE_GEMMA_7B_INSTRUCT.value:
                    model = LLMModels.GOOGLE_GEMMA_7B_INSTRUCT
                case LLMModels.GOOGLE_GEMMA_2B_INSTRUCT_GGUF.value:
                    model = LLMModels.GOOGLE_GEMMA_2B_INSTRUCT_GGUF
                case _:
                    model = LLMModels.LLAMA2_7B_CHAT_HF
        
            repo_config = {
                "name": name,
                "root": project_root,
                "repository_url": project_url,
                "output": output_dir,
                "llms": [model],
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
                "add_questions": False,
                "device": device,
            }
            user_config = {
                "llms": [model]
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
                device=repo_config["device"]
            )
            usr_conf = AutodocUserConfig(llms=user_config['llms'], streaming=True)
            index.index(repo_conf)
            st.session_state.repo_conf = repo_conf
            st.session_state.usr_conf = usr_conf
            st.session_state.chain = query.init_readme_chain(st.session_state.repo_conf, st.session_state.usr_conf)
            st.toast('Repository indexing done.')
            

    st.markdown('📖 Learn more about this app [here](https://github.com/souradipp76/ReadMeReady)!!')

left, right = st.columns(2, vertical_alignment="top")

with left:
    st.title("Chat")
    history = st.container(height=1000)
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Provide a heading to generate README section starting with ##?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        history.chat_message(message["role"]).write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Provide a heading to generate README section starting with ##?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


    # User-provided prompt
    if prompt := st.chat_input(disabled=not openai_api_key):
        st.session_state.messages.append({"role": "user", "content": prompt})
        history.chat_message("user").write(prompt)

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.spinner("Thinking..."):
                if "chain" not in st.session_state.keys():
                    full_response = 'Please setup model and repository!!'
                else:
                    chain = st.session_state.chain
                    full_response = ''
                    for chunk in chain.stream({'input': prompt}):
                        print(chunk)
                        if answer_chunk := chunk.get("answer"):
                            full_response += answer_chunk
                history.chat_message("assistant").markdown(full_response)
                    
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

with right:
    # Markdown editor
    st.title("Readme Editor")
    default_readme_content = "# "+ name
    if "readme_content" not in st.session_state.keys():
        st.session_state.readme_content = default_readme_content

    st.session_state.readme_content = st_ace(
        placeholder=st.session_state.readme_content,
        height = 850,
        language="markdown",
        theme="solarized_dark",
        keybinding="vscode",
        font_size=14,
        tab_size=4,
        show_gutter=True,
        show_print_margin=False,
        wrap=True,
        auto_update=True,
        readonly=False,
        min_lines=45,
        key="ace",
    )

    def validate_markdown():
        error_str = ""
        errors = PyMarkdownApi().scan_string(st.session_state.readme_content)
        if len(errors.scan_failures) > 0:
            print(errors.scan_failures)
            error_str = "\n".join([f'Line {failure.line_number}: Col {failure.column_number}: {failure.rule_id}: {failure.rule_description} {failure.extra_error_information} ({failure.rule_name})' for failure in errors.scan_failures])
        return error_str

    col1, col2, col3, col4 = st.columns(4, vertical_alignment="center")

    with col1:
        if st.button("Validate", use_container_width=True):
            st.session_state.error_str = validate_markdown()

    with col2:
        if st.download_button(
            label="Download",
            data=st.session_state.readme_content,
            file_name="README.md",
            mime="text/markdown",
            use_container_width=True
        ):
            col3.success("Downloaded")

    
    with st.expander("Validation", expanded=False):
        validate_container = st.empty()
        if "error_str" in st.session_state.keys():
            validate_container.text_area(
                "Results",
                value=st.session_state.error_str,
                height=150,
            )
    
    with st.expander("Preview", expanded=False):
        st.markdown(st.session_state.readme_content, unsafe_allow_html=True)

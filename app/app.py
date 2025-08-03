import streamlit as st
import os
import shutil
from streamlit_ace import st_ace
import requests

from streamlit.runtime.scriptrunner import get_script_run_ctx

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

    ctx = get_script_run_ctx()
    session_id = None
    if ctx:
        session_id = ctx.session_id
        # You can now use session_id for various purposes, e.g., logging or file naming
        print(f"Current Streamlit session ID: {session_id}")
    else:
        print("Not running within a Streamlit context.")

    with st.form("my_form"):
        st.subheader('Model')
        options = requests.get(url="http://127.0.0.1:8000/models").json()
        llm = st.selectbox('Choose a model', options, key='llm')
        device = st.selectbox('Choose a device', ["cpu", "auto"], key='device')
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
            request_body = {
                "name": name,
                "project_url": project_url,
                "model": llm,
                "device": device,
                "temperature": temperature,
                "top_p": top_p,
                "max_length": max_length,
                # "peft_model_path": None,
            }

            response = requests.post(
                url="http://localhost:8000/setup",
                json=request_body,
                headers={"x-session-id": session_id}
            ).json()

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
                with history.chat_message("assistant"):
                    response = requests.post(
                        url="http://localhost:8000/query",
                        json={"query": prompt},
                        headers={"x-session-id": session_id},
                        stream=True
                    )
                    
                    if response.status_code == 200:
                        placeholder = st.empty()
                        full_response = ''
                        for chunk in response.iter_content(chunk_size=10, decode_unicode=True):
                            full_response += chunk
                            placeholder.markdown(full_response)
                    else:
                        full_response = "Please setup model and repository!!"
                    
                    # if "chain" not in st.session_state.keys():
                    #     full_response = 'Please setup model and repository!!'
                    # else:
                    #     chain = st.session_state.chain
                    #     placeholder = st.empty()
                    #     full_response = ''
                    #     for chunk in chain.stream({'input': prompt}):
                    #         if answer_chunk := chunk.get("answer"):
                    #             full_response += answer_chunk
                    #             placeholder.markdown(full_response)
                    
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
        # errors = PyMarkdownApi().scan_string(st.session_state.readme_content)
        errors = requests.get(
            url="http://localhost:8000/validate_markdown",
            json={"content": st.session_state.readme_content}
        ).json()
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

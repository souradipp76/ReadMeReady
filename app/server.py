from pathlib import Path
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Annotated

from pymarkdown.api import PyMarkdownApi

import git
import os

openai_api_key = "dummy"
os.environ['OPENAI_API_KEY'] = openai_api_key

from readme_ready.query import query
from readme_ready.index import index
from readme_ready.types import AutodocRepoConfig, AutodocUserConfig, LLMModels

app = FastAPI()

session_details = {}

class QueryRequest(BaseModel):
    query: str

class IndexRequest(BaseModel):
    name: str
    project_url: str
    model: LLMModels
    device: str = "cpu"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1000
    peft_model_path: str = None

class ValidateMarkdownRequest(BaseModel):
    content: str

@app.get("/models")
async def get_models() -> list[str]:
    """
    Get the list of available LLM models.

    Returns:
        A list of available LLM models.
    """
    return [model.value for model in LLMModels]

async def generate_streamed_response(prompt: str, session_id: str):
    chain = session_details[session_id]["chain"]
    history = session_details[session_id]["messages"]
    async for chunk in chain.astream({'input': prompt, 'chat_history': history}):
        if answer_chunk := chunk.get('answer'):
            import sys
            sys.stdout.write(answer_chunk)
            yield answer_chunk

@app.post("/query")
async def handle_query(request: QueryRequest, x_session_id: Annotated[str, Header(alias="x-session-id")]):
    """
    Handle query requests to the ReadMeReady service.

    Args:
        request: A QueryRequest containing the query and user configuration.

    Returns:
        The response from the query function.
    """
    if x_session_id not in session_details:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session not found. Please set up the repository first"
        )
    return StreamingResponse(generate_streamed_response(request.query, x_session_id), media_type="text/event-stream")

@app.post("/setup")
async def handle_setup(request: IndexRequest, x_session_id: Annotated[str, Header(alias="x-session-id")]):
    """
    Handle setup requests to the ReadMeReady service.

    Args:
        request: An IndexRequest containing the repository configuration and dry run flag.

    Returns:
        The response from the setup function.
    """
    project_root = Path(__file__).parent.parent
    repo_dir = os.path.join(project_root, "tmp", request.name)
    output_dir = os.path.join(project_root, "output", request.name)
    session_id = x_session_id

    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    try:
        repo = git.Repo.clone_from(request.project_url, repo_dir)
    except:
        print('Project already exists.')

    model = LLMModels(request.model)

    repo_config = {
        "name": request.name,
        "root": repo_dir,
        "repository_url": request.project_url,
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
        "device": request.device,
    }

    user_config = {
        "llms": [model],
        "streaming": True,
    }

    repo_conf = AutodocRepoConfig(**repo_config)
    usr_conf = AutodocUserConfig(**user_config)

    index.index(repo_conf)
    chain = query.init_readme_chain(repo_conf, usr_conf)

    session_details[session_id] = {
        "chain": chain,
        "repo_config": repo_conf,
        "user_config": usr_conf,
        "messages": []
    }
    return {"message": "Setup completed successfully."}


@app.get("/validate_markdown")
async def validate_markdown(request: ValidateMarkdownRequest) -> str:
    """
    Validate the markdown files in the repository.

    Returns:
        A message indicating the validation status.
    """
    return PyMarkdownApi().scan_string(request.content)
    
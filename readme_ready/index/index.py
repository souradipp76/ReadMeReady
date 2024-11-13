"""
Index
"""

from pathlib import Path

from readme_ready.types import AutodocRepoConfig

from .convert_json_to_markdown import convert_json_to_markdown
from .create_vector_store import create_vector_store
from .process_repository import process_repository


def index(config: AutodocRepoConfig):
    """Index"""
    json_path = Path(config.output) / "docs" / "json"
    markdown_path = Path(config.output) / "docs" / "markdown"
    data_path = Path(config.output) / "docs" / "data"

    # Ensure directories exist
    json_path.mkdir(parents=True, exist_ok=True)
    markdown_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    # Process the repository to create JSON files
    print("Processing repository...")
    process_repository(
        AutodocRepoConfig(
            name=config.name,
            repository_url=config.repository_url,
            root=config.root,
            output=str(json_path),
            llms=config.llms,
            priority=config.priority,
            max_concurrent_calls=config.max_concurrent_calls,
            add_questions=config.add_questions,
            ignore=config.ignore,
            file_prompt=config.file_prompt,
            folder_prompt=config.folder_prompt,
            chat_prompt=config.chat_prompt,
            content_type=config.content_type,
            target_audience=config.target_audience,
            link_hosted=config.link_hosted,
            peft_model_path=config.peft_model_path,
            device=config.device,
        )
    )

    # Convert the JSON files to Markdown
    print("Creating markdown files...")
    convert_json_to_markdown(
        AutodocRepoConfig(
            name=config.name,
            repository_url=config.repository_url,
            root=str(json_path),
            output=str(markdown_path),
            llms=config.llms,
            priority=config.priority,
            max_concurrent_calls=config.max_concurrent_calls,
            add_questions=config.add_questions,
            ignore=config.ignore,
            file_prompt=config.file_prompt,
            folder_prompt=config.folder_prompt,
            chat_prompt=config.chat_prompt,
            content_type=config.content_type,
            target_audience=config.target_audience,
            link_hosted=config.link_hosted,
            peft_model_path=config.peft_model_path,
            device=config.device,
        )
    )

    # Create a vector store from the Markdown documents
    print("Creating vector files...")
    create_vector_store(
        str(config.root),
        str(data_path),
        config.ignore,
        config.llms,
        config.device,
    )

from pathlib import Path

from doc_generator.types import AutodocRepoConfig

from .convertJsonToMarkdown import convertJsonToMarkdown
from .processRepository import processRepository
from .createVectorStore import createVectorStore


def index(config: AutodocRepoConfig):
    json_path = Path(config.output) / 'docs' / 'json'
    markdown_path = Path(config.output) / 'docs' / 'markdown'
    data_path = Path(config.output) / 'docs' / 'data'

    # Ensure directories exist
    json_path.mkdir(parents=True, exist_ok=True)
    markdown_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    # Process the repository to create JSON files
    print('Processing repository...')
    processRepository(AutodocRepoConfig(
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
    ))

    # Convert the JSON files to Markdown
    print('Creating markdown files...')
    convertJsonToMarkdown(AutodocRepoConfig(
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
    ))

    # Create a vector store from the Markdown documents
    print('Creating vector files...')
    createVectorStore(config.root, data_path, config.llms)

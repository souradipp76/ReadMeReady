"""
Convert Json to Markdown
"""

import json
from pathlib import Path

from readme_ready.types import (
    AutodocRepoConfig,
    FileSummary,
    FolderSummary,
    ProcessFileParams,
    TraverseFileSystemParams,
)
from readme_ready.utils.file_utils import get_file_name
from readme_ready.utils.traverse_file_system import traverse_file_system


def convert_json_to_markdown(config: AutodocRepoConfig):
    """Convert Json to Markdown"""
    project_name = config.name
    input_root = Path(config.root)
    output_root = Path(config.output)
    file_prompt = config.file_prompt
    folder_prompt = config.folder_prompt
    content_type = config.content_type
    target_audience = config.target_audience
    link_hosted = config.link_hosted

    # Count the number of files in the project
    files = 0

    def count_files(process_file_params: ProcessFileParams):
        nonlocal files
        files += 1
        return

    traverse_file_system(
        TraverseFileSystemParams(
            str(input_root),
            project_name,
            count_files,
            None,
            [],
            file_prompt,
            folder_prompt,
            content_type,
            target_audience,
            link_hosted,
        )
    )

    # Process and create markdown files for each code file in the project
    def process_file(process_file_params: ProcessFileParams) -> None:
        file_path = Path(process_file_params.file_path)
        file_name = process_file_params.file_name
        content = file_path.read_text(encoding="utf-8")

        if not content or len(content) == 0:
            return

        markdown_file_path = output_root.joinpath(
            file_path.relative_to(input_root)
        )

        # Create the output directory if it doesn't exist
        markdown_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse JSON content based on the file name
        data = json.loads(content)
        if file_name == "summary.json":
            data = FolderSummary(**data)
        else:
            data = FileSummary(**data)

        # Only include the file if it has a summary
        markdown = ""
        if data.summary:
            markdown = f"[View code on GitHub]({data.url})\n\n{data.summary}\n"
            if data.questions:
                markdown += f"## Questions: \n{data.questions}"

        output_path = get_file_name(markdown_file_path, ".", ".md")
        output_path.write_text(markdown, encoding="utf-8")

    traverse_file_system(
        TraverseFileSystemParams(
            str(input_root),
            project_name,
            process_file,
            None,
            [],
            file_prompt,
            folder_prompt,
            content_type,
            target_audience,
            link_hosted,
        )
    )

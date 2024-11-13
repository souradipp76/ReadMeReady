"""
Prompts
"""

from typing import List

from readme_ready.types import FileSummary, FolderSummary


def create_code_file_summary(
    file_path: str,
    project_name: str,
    file_contents: str,
    content_type: str,
    file_prompt: str,
) -> str:
    """Create Code File Summary"""
    return f"""
    You are acting as a {content_type} documentation expert
    for a project called {project_name}.
    Below is the {content_type} from a file located at `{file_path}`.
    {file_prompt}
    Do not say "this file is a part of the {project_name} project".

    {content_type}:
    {file_contents}

    Response:

    """


def create_code_questions(
    file_path: str,
    project_name: str,
    file_contents: str,
    content_type: str,
    target_audience: str,
) -> str:
    """Create Code Questions"""
    return f"""
    You are acting as a {content_type} documentation expert
    for a project called {project_name}.
    Below is the {content_type} from a file located at `{file_path}`.
    What are 3 questions that a {target_audience} might have about
    this {content_type}?
    Answer each question in 1-2 sentences. Output should be in markdown format.

    {content_type}:
    {file_contents}

    Questions and Answers:

    """


def folder_summary_prompt(
    folder_path: str,
    project_name: str,
    files: List[FileSummary],
    folders: List[FolderSummary],
    content_type: str,
    folder_prompt: str,
) -> str:
    """Folder Summary Prompt"""
    files_summary = "\n".join(
        [
            f"""
        Name: {file.file_name}
        Summary: {file.summary}

      """
            for file in files
        ]
    )

    folders_summary = "\n".join(
        [
            f"""
        Name: {folder.folder_name}
        Summary: {folder.summary}

      """
            for folder in folders
        ]
    )

    return f"""
    You are acting as a {content_type} documentation expert
    for a project called {project_name}.
    You are currently documenting the folder located at `{folder_path}`.

    Below is a list of the files in this folder and a summary
    of the contents of each file:
    {files_summary}

    And here is a list of the subfolders in this folder and a
    summary of the contents of each subfolder:
    {folders_summary}

    {folder_prompt}
    Do not say "this file is a part of the {project_name} project".
    Do not just list the files and folders.

    Response:
    """

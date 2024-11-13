import textwrap
from readme_ready.index.prompts import (
    create_code_file_summary,
    create_code_questions,
    folder_summary_prompt,
)
from readme_ready.types import FolderSummary, FileSummary


def test_create_code_file_summary():
    file_path = "src/main.py"
    project_name = "TestProject"
    file_contents = "def main():\n    print('Hello, World!')"
    content_type = "Python code"
    file_prompt = "Please summarize the functionality of this code."

    result = create_code_file_summary(
        file_path, project_name, file_contents, content_type, file_prompt
    )

    expected_output = f"""
    You are acting as a {content_type} documentation expert
    for a project called {project_name}.
    Below is the {content_type} from a file located at `{file_path}`.
    {file_prompt}
    Do not say "this file is a part of the {project_name} project".

    {content_type}:
    {file_contents}

    Response:

    """
    # Dedent and strip to normalize whitespace
    result = textwrap.dedent(result).strip()
    expected_output = textwrap.dedent(expected_output).strip()
    assert result == expected_output


def test_create_code_questions():
    file_path = "src/utils.py"
    project_name = "TestProject"
    file_contents = "def add(a, b):\n    return a + b"
    content_type = "Python code"
    target_audience = "new developers"

    result = create_code_questions(
        file_path, project_name, file_contents, content_type, target_audience
    )

    expected_output = f"""
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
    # Dedent and strip to normalize whitespace
    result = textwrap.dedent(result).strip()
    expected_output = textwrap.dedent(expected_output).strip()
    assert result == expected_output


def test_folder_summary_prompt():
    folder_path = "src"
    project_name = "TestProject"
    content_type = "Python code"
    folder_prompt = "Please provide an overview of this folder."

    files = [
        FileSummary(
            file_name="main.py",
            file_path="src/main.py",
            url="http://example.com/main.py",
            summary="This file contains the main execution logic.",
            questions=[],
            checksum="abc123",
        ),
        FileSummary(
            file_name="utils.py",
            file_path="src/utils.py",
            url="http://example.com/utils.py",
            summary="Utility functions used across the project.",
            questions=[],
            checksum="def456",
        ),
    ]

    folders = [
        FolderSummary(
            folder_name="helpers",
            folder_path="src/helpers",
            url="http://example.com/helpers",
            summary="Helper modules for specific tasks.",
            files=[],
            folders=[],
            questions=[],
            checksum="abcd1234",
        ),
        FolderSummary(
            folder_name="tests",
            folder_path="src/tests",
            url="http://example.com/tests",
            summary="Unit tests for the project.",
            files=[],
            folders=[],
            questions=[],
            checksum="defg5678",
        ),
    ]

    result = folder_summary_prompt(
        folder_path, project_name, files, folders, content_type, folder_prompt
    )

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

    expected_output = f"""
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
    # Dedent and strip to normalize whitespace
    result = textwrap.dedent(result).strip()
    expected_output = textwrap.dedent(expected_output).strip()
    assert result == expected_output


def test_folder_summary_prompt_no_files_no_folders():
    folder_path = "src/empty_folder"
    project_name = "TestProject"
    content_type = "Python code"
    folder_prompt = "Please provide an overview of this folder."

    files = []
    folders = []

    result = folder_summary_prompt(
        folder_path, project_name, files, folders, content_type, folder_prompt
    )

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

    expected_output = f"""
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
    # Dedent and strip to normalize whitespace
    result = textwrap.dedent(result).strip()
    expected_output = textwrap.dedent(expected_output).strip()
    assert result == expected_output

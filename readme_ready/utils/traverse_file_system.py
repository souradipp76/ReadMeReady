"""
Traverse File System
"""

import fnmatch
from pathlib import Path

import magic

from readme_ready.types import (
    ProcessFileParams,
    ProcessFolderParams,
    TraverseFileSystemParams,
)


def traverse_file_system(params: TraverseFileSystemParams):
    """Traverse File System"""
    try:
        input_path = Path(params.input_path)
        if not input_path.exists():
            print("The provided folder path does not exist.")
            return

        def should_ignore(file_name: str):
            return any(
                fnmatch.fnmatch(file_name, pattern)
                for pattern in params.ignore
            )

        def dfs(current_path: Path):
            contents = [
                entry
                for entry in current_path.iterdir()
                if not should_ignore(entry.name)
            ]

            # Process directories first
            for entry in contents:
                if entry.is_dir():
                    dfs(entry)
                    if params.process_folder:
                        params.process_folder(
                            ProcessFolderParams(
                                str(params.input_path),
                                entry.name,
                                str(entry),
                                params.project_name,
                                params.content_type,
                                params.folder_prompt,
                                params.target_audience,
                                params.link_hosted,
                                should_ignore,
                            )
                        )

            # Process files
            for entry in contents:
                if entry.is_file():
                    file_path = str(entry)
                    with open(file_path, "rb") as file:
                        content = file.read()
                        if magic.from_buffer(content, mime=True).startswith(
                            "text/"
                        ):
                            if params.process_file:
                                params.process_file(
                                    ProcessFileParams(
                                        entry.name,
                                        file_path,
                                        params.project_name,
                                        params.content_type,
                                        params.file_prompt,
                                        params.target_audience,
                                        params.link_hosted,
                                    )
                                )

        dfs(input_path)

    except RuntimeError as e:
        print(f"Error during traversal: {e}")

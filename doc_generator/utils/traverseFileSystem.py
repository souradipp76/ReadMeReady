from pathlib import Path
from doc_generator.types import TraverseFileSystemParams, ProcessFolderParams, ProcessFileParams
import fnmatch
import magic


def traverseFileSystem(params: TraverseFileSystemParams):
    try:
        inputPath = Path(params.input_path)
        if not inputPath.exists():
            print('The provided folder path does not exist.')
            return

        def shouldIgnore(fileName):
            return any(fnmatch.fnmatch(fileName, pattern) for pattern in params.ignore)

        def dfs(currentPath: Path):
            contents = [entry for entry in currentPath.iterdir() if not shouldIgnore(entry.name)]

            # Process directories first
            for entry in contents:
                if entry.is_dir():
                    dfs(entry)
                    if params.process_folder:
                        params.process_folder(ProcessFolderParams(
                            str(params.input_path),
                            entry.name,
                            str(entry),
                            params.project_name,
                            shouldIgnore,
                            params.folder_prompt,
                            params.content_type,
                            params.target_audience,
                            params.link_hosted,
                        ))

            # Process files
            for entry in contents:
                if entry.is_file():
                    filePath = str(entry)
                    with open(filePath, 'rb') as file:
                        content = file.read()
                        if magic.from_buffer(content, mime=True).startswith('text/'):
                            if params.process_file:
                                params.process_file(ProcessFileParams(
                                    entry.name,
                                    filePath,
                                    params.process_file,
                                    params.file_prompt,
                                    params.content_type,
                                    params.target_audience,
                                    params.link_hosted
                                ))

        dfs(inputPath)

    except Exception as e:
        print(f"Error during traversal: {e}")


import asyncio
from pathlib import Path
import fnmatch
import magic

class TraverseFileSystemParams:
    def __init__(self, inputPath, projectName, processFile=None, processFolder=None,
                 ignore=None, filePrompt=None, folderPrompt=None, contentType=None,
                 targetAudience=None, linkHosted=None):
        self.inputPath = inputPath
        self.projectName = projectName
        self.processFile = processFile
        self.processFolder = processFolder
        self.ignore = ignore if ignore else []
        self.filePrompt = filePrompt
        self.folderPrompt = folderPrompt
        self.contentType = contentType
        self.targetAudience = targetAudience
        self.linkHosted = linkHosted

def traverseFileSystem(params: TraverseFileSystemParams):
    try:
        inputPath = Path(params.inputPath)
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
                    if params.processFolder:
                        params.processFolder({
                            'inputPath': str(params.inputPath),
                            'folderName': entry.name,
                            'folderPath': str(entry),
                            'projectName': params.projectName,
                            'shouldIgnore': shouldIgnore,
                            'folderPrompt': params.folderPrompt,
                            'contentType': params.contentType,
                            'targetAudience': params.targetAudience,
                            'linkHosted': params.linkHosted,
                        })

            # Process files
            for entry in contents:
                if entry.is_file():
                    filePath = str(entry)
                    with open(filePath, 'rb') as file:
                        content = file.read()
                        if magic.from_buffer(content, mime=True).startswith('text/'):
                            if params.processFile:
                                params.processFile({
                                    'fileName': entry.name,
                                    'filePath': filePath,
                                    'projectName': params.projectName,
                                    'filePrompt': params.filePrompt,
                                    'contentType': params.contentType,
                                    'targetAudience': params.targetAudience,
                                    'linkHosted': params.linkHosted,
                                })

        dfs(inputPath)

    except Exception as e:
        print(f"Error during traversal: {e}")


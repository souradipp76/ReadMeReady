import asyncio
from pathlib import Path
import json
from typing import Any, Dict

# Assume necessary imports for custom types and utility functions
from types import AutodocRepoConfig, FileSummary, FolderSummary
from utils import traverseFileSystem
from spinner import spinnerSuccess, updateSpinnerText
from file_util import getFileName

async def convertJsonToMarkdown(config: AutodocRepoConfig):
    projectName = config.name
    inputRoot = Path(config.root)
    outputRoot = Path(config.output)
    filePrompt = config.filePrompt
    folderPrompt = config.folderPrompt
    contentType = config.contentType
    targetAudience = config.targetAudience
    linkHosted = config.linkHosted

    # Count the number of files in the project
    files = 0

    async def count_files(file_info: Dict[str, Any]):
        nonlocal files
        files += 1
        return

    await traverseFileSystem({
        'inputPath': str(inputRoot),
        'projectName': projectName,
        'processFile': count_files,
        'ignore': [],
        'filePrompt': filePrompt,
        'folderPrompt': folderPrompt,
        'contentType': contentType,
        'targetAudience': targetAudience,
        'linkHosted': linkHosted,
    })

    # Process and create markdown files for each code file in the project
    async def process_file(file_info: Dict[str, Any]):
        filePath = Path(file_info['filePath'])
        fileName = file_info['fileName']
        content = filePath.read_text(encoding='utf-8')

        # Handle error if content is empty
        if not content:
            return

        markdownFilePath = outputRoot.joinpath(filePath.relative_to(inputRoot))

        # Create the output directory if it doesn't exist
        markdownFilePath.parent.mkdir(parents=True, exist_ok=True)

        # Parse JSON content based on the file name
        data = json.loads(content)
        if fileName == 'summary.json':
            data = FolderSummary(**data)
        else:
            data = FileSummary(**data)

        # Only include the file if it has a summary
        markdown = ''
        if data.summary:
            markdown = f"[View code on GitHub]({data.url})\n\n{data.summary}\n"
            if data.questions:
                markdown += f"## Questions: \n{data.questions}"

        outputPath = getFileName(markdownFilePath, '.', '.md')
        outputPath.write_text(markdown, encoding='utf-8')

    updateSpinnerText(f"Creating {files} markdown files...")
    await traverseFileSystem({
        'inputPath': str(inputRoot),
        'projectName': projectName,
        'processFile': process_file,
        'ignore': [],
        'filePrompt': filePrompt,
        'folderPrompt': folderPrompt,
        'contentType': contentType,
        'targetAudience': targetAudience,
        'linkHosted': linkHosted,
    })
    spinnerSuccess(f"Created {files} markdown files...")

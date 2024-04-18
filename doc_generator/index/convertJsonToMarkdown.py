from pathlib import Path
import json
from pathlib import Path
from typing import Any, Dict

from doc_generator.types import AutodocRepoConfig, FileSummary, FolderSummary, ProcessFileParams, TraverseFileSystemParams
from doc_generator.utils.traverseFileSystem import traverseFileSystem
from doc_generator.utils.FileUtils import get_file_name

def convertJsonToMarkdown(config: AutodocRepoConfig):
    projectName = config.name
    inputRoot = Path(config.root)
    outputRoot = Path(config.output)
    filePrompt = config.file_prompt
    folderPrompt = config.folder_prompt
    contentType = config.content_type
    targetAudience = config.target_audience
    linkHosted = config.link_hosted

    # Count the number of files in the project
    files = 0

    def count_files(x):
        nonlocal files
        files += 1
        return

    traverseFileSystem(TraverseFileSystemParams(
        str(inputRoot),
        projectName,
        count_files,
        None,
        [],
        filePrompt,
        folderPrompt,
        contentType,
        targetAudience,
        linkHosted
    ))

    # Process and create markdown files for each code file in the project
    def process_file(processFileParams: ProcessFileParams):
        filePath = Path(processFileParams.file_path)
        fileName = processFileParams.file_name
        content = filePath.read_text(encoding='utf-8')


        if not content or len(content) == 0:
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

        outputPath = get_file_name(markdownFilePath, '.', '.md')
        outputPath.write_text(markdown, encoding='utf-8')

    traverseFileSystem(TraverseFileSystemParams(
        str(inputRoot),
        projectName,
        process_file,
        None,
        [],
        filePrompt,
        folderPrompt,
        contentType,
        targetAudience,
        linkHosted
    ))

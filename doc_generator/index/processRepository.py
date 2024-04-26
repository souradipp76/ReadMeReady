import hashlib
import json
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_experimental.chat_models import Llama2Chat
import tiktoken

from doc_generator.types import (
    AutodocRepoConfig,
    FileSummary,
    FolderSummary,
    ProcessFileParams,
    ProcessFolderParams,
    TraverseFileSystemParams,
)
from doc_generator.utils.FileUtils import (
    get_file_name,
    github_file_url,
    github_folder_url,
)
from doc_generator.utils.LLMUtils import models
from doc_generator.utils.traverseFileSystem import traverseFileSystem

from .prompts import (
    create_code_file_summary,
    create_code_questions,
    folder_summary_prompt,
)
from .selectModel import select_model


def processRepository(config: AutodocRepoConfig, dryRun=False):

    def read_file(path):
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def write_file(path, content):
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)

    def callLLM(prompt: str, model: Llama2Chat):
        return model.invoke(prompt)

    def isModel(model):
        return model is not None

    def filesAndFolders(config):
        files, folders = 0, 0

        def count_files(x):
            nonlocal files
            files += 1
            return

        def count_folder(x):
            nonlocal folders
            folders += 1
            return

        params = TraverseFileSystemParams(
            config.root,
            config.name,
            count_files,
            count_folder,
            config.ignore,
            config.file_prompt,
            config.folder_prompt,
            config.content_type,
            config.target_audience,
            config.link_hosted,
        )
        traverseFileSystem(params)
        return {"files": files, "folders": folders}

    def processFile(processFileParams: ProcessFileParams):
        fileName = processFileParams.file_name
        filePath = processFileParams.file_path
        content = read_file(filePath)

        newChecksum = calculateChecksum([content])

        reindex = shouldReindex(
            Path(config.root) / Path(filePath).parent,
            f"{Path(fileName).stem}.json",
            newChecksum,
        )
        if not reindex:
            return

        markdownFilePath = Path(config.output) / filePath
        url = github_file_url(
            config.repository_url, config.root, filePath, config.link_hosted
        )

        summaryPrompt = create_code_file_summary(
            processFileParams.project_name,
            processFileParams.project_name,
            content,
            processFileParams.content_type,
            processFileParams.file_prompt,
        )
        questionsPrompt = create_code_questions(
            processFileParams.project_name,
            processFileParams.project_name,
            content,
            processFileParams.content_type,
            processFileParams.target_audience,
        )
        prompts = (
            [summaryPrompt, questionsPrompt]
            if config.add_questions
            else [summaryPrompt]
        )

        model = select_model(prompts, config.llms, models, config.priority)
        if not isModel(model):
            return

        encoding = tiktoken.encoding_for_model(model.name)
        summaryLength = len(encoding.encode(summaryPrompt))
        questionLength = len(encoding.encode(questionsPrompt))

        if not dryRun:
            responses = [callLLM(prompt, model.llm) for prompt in prompts]

            fileSummary = FileSummary(
                file_name=fileName,
                file_path=str(filePath),
                url=url,
                summary=responses[0],
                questions=responses[1] if config.add_questions else "",
                checksum=newChecksum,
            )

            outputPath = get_file_name(markdownFilePath, ".", ".json")
            output_content = (
                json.dumps(fileSummary, indent=2) if fileSummary.summary else ""
            )

            write_file(outputPath, output_content)

            model.inputTokens += summaryLength + (
                questionLength if config.add_questions else 0
            )
            model.outputTokens += 1000  # Example token adjustment
            model.total += 1
            model.succeeded += 1

    def processFolder(processFolderParams: ProcessFolderParams):
        if dryRun:
            return

        folderName = processFolderParams.folder_name
        folderPath = processFolderParams.folder_path
        contents_list = list(Path(folderPath).iterdir())
        contents = []
        for x in contents_list:
            if not processFolderParams.should_ignore(x.as_posix()):
                contents.append(x.as_posix())

        newChecksum = calculateChecksum(contents)

        reindex = shouldReindex(folderPath, "summary.json", newChecksum)
        if not reindex:
            return

        url = github_folder_url(
            config.repository_url,
            config.root,
            folderPath,
            processFolderParams.link_hosted,
        )
        fileSummaries = []
        for f in contents:
            entryPath = Path(folderPath) / f
            if entryPath.is_file() and f.name != "summary.json":
                file = read_file(entryPath)
                if len(file) > 0:
                    fileSummaries.append(json.dumps(file))
        folderSummaries = []
        for f in contents:
            entryPath = Path(folderPath) / f
            if entryPath.is_dir():
                summaryFilePath = Path(entryPath) / "summary.json"
                file = read_file(summaryFilePath)
                if len(file) > 0:
                    folderSummaries.append(json.dumps(file))
        summaryPrompt = folder_summary_prompt(
            folderPath,
            processFolderParams.project_name,
            fileSummaries,
            folderSummaries,
            processFolderParams.content_type,
            processFolderParams.folder_prompt,
        )
        model = select_model([summaryPrompt], config.llms, models, config.priority)
        if not isModel(model):
            return

        summary = callLLM(summaryPrompt, model.llm)

        folderSummary = FolderSummary(
            folder_name=folderName,
            folder_path=str(folderPath),
            url=url,
            files=fileSummaries,
            folders=folderSummaries,
            summary=summary,
            questions="",
            checksum=newChecksum,
        )

        outputPath = Path(folderPath) / "summary.json"
        write_file(str(outputPath), json.dumps(folderSummary, indent=2))

    files_folders_count = filesAndFolders(config)
    print(
        f"Processing {files_folders_count['files']} files and {files_folders_count['folders']} folders..."
    )
    params = TraverseFileSystemParams(
        config.root,
        config.name,
        processFile,
        processFolder,
        config.ignore,
        config.file_prompt,
        config.folder_prompt,
        config.content_type,
        config.target_audience,
        config.link_hosted,
    )
    traverseFileSystem(params)
    print("Processing complete.")


def calculateChecksum(contents):
    m = hashlib.md5()
    [m.update(str(content).encode("utf-8")) for content in contents]
    return m.hexdigest()


def shouldReindex(contentPath, name, newChecksum):
    jsonPath = Path(contentPath) / name
    try:
        with open(jsonPath, "r", encoding="utf-8") as file:
            oldChecksum = json.load(file)["checksum"]
        return oldChecksum != newChecksum
    except FileNotFoundError:
        return True

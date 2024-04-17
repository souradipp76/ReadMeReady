import hashlib
import json
from pathlib import Path

import tiktoken

from doc_generator.types import (AutodocRepoConfig, FileSummary, FolderSummary,
                                 TraverseFileSystemParams)
from doc_generator.utils.FileUtils import (getFileName, githubFileUrl,
                                           githubFolderUrl)
from doc_generator.utils.LLMUtils import models, selectModel
from doc_generator.utils.traverseFileSystem import traverseFileSystem
from doc_generator.utils.APIRateLimit import APIRateLimit

from .prompts import (create_code_file_summary, create_code_questions,
                      folder_summary_prompt)


def processRepository(config: AutodocRepoConfig, dryRun=False):
    rateLimit = APIRateLimit(config.max_concurrent_calls)

    def callLLM(prompt, model):
        def model_call():
            return model.call(prompt)
        return rateLimit.call_api(model_call)

    def isModel(model):
        return model is not None

    def processFile(fileInfo):
        fileName = fileInfo['fileName']
        filePath = fileInfo['filePath']
        content = read_file(filePath)

        newChecksum = calculateChecksum(content)

        reindex = shouldReindex(
            Path(config.outputRoot) / Path(filePath).parent,
            f"{Path(fileName).stem}.json",
            newChecksum
        )
        if not reindex:
            return

        markdownFilePath = Path(config.outputRoot) / filePath
        url = githubFileUrl(config.repositoryUrl, config.inputRoot, filePath, config.link_hosted)

        summaryPrompt = create_code_file_summary(
            config.name, config.name, content, config.contentType, config.file_prompt
        )
        questionsPrompt = create_code_questions(
            config.name, config.name, content, config.contentType, config.target_audience
        )
        prompts = [summaryPrompt, questionsPrompt] if config.addQuestions else [summaryPrompt]

        model = selectModel(prompts, config.llms, models, config.priority)
        if not isModel(model):
            return

        encoding = tiktoken.get_encoding(model.name)
        summaryLength = len(encoding.encode(summaryPrompt))
        questionLength = len(encoding.encode(questionsPrompt))

        if not dryRun:
            responses = [callLLM(prompt, model.llm) for prompt in prompts]

            fileSummary = FileSummary(
                file_name=fileName,
                file_path=str(filePath),
                url=url,
                summary=responses[0],
                questions=responses[1] if config.add_questions else '',
                checksum=newChecksum
            )

            outputPath = getFileName(markdownFilePath, '.', '.json')
            content = json.dumps(fileSummary, indent=2) if fileSummary.summary else ''

            write_file(outputPath, content)

            model.inputTokens += summaryLength + (questionLength if config.add_questions else 0)
            model.outputTokens += 1000  # Example token adjustment
            model.total += 1
            model.succeeded += 1

    def processFolder(folderInfo):
        if dryRun:
            return

        folderName = folderInfo['folderName']
        folderPath = folderInfo['folderPath']
        contents = list(Path(folderPath).iterdir())

        newChecksum = calculateChecksum(contents)

        reindex = shouldReindex(folderPath, 'summary.json', newChecksum)
        if not reindex:
            return

        url = githubFolderUrl(config.repository_url, config.root, folderPath, config.link_hosted)
        fileSummaries = [processFile({'fileName': f.name, 'filePath': str(f)}) for f in contents if f.is_file() and f.name != 'summary.json']
        folderSummaries = [processFolder({'folderName': f.name, 'folderPath': str(f)}) for f in contents if f.is_dir()]

        summaryPrompt = folder_summary_prompt(folderPath, config.name, fileSummaries, folderSummaries, config.content_type, config.folder_prompt)
        model = selectModel([summaryPrompt], config.llms, models, config.priority)
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
            questions='',
            checksum=newChecksum
        )

        outputPath = Path(folderPath) / 'summary.json'
        write_file(str(outputPath), json.dumps(folderSummary, indent=2))

    files_folders_count = filesAndFolders(config)
    print(f"Processing {files_folders_count['files']} files and {files_folders_count['folders']} folders...")
    params = TraverseFileSystemParams(config.root,
                                      config.name,
                                      processFile,
                                      processFolder,
                                      config.ignore,
                                      config.file_prompt,
                                      config.folder_prompt,
                                      config.content_type,
                                      config.target_audience,
                                      config.link_hosted
                                      )
    traverseFileSystem(params)
    print("Processing complete.")

def read_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)

def calculateChecksum(contents):
    m = hashlib.md5()
    [m.update(content.encode('utf-8')) for content in contents]
    return m.hexdigest()

def shouldReindex(contentPath, name, newChecksum):
    jsonPath = Path(contentPath) / name
    try:
        with open(jsonPath, 'r', encoding='utf-8') as file:
            oldChecksum = json.load(file)['checksum']
        return oldChecksum != newChecksum
    except FileNotFoundError:
        return True


def filesAndFolders(config):
    files, folders = 0, 0
    params = TraverseFileSystemParams(config.root,
                                      config.name,
                                      lambda: setattr(files, files + 1),
                                      lambda: setattr(folders, folders + 1),
                                      config.ignore,
                                      config.file_prompt,
                                      config.folder_prompt,
                                      config.content_type,
                                      config.target_audience,
                                      config.link_hosted
                                      )
    traverseFileSystem(params)
    return {'files': files, 'folders': folders}


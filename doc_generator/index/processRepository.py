import asyncio
import hashlib
from pathlib import Path
import json
from langchain.llms import OpenAIChat
from your_module.tiktoken import encoding_for_model
from your_module.api_rate_limit import APIRateLimit
from your_module.prompts import createCodeFileSummary, createCodeQuestions, folderSummaryPrompt
from your_module.types import AutodocRepoConfig, FileSummary, FolderSummary, ProcessFile, ProcessFolder
from your_module.traverse_file_system import traverseFileSystem
from your_module.spinner import spinnerSuccess, stopSpinner, updateSpinnerText
from your_module.file_util import getFileName, githubFileUrl, githubFolderUrl
from your_module.llm_util import models, selectModel

async def processRepository(config: AutodocRepoConfig, dryRun=False):
    rateLimit = APIRateLimit(config.maxConcurrentCalls)

    async def callLLM(prompt, model):
        async def model_call():
            return await model.call(prompt)
        return await rateLimit.callApi(model_call)

    def isModel(model):
        return model is not None

    async def processFile(fileInfo):
        fileName = fileInfo['fileName']
        filePath = fileInfo['filePath']
        content = await asyncio.to_thread(read_file, filePath)

        newChecksum = calculateChecksum(content)

        reindex = await shouldReindex(
            Path(config.outputRoot) / Path(filePath).parent,
            f"{Path(fileName).stem}.json",
            newChecksum
        )
        if not reindex:
            return

        markdownFilePath = Path(config.outputRoot) / filePath
        url = githubFileUrl(config.repositoryUrl, config.inputRoot, filePath, config.linkHosted)

        summaryPrompt = createCodeFileSummary(
            config.name, config.name, content, config.contentType, config.filePrompt
        )
        questionsPrompt = createCodeQuestions(
            config.name, config.name, content, config.contentType, config.targetAudience
        )
        prompts = [summaryPrompt, questionsPrompt] if config.addQuestions else [summaryPrompt]

        model = selectModel(prompts, config.llms, models, config.priority)
        if not isModel(model):
            return

        encoding = encoding_for_model(model.name)
        summaryLength = len(encoding.encode(summaryPrompt))
        questionLength = len(encoding.encode(questionsPrompt))

        if not dryRun:
            responses = await asyncio.gather(*(callLLM(prompt, model.llm) for prompt in prompts))

            fileSummary = FileSummary(
                fileName=fileName,
                filePath=str(filePath),
                url=url,
                summary=responses[0],
                questions=responses[1] if config.addQuestions else '',
                checksum=newChecksum
            )

            outputPath = getFileName(markdownFilePath, '.', '.json')
            content = json.dumps(fileSummary, indent=2) if fileSummary.summary else ''

            await asyncio.to_thread(write_file, outputPath, content)

            model.inputTokens += summaryLength + (questionLength if config.addQuestions else 0)
            model.outputTokens += 1000  # Example token adjustment
            model.total += 1
            model.succeeded += 1

    async def processFolder(folderInfo):
        if dryRun:
            return

        folderName = folderInfo['folderName']
        folderPath = folderInfo['folderPath']
        contents = list(Path(folderPath).iterdir())

        newChecksum = calculateChecksum(contents)

        reindex = await shouldReindex(folderPath, 'summary.json', newChecksum)
        if not reindex:
            return

        url = githubFolderUrl(config.repositoryUrl, config.inputRoot, folderPath, config.linkHosted)
        fileSummaries = await asyncio.gather(*(processFile({'fileName': f.name, 'filePath': str(f)}) for f in contents if f.is_file() and f.name != 'summary.json'))
        folderSummaries = [await processFolder({'folderName': f.name, 'folderPath': str(f)}) for f in contents if f.is_dir()]

        summaryPrompt = folderSummaryPrompt(folderPath, config.name, fileSummaries, folderSummaries, config.contentType, config.folderPrompt)
        model = selectModel([summaryPrompt], config.llms, models, config.priority)
        if not isModel(model):
            return

        summary = await callLLM(summaryPrompt, model.llm)

        folderSummary = FolderSummary(
            folderName=folderName,
            folderPath=str(folderPath),
            url=url,
            files=fileSummaries,
            folders=folderSummaries,
            summary=summary,
            questions='',
            checksum=newChecksum
        )

        outputPath = Path(folderPath) / 'summary.json'
        await asyncio.to_thread(write_file, str(outputPath), json.dumps(folderSummary, indent=2))

    files_folders_count = await filesAndFolders(config.inputRoot, config)
    updateSpinnerText(f"Processing {files_folders_count['files']} files and {files_folders_count['folders']} folders...")
    await traverseFileSystem(config.inputRoot, config, processFile, processFolder)
    spinnerSuccess("Processing complete.")
    stopSpinner()

async def read_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

async def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)

def calculateChecksum(contents):
    m = hashlib.md5()
    [m.update(content.encode('utf-8')) for content in contents]
    return m.hexdigest()

async def shouldReindex(contentPath, name, newChecksum):
    jsonPath = Path(contentPath) / name
    try:
        with open(jsonPath, 'r', encoding='utf-8') as file:
            oldChecksum = json.load(file)['checksum']
        return oldChecksum != newChecksum
    except FileNotFoundError:
        return True

async def filesAndFolders(rootPath, config):
    files, folders = 0, 0
    await traverseFileSystem(rootPath, config,
                             lambda: setattr(files, files + 1),
                             lambda: setattr(folders, folders + 1))
    return {'files': files, 'folders': folders}

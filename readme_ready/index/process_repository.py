"""
Process Repository
"""

import hashlib
import json
import os
from pathlib import Path

import tiktoken
from langchain.chat_models.base import BaseChatModel

from readme_ready.types import (
    AutodocRepoConfig,
    FileSummary,
    FolderSummary,
    ProcessFileParams,
    ProcessFolderParams,
    TraverseFileSystemParams,
)
from readme_ready.utils.file_utils import (
    get_file_name,
    github_file_url,
    github_folder_url,
)
from readme_ready.utils.llm_utils import get_tokenizer, models
from readme_ready.utils.traverse_file_system import traverse_file_system

from .prompts import (
    create_code_file_summary,
    create_code_questions,
    folder_summary_prompt,
)
from .select_model import select_model


def process_repository(config: AutodocRepoConfig, dry_run=False):
    """
    Process Repository
    """

    def read_file(path):
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def write_file(path, content):
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)

    def call_llm(prompt: str, model: BaseChatModel | None):
        if model is None:
            return None
        return model.invoke(prompt)

    def is_model(model):
        return model is not None

    def files_and_folders(config):
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
        traverse_file_system(params)
        return {"files": files, "folders": folders}

    def process_file(process_file_params: ProcessFileParams):
        file_name = process_file_params.file_name
        file_path = process_file_params.file_path
        content = read_file(file_path)

        new_checksum = calculate_checksum([content])

        reindex = should_reindex(
            Path(config.root) / Path(file_path).parent,
            f"{Path(file_name).stem}.json",
            new_checksum,
        )
        if not reindex:
            return

        markdown_file_path = Path(config.output) / file_path
        url = github_file_url(
            config.repository_url, config.root, file_path, config.link_hosted
        )

        summary_prompt = create_code_file_summary(
            process_file_params.project_name,
            process_file_params.project_name,
            content,
            process_file_params.content_type,
            process_file_params.file_prompt,
        )
        questions_prompt = create_code_questions(
            process_file_params.project_name,
            process_file_params.project_name,
            content,
            process_file_params.content_type,
            process_file_params.target_audience,
        )
        prompts = (
            [summary_prompt, questions_prompt]
            if config.add_questions
            else [summary_prompt]
        )

        model = select_model(prompts, config.llms, models, config.priority)
        if not is_model(model):
            return
        assert model is not None

        if "llama" in model.name.lower() or "gemma" in model.name.lower():
            encoding = get_tokenizer(model.name.value)
            summary_length = len(encoding.tokenize(summary_prompt))
            question_length = len(encoding.tokenize(questions_prompt))
        else:
            encoding = tiktoken.encoding_for_model(model.name)
            summary_length = len(encoding.encode(summary_prompt))
            question_length = len(encoding.encode(questions_prompt))

        if not dry_run:
            responses = [call_llm(prompt, model.llm) for prompt in prompts]

            file_summary = FileSummary(
                file_name=file_name,
                file_path=str(file_path),
                url=url,
                summary=responses[0],
                questions=responses[1] if config.add_questions else "",
                checksum=new_checksum,
            )

            os.makedirs(
                markdown_file_path.as_posix().replace(file_name, ""),
                exist_ok=True,
            )
            output_path = get_file_name(
                markdown_file_path.as_posix(), ".", ".json"
            )
            output_content = (
                json.dumps(
                    file_summary, indent=2, default=lambda o: o.__dict__
                )
                if file_summary.summary
                else ""
            )

            write_file(output_path, output_content)

            model.input_tokens += summary_length + (
                question_length if config.add_questions else 0
            )
            model.output_tokens += 1000  # Example token adjustment
            model.total += 1
            model.succeeded += 1

    def process_folder(process_folder_params: ProcessFolderParams):
        if dry_run:
            return

        folder_name = process_folder_params.folder_name
        folder_path = process_folder_params.folder_path
        contents_list = list(Path(folder_path).iterdir())
        contents = []
        for x in contents_list:
            if not process_folder_params.should_ignore(x.as_posix()):
                contents.append(x.as_posix())

        new_checksum = calculate_checksum(contents)

        reindex = should_reindex(folder_path, "summary.json", new_checksum)
        if not reindex:
            return

        url = github_folder_url(
            config.repository_url,
            config.root,
            folder_path,
            process_folder_params.link_hosted,
        )
        file_summaries = []
        for f in contents:
            entry_path = Path(folder_path) / f
            if entry_path.is_file() and f != "summary.json":
                file = read_file(entry_path)
                if len(file) > 0:
                    file_summaries.append(json.loads(file))
        folder_summaries = []
        for f in contents:
            entry_path = Path(folder_path) / f
            if entry_path.is_dir():
                summary_file_path = Path(entry_path) / "summary.json"
                file = read_file(summary_file_path)
                if len(file) > 0:
                    folder_summaries.append(json.loads(file))
        summary_prompt = folder_summary_prompt(
            folder_path,
            process_folder_params.project_name,
            file_summaries,
            folder_summaries,
            process_folder_params.content_type,
            process_folder_params.folder_prompt,
        )
        model = select_model(
            [summary_prompt], config.llms, models, config.priority
        )
        if not is_model(model):
            return
        assert model is not None
        summary = call_llm(summary_prompt, model.llm)

        folder_summary = FolderSummary(
            folder_name=folder_name,
            folder_path=str(folder_path),
            url=url,
            files=file_summaries,
            folders=folder_summaries,
            summary=summary,
            questions="",
            checksum=new_checksum,
        )

        output_path = Path(folder_path) / "summary.json"
        write_file(
            str(output_path),
            json.dumps(folder_summary, indent=2, default=lambda o: o.__dict__),
        )

    files_folders_count = files_and_folders(config)
    print(
        f"Processing {files_folders_count['files']} files "
        + f"and {files_folders_count['folders']} folders..."
    )
    params = TraverseFileSystemParams(
        config.root,
        config.name,
        process_file,
        process_folder,
        config.ignore,
        config.file_prompt,
        config.folder_prompt,
        config.content_type,
        config.target_audience,
        config.link_hosted,
    )
    traverse_file_system(params)
    print("Processing complete.")


def calculate_checksum(contents: list[str]):
    """Calculate Checksum"""
    checksums = [
        hashlib.md5(content.encode()).hexdigest() for content in contents
    ]
    concatenated_checksum = "".join(checksums)
    final_checksum = hashlib.md5(concatenated_checksum.encode())
    return final_checksum.hexdigest()


def should_reindex(content_path, name, new_checksum):
    """Should Reindex"""
    json_path = Path(content_path) / name
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            data = json.loads(data)
            old_checksum = data["checksum"]
        return old_checksum != new_checksum
    except FileNotFoundError:
        return True

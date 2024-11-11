import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json
import os

from doc_generator.index.process_repository import (
    process_repository,
    calculate_checksum,
    should_reindex,
)
from doc_generator.types import (
    AutodocRepoConfig,
    FileSummary,
    FolderSummary,
    ProcessFileParams,
    ProcessFolderParams,
    TraverseFileSystemParams,
    LLMModels,
    Priority,
)
from doc_generator.utils.traverse_file_system import traverse_file_system


def test_calculate_checksum():
    contents = ["content1", "content2"]
    checksum = calculate_checksum(contents)
    assert isinstance(checksum, str)
    assert len(checksum) == 32  # MD5 checksum length


def test_should_reindex_file_not_found(tmp_path):
    content_path = tmp_path / "non_existent"
    name = "summary.json"
    new_checksum = "newchecksum"
    result = should_reindex(content_path, name, new_checksum)
    assert result is True


def test_should_reindex_same_checksum(tmp_path):
    content_path = tmp_path
    name = "summary.json"
    data = {"checksum": "checksum123"}
    (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    result = should_reindex(content_path, name, "checksum123")
    assert result is False


def test_should_reindex_different_checksum(tmp_path):
    content_path = tmp_path
    name = "summary.json"
    data = {"checksum": "oldchecksum"}
    (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    result = should_reindex(content_path, name, "newchecksum")
    assert result is True


@patch("doc_generator.index.process_repository.traverse_file_system")
def test_process_repository(mock_traverse, tmp_path):
    # Set up configuration
    config = AutodocRepoConfig(
        name="TestRepo",
        repository_url="https://github.com/test/repo",
        root=str(tmp_path / "input"),
        output=str(tmp_path / "output"),
        llms=[LLMModels.GPT3],
        priority=Priority.COST,
        max_concurrent_calls=1,
        add_questions=True,
        ignore=[],
        file_prompt=None,
        folder_prompt=None,
        chat_prompt=None,
        content_type=None,
        target_audience=None,
        link_hosted=None,
        peft_model_path=None,
        device=None,
    )

    # Mock files_and_folders function
    with patch("doc_generator.index.process_repository.files_and_folders", return_value={"files": 1, "folders": 1}):
        # Mock inner functions
        with patch("doc_generator.index.process_repository.read_file", return_value="file content"):
            with patch("doc_generator.index.process_repository.write_file") as mock_write_file:
                with patch("doc_generator.index.process_repository.call_llm", return_value="LLM response"):
                    with patch("doc_generator.index.process_repository.is_model", return_value=True):
                        with patch("doc_generator.index.process_repository.select_model") as mock_select_model:
                            mock_model = MagicMock()
                            mock_model.name = "gpt-3.5-turbo"
                            mock_model.llm = MagicMock()
                            mock_select_model.return_value = mock_model

                            # Mock calculate_checksum and should_reindex
                            with patch("doc_generator.index.process_repository.calculate_checksum", return_value="checksum123"):
                                with patch("doc_generator.index.process_repository.should_reindex", return_value=True):
                                    # Run the function
                                    process_repository(config)

                                    # Assertions
                                    mock_traverse.assert_called()
                                    mock_write_file.assert_called()
                                    mock_select_model.assert_called()

                                    # Check if call_llm was called with prompts
                                    assert mock_model.llm.invoke.call_count == 2  # Summary and Questions


def test_process_repository_no_model(tmp_path):
    # Set up configuration
    config = AutodocRepoConfig(
        name="TestRepo",
        repository_url="https://github.com/test/repo",
        root=str(tmp_path / "input"),
        output=str(tmp_path / "output"),
        llms=[],
        priority=Priority.COST,
        max_concurrent_calls=1,
        add_questions=False,
        ignore=[],
        file_prompt=None,
        folder_prompt=None,
        chat_prompt=None,
        content_type=None,
        target_audience=None,
        link_hosted=None,
        peft_model_path=None,
        device=None,
    )

    # Mock files_and_folders function
    with patch("doc_generator.index.process_repository.files_and_folders", return_value={"files": 1, "folders": 1}):
        # Mock inner functions
        with patch("doc_generator.index.process_repository.read_file", return_value="file content"):
            with patch("doc_generator.index.process_repository.write_file") as mock_write_file:
                with patch("doc_generator.index.process_repository.call_llm", return_value="LLM response"):
                    with patch("doc_generator.index.process_repository.is_model", return_value=False):
                        with patch("doc_generator.index.process_repository.select_model") as mock_select_model:
                            mock_select_model.return_value = None

                            # Run the function
                            process_repository(config)

                            # Assertions
                            mock_write_file.assert_not_called()
                            mock_select_model.assert_called()


def test_process_repository_dry_run(tmp_path):
    # Set up configuration
    config = AutodocRepoConfig(
        name="TestRepo",
        repository_url="https://github.com/test/repo",
        root=str(tmp_path / "input"),
        output=str(tmp_path / "output"),
        llms=[LLMModels.GPT3],
        priority=Priority.COST,
        max_concurrent_calls=1,
        add_questions=False,
        ignore=[],
        file_prompt=None,
        folder_prompt=None,
        chat_prompt=None,
        content_type=None,
        target_audience=None,
        link_hosted=None,
        peft_model_path=None,
        device=None,
    )

    # Mock files_and_folders function
    with patch("doc_generator.index.process_repository.files_and_folders", return_value={"files": 1, "folders": 1}):
        # Mock inner functions
        with patch("doc_generator.index.process_repository.read_file", return_value="file content"):
            with patch("doc_generator.index.process_repository.write_file") as mock_write_file:
                with patch("doc_generator.index.process_repository.call_llm", return_value="LLM response"):
                    with patch("doc_generator.index.process_repository.is_model", return_value=True):
                        with patch("doc_generator.index.process_repository.select_model") as mock_select_model:
                            mock_model = MagicMock()
                            mock_model.name = "gpt-3.5-turbo"
                            mock_model.llm = MagicMock()
                            mock_select_model.return_value = mock_model

                            # Run the function with dry_run=True
                            process_repository(config, dry_run=True)

                            # Assertions
                            mock_write_file.assert_not_called()
                            mock_select_model.assert_called()


def test_process_repository_no_reindex(tmp_path):
    # Set up configuration
    config = AutodocRepoConfig(
        name="TestRepo",
        repository_url="https://github.com/test/repo",
        root=str(tmp_path / "input"),
        output=str(tmp_path / "output"),
        llms=[LLMModels.GPT3],
        priority=Priority.COST,
        max_concurrent_calls=1,
        add_questions=False,
        ignore=[],
        file_prompt=None,
        folder_prompt=None,
        chat_prompt=None,
        content_type=None,
        target_audience=None,
        link_hosted=None,
        peft_model_path=None,
        device=None,
    )

    # Mock files_and_folders function
    with patch("doc_generator.index.process_repository.files_and_folders", return_value={"files": 1, "folders": 1}):
        # Mock inner functions
        with patch("doc_generator.index.process_repository.read_file", return_value="file content"):
            with patch("doc_generator.index.process_repository.write_file") as mock_write_file:
                with patch("doc_generator.index.process_repository.call_llm", return_value="LLM response"):
                    with patch("doc_generator.index.process_repository.is_model", return_value=True):
                        with patch("doc_generator.index.process_repository.select_model") as mock_select_model:
                            mock_model = MagicMock()
                            mock_model.name = "gpt-3.5-turbo"
                            mock_model.llm = MagicMock()
                            mock_select_model.return_value = mock_model

                            # Mock should_reindex to return False
                            with patch("doc_generator.index.process_repository.should_reindex", return_value=False):
                                # Run the function
                                process_repository(config)

                                # Assertions
                                mock_write_file.assert_not_called()
                                mock_select_model.assert_not_called()


def test_process_file_no_reindex(tmp_path):
    # Prepare test environment
    file_path = tmp_path / "test.py"
    file_path.write_text("print('Hello World')", encoding="utf-8")
    process_file_params = ProcessFileParams(
        file_name="test.py",
        file_path=str(file_path),
        project_name="TestProject",
        content_type=None,
        file_prompt=None,
        target_audience=None,
        link_hosted=None,
    )
    config = AutodocRepoConfig(
        name="TestProject",
        repository_url="https://github.com/test/repo",
        root=str(tmp_path),
        output=str(tmp_path / "output"),
        llms=[LLMModels.GPT3],
        priority=Priority.COST,
        max_concurrent_calls=1,
        add_questions=False,
        ignore=[],
        file_prompt=None,
        folder_prompt=None,
        chat_prompt=None,
        content_type=None,
        target_audience=None,
        link_hosted=None,
        peft_model_path=None,
        device=None,
    )

    # Mock functions
    with patch("doc_generator.index.process_repository.read_file", return_value="print('Hello World')"):
        with patch("doc_generator.index.process_repository.should_reindex", return_value=False):
            with patch("doc_generator.index.process_repository.write_file") as mock_write_file:
                from doc_generator.index.process_repository import process_file

                process_file(process_file_params)
                mock_write_file.assert_not_called()


def test_process_folder_no_reindex(tmp_path):
    # Prepare test environment
    folder_path = tmp_path / "test_folder"
    folder_path.mkdir()
    process_folder_params = ProcessFolderParams(
        folder_name="test_folder",
        folder_path=str(folder_path),
        project_name="TestProject",
        content_type=None,
        folder_prompt=None,
        link_hosted=None,
        should_ignore=lambda x: False,
    )
    config = AutodocRepoConfig(
        name="TestProject",
        repository_url="https://github.com/test/repo",
        root=str(tmp_path),
        output=str(tmp_path / "output"),
        llms=[LLMModels.GPT3],
        priority=Priority.COST,
        max_concurrent_calls=1,
        add_questions=False,
        ignore=[],
        file_prompt=None,
        folder_prompt=None,
        chat_prompt=None,
        content_type=None,
        target_audience=None,
        link_hosted=None,
        peft_model_path=None,
        device=None,
    )

    # Mock functions
    with patch("doc_generator.index.process_repository.should_reindex", return_value=False):
        with patch("doc_generator.index.process_repository.write_file") as mock_write_file:
            from doc_generator.index.process_repository import process_folder

            process_folder(process_folder_params)
            mock_write_file.assert_not_called()


def test_calculate_checksum_empty():
    contents = []
    checksum = calculate_checksum(contents)
    assert isinstance(checksum, str)
    assert len(checksum) == 32  # MD5 checksum length


def test_process_file_no_content(tmp_path):
    # Prepare test environment
    file_path = tmp_path / "empty.py"
    file_path.write_text("", encoding="utf-8")
    process_file_params = ProcessFileParams(
        file_name="empty.py",
        file_path=str(file_path),
        project_name="TestProject",
        content_type=None,
        file_prompt=None,
        target_audience=None,
        link_hosted=None,
    )
    config = AutodocRepoConfig(
        name="TestProject",
        repository_url="https://github.com/test/repo",
        root=str(tmp_path),
        output=str(tmp_path / "output"),
        llms=[LLMModels.GPT3],
        priority=Priority.COST,
        max_concurrent_calls=1,
        add_questions=False,
        ignore=[],
        file_prompt=None,
        folder_prompt=None,
        chat_prompt=None,
        content_type=None,
        target_audience=None,
        link_hosted=None,
        peft_model_path=None,
        device=None,
    )

    # Mock functions
    with patch("doc_generator.index.process_repository.read_file", return_value=""):
        with patch("doc_generator.index.process_repository.write_file") as mock_write_file:
            from doc_generator.index.process_repository import process_file

            process_file(process_file_params)
            mock_write_file.assert_not_called()


def test_process_folder_dry_run(tmp_path):
    # Prepare test environment
    folder_path = tmp_path / "test_folder"
    folder_path.mkdir()
    process_folder_params = ProcessFolderParams(
        folder_name="test_folder",
        folder_path=str(folder_path),
        project_name="TestProject",
        content_type=None,
        folder_prompt=None,
        link_hosted=None,
        should_ignore=lambda x: False,
    )

    # Mock functions
    with patch("doc_generator.index.process_repository.write_file") as mock_write_file:
        from doc_generator.index.process_repository import process_folder

        process_folder(process_folder_params)
        mock_write_file.assert_not_called()

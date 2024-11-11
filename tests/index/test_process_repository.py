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
    data = "{\"checksum\": \"checksum123\"}"
    (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    result = should_reindex(content_path, name, "checksum123")
    assert result is False


def test_should_reindex_different_checksum(tmp_path):
    content_path = tmp_path
    name = "summary.json"
    data = "{\"checksum\": \"oldchecksum\"}"
    (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    result = should_reindex(content_path, name, "newchecksum")
    assert result is True


# @patch("doc_generator.index.process_repository.traverse_file_system")
# @patch("doc_generator.index.process_repository.select_model")
# @patch("doc_generator.index.process_repository.calculate_checksum", return_value="checksum123")
# @patch("doc_generator.index.process_repository.should_reindex", return_value=True)
# @patch("builtins.open", new_callable=mock_open, read_data="file content")
# def test_process_repository(mock_open_file, mock_should_reindex, mock_calculate_checksum, mock_select_model, mock_traverse, tmp_path):
#     # Set up configuration
#     config = AutodocRepoConfig(
#         name="TestRepo",
#         repository_url="https://github.com/test/repo",
#         root=str(tmp_path / "input"),
#         output=str(tmp_path / "output"),
#         llms=[LLMModels.GPT3],
#         priority=Priority.COST,
#         max_concurrent_calls=1,
#         add_questions=True,
#         ignore=[],
#         file_prompt=None,
#         folder_prompt=None,
#         chat_prompt=None,
#         content_type=None,
#         target_audience=None,
#         link_hosted=None,
#         peft_model_path=None,
#         device=None,
#     )

#     # Set up the model mock
#     mock_model = MagicMock()
#     mock_model.name = "gpt-3.5-turbo"
#     mock_model.llm = MagicMock()
#     mock_select_model.return_value = mock_model

#     # Simulate 'traverse_file_system' calling 'count_files' and 'count_folder'
#     def side_effect_traverse_file_system(params):
#         # Simulate processing a file and a folder
#         params.process_file(ProcessFileParams(
#             file_name="test.py",
#             file_path="test.py",
#             project_name="TestRepo",
#             content_type=None,
#             file_prompt=None,
#             target_audience=None,
#             link_hosted=None,
#         ))
#         params.process_folder(ProcessFolderParams(
#             input_path="",
#             folder_name="test_folder",
#             folder_path=str(tmp_path / "test_folder"),
#             project_name="TestRepo",
#             content_type=None,
#             folder_prompt=None,
#             target_audience=None,
#             link_hosted=None,
#             should_ignore=lambda x: False,
#         ))
#     mock_traverse.side_effect = side_effect_traverse_file_system

#     # Run the function
#     process_repository(config)

#     # Assertions
#     mock_traverse.assert_called()
#     mock_open_file.assert_called()
#     mock_select_model.assert_called()

#     # Check if call_llm was called with prompts
#     assert mock_model.llm.invoke.call_count == 2  # Summary and Questions


# @patch("doc_generator.index.process_repository.traverse_file_system")
# @patch("doc_generator.index.process_repository.select_model", return_value=None)
# @patch("builtins.open", new_callable=mock_open, read_data="file content")
# def test_process_repository_no_model(mock_open_file, mock_select_model, mock_traverse, tmp_path):
#     # Set up configuration
#     config = AutodocRepoConfig(
#         name="TestRepo",
#         repository_url="https://github.com/test/repo",
#         root=str(tmp_path / "input"),
#         output=str(tmp_path / "output"),
#         llms=[],
#         priority=Priority.COST,
#         max_concurrent_calls=1,
#         add_questions=False,
#         ignore=[],
#         file_prompt=None,
#         folder_prompt=None,
#         chat_prompt=None,
#         content_type=None,
#         target_audience=None,
#         link_hosted=None,
#         peft_model_path=None,
#         device=None,
#     )

#     # Simulate 'traverse_file_system' calling 'process_file' and 'process_folder'
#     def side_effect_traverse_file_system(params):
#         params.process_file(None)
#         params.process_folder(None)
#     mock_traverse.side_effect = side_effect_traverse_file_system

#     # Run the function
#     process_repository(config)

#     # Assertions
#     mock_open_file.assert_not_called()
#     mock_select_model.assert_called()


# @patch("doc_generator.index.process_repository.traverse_file_system")
# @patch("doc_generator.index.process_repository.select_model")
# @patch("builtins.open", new_callable=mock_open, read_data="file content")
# def test_process_repository_dry_run(mock_open_file, mock_select_model, mock_traverse, tmp_path):
#     # Set up configuration
#     config = AutodocRepoConfig(
#         name="TestRepo",
#         repository_url="https://github.com/test/repo",
#         root=str(tmp_path / "input"),
#         output=str(tmp_path / "output"),
#         llms=[LLMModels.GPT3],
#         priority=Priority.COST,
#         max_concurrent_calls=1,
#         add_questions=False,
#         ignore=[],
#         file_prompt=None,
#         folder_prompt=None,
#         chat_prompt=None,
#         content_type=None,
#         target_audience=None,
#         link_hosted=None,
#         peft_model_path=None,
#         device=None,
#     )

#     # Simulate 'traverse_file_system' calling 'process_file' and 'process_folder'
#     def side_effect_traverse_file_system(params):
#         params.process_file(None)
#         params.process_folder(None)
#     mock_traverse.side_effect = side_effect_traverse_file_system

#     # Run the function with dry_run=True
#     process_repository(config, dry_run=True)

#     # Assertions
#     mock_open_file.assert_not_called()
#     mock_select_model.assert_called()


# @patch("doc_generator.index.process_repository.traverse_file_system")
# @patch("doc_generator.index.process_repository.select_model")
# @patch("doc_generator.index.process_repository.should_reindex", return_value=False)
# @patch("builtins.open", new_callable=mock_open, read_data="file content")
# def test_process_repository_no_reindex(mock_open_file, mock_should_reindex, mock_select_model, mock_traverse, tmp_path):
#     # Set up configuration
#     config = AutodocRepoConfig(
#         name="TestRepo",
#         repository_url="https://github.com/test/repo",
#         root=str(tmp_path / "input"),
#         output=str(tmp_path / "output"),
#         llms=[LLMModels.GPT3],
#         priority=Priority.COST,
#         max_concurrent_calls=1,
#         add_questions=False,
#         ignore=[],
#         file_prompt=None,
#         folder_prompt=None,
#         chat_prompt=None,
#         content_type=None,
#         target_audience=None,
#         link_hosted=None,
#         peft_model_path=None,
#         device=None,
#     )

#     # Simulate 'traverse_file_system' calling 'process_file' and 'process_folder'
#     def side_effect_traverse_file_system(params):
#         params.process_file(None)
#         params.process_folder(None)
#     mock_traverse.side_effect = side_effect_traverse_file_system

#     # Run the function
#     process_repository(config)

#     # Assertions
#     mock_open_file.assert_not_called()
#     mock_select_model.assert_not_called()


def test_calculate_checksum_empty():
    contents = []
    checksum = calculate_checksum(contents)
    assert isinstance(checksum, str)
    assert len(checksum) == 32  # MD5 checksum length


# def test_process_file_no_content(tmp_path):
#     # Prepare test environment
#     file_path = tmp_path / "empty.py"
#     file_path.write_text("", encoding="utf-8")
#     process_file_params = ProcessFileParams(
#         file_name="empty.py",
#         file_path=str(file_path),
#         project_name="TestProject",
#         content_type=None,
#         file_prompt=None,
#         target_audience=None,
#         link_hosted=None,
#     )

#     # Import the function
#     from doc_generator.index.process_repository import process_file

#     # Mock functions
#     with patch("builtins.open", new_callable=mock_open, read_data=""):
#         with patch("doc_generator.index.process_repository.write_file") as mock_write_file:
#             process_file(process_file_params)
#             mock_write_file.assert_not_called()


# def test_process_folder_dry_run(tmp_path):
#     # Prepare test environment
#     folder_path = tmp_path / "test_folder"
#     folder_path.mkdir()
#     process_folder_params = ProcessFolderParams(
#         input_path="",
#         folder_name="test_folder",
#         folder_path=str(folder_path),
#         project_name="TestProject",
#         content_type=None,
#         folder_prompt=None,
#         target_audience="",
#         link_hosted=None,
#         should_ignore=lambda x: False,

#     )

#     # Import the function
#     from doc_generator.index.process_repository import process_folder

#     # Mock functions
#     with patch("builtins.open", new_callable=mock_open) as mock_open_file:
#         process_folder(process_folder_params)
#         mock_open_file.assert_not_called()

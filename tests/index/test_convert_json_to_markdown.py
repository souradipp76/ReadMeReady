from unittest.mock import MagicMock, patch
from pathlib import Path
from readme_ready.index.convert_json_to_markdown import (
    convert_json_to_markdown,
)


@patch("readme_ready.index.convert_json_to_markdown.traverse_file_system")
@patch("readme_ready.index.convert_json_to_markdown.get_file_name")
@patch("readme_ready.index.convert_json_to_markdown.FileSummary")
@patch("readme_ready.index.convert_json_to_markdown.FolderSummary")
@patch("readme_ready.index.convert_json_to_markdown.Path")
def test_convert_json_to_markdown(
    mock_Path,
    mock_FolderSummary,
    mock_FileSummary,
    mock_get_file_name,
    mock_traverse_file_system,
):
    # Set up the config
    config = MagicMock()
    config.name = "test_project"
    config.root = "/input/root"
    config.output = "/output/root"
    config.file_prompt = "file_prompt"
    config.folder_prompt = "folder_prompt"
    config.content_type = "content_type"
    config.target_audience = "target_audience"
    config.link_hosted = "link_hosted"

    # Prepare different files with different contents
    files = [
        {
            "file_path": "/input/root/empty_file.json",
            "file_name": "empty_file.json",
            "content": "",  # Empty content
        },
        {
            "file_path": "/input/root/summary.json",
            "file_name": "summary.json",
            "content": '{"summary": "Folder summary.", "url": "http://example.com/folder"}',
            "is_folder_summary": True,
        },
        {
            "file_path": "/input/root/file_with_summary.json",
            "file_name": "file_with_summary.json",
            "content": '{"summary": "File summary.", "url": "http://example.com/file"}',
            "is_file_summary": True,
        },
        {
            "file_path": "/input/root/file_without_summary.json",
            "file_name": "file_without_summary.json",
            "content": '{"summary": "", "url": "http://example.com/empty"}',
            "is_file_summary": True,
        },
        {
            "file_path": "/input/root/file_with_questions.json",
            "file_name": "file_with_questions.json",
            "content": '{"summary": "File with questions.", "url": "http://example.com/questions", "questions": "Question content."}',
            "is_file_summary": True,
        },
    ]

    # Map file paths to mock Path instances
    file_paths = {}

    for file_info in files:
        path_instance = MagicMock()
        path_instance.read_text.return_value = file_info["content"]
        relative_path = Path(file_info["file_path"]).relative_to("/input/root")
        path_instance.relative_to.return_value = relative_path
        path_instance.parent = MagicMock()
        path_instance.parent.mkdir.return_value = None
        path_instance.joinpath.return_value = path_instance
        file_paths[file_info["file_path"]] = path_instance

    # Mock Path to return the appropriate mock Path instance
    def path_side_effect(path_str, *args, **kwargs):
        return file_paths.get(path_str, MagicMock())

    mock_Path.side_effect = path_side_effect

    # Keep track of the 'files' variable in convert_json_to_markdown
    files_counter = {"count": 0}

    # Define side effect for traverse_file_system
    def traverse_fs_side_effect(*args, **kwargs):
        params = args[0]
        if params.process_file.__name__ == "count_files":
            # First call, simulate calling count_files for each file
            for file_info in files:
                process_file_params = MagicMock()
                process_file_params.file_path = file_info["file_path"]
                process_file_params.file_name = file_info["file_name"]
                params.process_file(process_file_params)
                files_counter["count"] += 1
        elif params.process_file.__name__ == "process_file":
            # Second call, simulate calling process_file for each file
            for file_info in files:
                process_file_params = MagicMock()
                process_file_params.file_path = file_info["file_path"]
                process_file_params.file_name = file_info["file_name"]
                params.process_file(process_file_params)
        else:
            pass

    mock_traverse_file_system.side_effect = traverse_fs_side_effect

    # Set up mock for get_file_name
    mock_output_path = MagicMock()
    mock_get_file_name.return_value = mock_output_path

    # Mock write_text
    mock_output_path.write_text.return_value = None

    # Define mock classes for FileSummary and FolderSummary
    class MockFileSummary:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.questions = ["Question content."]
            self.checksum = ""

    class MockFolderSummary:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.questions = ["Question content."]
            self.checksum = ""

    mock_FileSummary.side_effect = MockFileSummary
    mock_FolderSummary.side_effect = MockFolderSummary

    # Call the function under test
    convert_json_to_markdown(config)

    # Now we can make assertions
    # Check that files were counted correctly
    assert files_counter["count"] == len(files)

    expected_write_calls = 4

    assert mock_output_path.write_text.call_count == expected_write_calls

from unittest.mock import MagicMock, mock_open, patch

from readme_ready.utils.traverse_file_system import traverse_file_system
from readme_ready.types import (
    ProcessFileParams,
    ProcessFolderParams,
    TraverseFileSystemParams,
)


def test_input_path_does_not_exist(capsys):
    with patch("readme_ready.utils.traverse_file_system.Path") as MockPath:
        mock_input_path = MockPath.return_value
        mock_input_path.exists.return_value = False

        params = TraverseFileSystemParams(
            input_path="/non/existent/path",
            ignore=[],
            process_file=None,
            process_folder=None,
            project_name="TestProject",
            content_type="TestContentType",
            file_prompt="TestFilePrompt",
            folder_prompt="TestFolderPrompt",
            target_audience="TestAudience",
            link_hosted="TestLinkHosted",
        )

        traverse_file_system(params)
        captured = capsys.readouterr()
        assert "The provided folder path does not exist." in captured.out


def test_input_path_exists_with_no_contents():
    with patch("readme_ready.utils.traverse_file_system.Path") as MockPath:
        mock_input_path = MockPath.return_value
        mock_input_path.exists.return_value = True
        mock_input_path.iterdir.return_value = []

        params = TraverseFileSystemParams(
            input_path="/empty/path",
            ignore=[],
            process_file=None,
            process_folder=None,
            project_name="TestProject",
            content_type="TestContentType",
            file_prompt="TestFilePrompt",
            folder_prompt="TestFolderPrompt",
            target_audience="TestAudience",
            link_hosted="TestLinkHosted",
        )

        traverse_file_system(params)
        # Should complete without errors


def test_ignore_pattern_matches_file():
    with patch("readme_ready.utils.traverse_file_system.Path") as MockPath:
        mock_input_path = MockPath.return_value
        mock_input_path.exists.return_value = True

        mock_file = MagicMock()
        mock_file.name = "ignore_me.txt"
        mock_file.is_dir.return_value = False
        mock_file.is_file.return_value = True

        mock_input_path.iterdir.return_value = [mock_file]

        params = TraverseFileSystemParams(
            input_path="/path",
            ignore=["ignore_me.txt"],
            process_file=MagicMock(),
            process_folder=MagicMock(),
            project_name="TestProject",
            content_type="TestContentType",
            file_prompt="TestFilePrompt",
            folder_prompt="TestFolderPrompt",
            target_audience="TestAudience",
            link_hosted="TestLinkHosted",
        )

        traverse_file_system(params)
        params.process_file.assert_not_called()


def test_process_directory_called():
    with patch("readme_ready.utils.traverse_file_system.Path") as MockPath:
        mock_input_path = MockPath.return_value
        mock_input_path.exists.return_value = True

        mock_dir = MagicMock()
        mock_dir.name = "test_dir"
        mock_dir.is_dir.return_value = True
        mock_dir.is_file.return_value = False
        mock_dir.iterdir.return_value = []

        mock_input_path.iterdir.return_value = [mock_dir]

        params = TraverseFileSystemParams(
            input_path="/path",
            ignore=[],
            process_file=None,
            process_folder=MagicMock(),
            project_name="TestProject",
            content_type="TestContentType",
            file_prompt="TestFilePrompt",
            folder_prompt="TestFolderPrompt",
            target_audience="TestAudience",
            link_hosted="TestLinkHosted",
        )

        traverse_file_system(params)
        assert params.process_folder.called
        args = params.process_folder.call_args[0][0]
        assert isinstance(args, ProcessFolderParams)
        assert args.folder_name == "test_dir"


def test_process_file_called_for_text_file():
    with patch("readme_ready.utils.traverse_file_system.Path") as MockPath:
        with patch(
            "readme_ready.utils.traverse_file_system.open",
            mock_open(read_data="dummy data"),
        ) as mock_file:
            with patch(
                "readme_ready.utils.traverse_file_system.magic.from_buffer"
            ) as mock_magic:
                mock_magic.return_value = "text/plain"

                mock_input_path = MockPath.return_value
                mock_input_path.exists.return_value = True

                mock_file_path = MagicMock()
                mock_file_path.name = "test_file.txt"
                mock_file_path.is_file.return_value = True
                mock_file_path.is_dir.return_value = False

                mock_input_path.iterdir.return_value = [mock_file_path]

                params = TraverseFileSystemParams(
                    input_path="/path",
                    ignore=[],
                    process_file=MagicMock(),
                    process_folder=None,
                    project_name="TestProject",
                    content_type="TestContentType",
                    file_prompt="TestFilePrompt",
                    folder_prompt="TestFolderPrompt",
                    target_audience="TestAudience",
                    link_hosted="TestLinkHosted",
                )

                traverse_file_system(params)
                assert params.process_file.called
                args = params.process_file.call_args[0][0]
                assert isinstance(args, ProcessFileParams)
                assert args.file_name == "test_file.txt"


def test_process_file_not_called_for_non_text_file():
    with patch("readme_ready.utils.traverse_file_system.Path") as MockPath:
        with patch(
            "readme_ready.utils.traverse_file_system.open",
            mock_open(read_data="dummy data"),
        ) as mock_file:
            with patch(
                "readme_ready.utils.traverse_file_system.magic.from_buffer"
            ) as mock_magic:
                mock_magic.return_value = "application/octet-stream"

                mock_input_path = MockPath.return_value
                mock_input_path.exists.return_value = True

                mock_file_path = MagicMock()
                mock_file_path.name = "binary_file.bin"
                mock_file_path.is_file.return_value = True
                mock_file_path.is_dir.return_value = False

                mock_input_path.iterdir.return_value = [mock_file_path]

                params = TraverseFileSystemParams(
                    input_path="/path",
                    ignore=[],
                    process_file=MagicMock(),
                    process_folder=None,
                    project_name="TestProject",
                    content_type="TestContentType",
                    file_prompt="TestFilePrompt",
                    folder_prompt="TestFolderPrompt",
                    target_audience="TestAudience",
                    link_hosted="TestLinkHosted",
                )

                traverse_file_system(params)
                params.process_file.assert_not_called()


def test_process_folder_is_none():
    with patch("readme_ready.utils.traverse_file_system.Path") as MockPath:
        mock_input_path = MockPath.return_value
        mock_input_path.exists.return_value = True

        mock_dir = MagicMock()
        mock_dir.name = "test_dir"
        mock_dir.is_dir.return_value = True
        mock_dir.is_file.return_value = False
        mock_dir.iterdir.return_value = []

        mock_input_path.iterdir.return_value = [mock_dir]

        params = TraverseFileSystemParams(
            input_path="/path",
            ignore=[],
            process_file=None,
            process_folder=None,
            project_name="TestProject",
            content_type="TestContentType",
            file_prompt="TestFilePrompt",
            folder_prompt="TestFolderPrompt",
            target_audience="TestAudience",
            link_hosted="TestLinkHosted",
        )

        traverse_file_system(params)
        # Should complete without errors


def test_process_file_is_none():
    with patch("readme_ready.utils.traverse_file_system.Path") as MockPath:
        with patch(
            "readme_ready.utils.traverse_file_system.open",
            mock_open(read_data="dummy data"),
        ) as mock_file:
            with patch(
                "readme_ready.utils.traverse_file_system.magic.from_buffer"
            ) as mock_magic:
                mock_magic.return_value = "text/plain"

                mock_input_path = MockPath.return_value
                mock_input_path.exists.return_value = True

                mock_file_path = MagicMock()
                mock_file_path.name = "test_file.txt"
                mock_file_path.is_file.return_value = True
                mock_file_path.is_dir.return_value = False

                mock_input_path.iterdir.return_value = [mock_file_path]

                params = TraverseFileSystemParams(
                    input_path="/path",
                    ignore=[],
                    process_file=None,
                    process_folder=None,
                    project_name="TestProject",
                    content_type="TestContentType",
                    file_prompt="TestFilePrompt",
                    folder_prompt="TestFolderPrompt",
                    target_audience="TestAudience",
                    link_hosted="TestLinkHosted",
                )

                traverse_file_system(params)
                # Should complete without errors


def test_runtime_error_caught(capsys):
    with patch("readme_ready.utils.traverse_file_system.Path") as MockPath:
        mock_input_path = MockPath.return_value
        mock_input_path.exists.return_value = True

        def side_effect(*args, **kwargs):
            raise RuntimeError("Test runtime error")

        mock_input_path.iterdir.side_effect = side_effect

        params = TraverseFileSystemParams(
            input_path="/path",
            ignore=[],
            process_file=None,
            process_folder=None,
            project_name="TestProject",
            content_type="TestContentType",
            file_prompt="TestFilePrompt",
            folder_prompt="TestFolderPrompt",
            target_audience="TestAudience",
            link_hosted="TestLinkHosted",
        )

        traverse_file_system(params)
        captured = capsys.readouterr()
        assert "Error during traversal: Test runtime error" in captured.out

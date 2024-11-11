import pytest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
from doc_generator.index.convert_json_to_markdown import convert_json_to_markdown

from doc_generator.types import (
    AutodocRepoConfig,
    ProcessFileParams,
    FileSummary,
    FolderSummary,
    TraverseFileSystemParams,
)

def test_convert_json_to_markdown(tmp_path):
    # Set up test configuration
    config = AutodocRepoConfig(
        name="TestProject",
        repository_url="",
        root=str(tmp_path / "input"),
        output=str(tmp_path / "output"),
        llms=[],
        priority=None,
        max_concurrent_calls=10,
        add_questions=False,
        ignore=[],
        file_prompt=None,
        folder_prompt=None,
        content_type=None,
        target_audience=None,
        link_hosted=None,
        chat_prompt="",
        peft_model_path=None,
        device="cpu",
    )

    # Create input directory and files
    input_root = tmp_path / "input"
    input_root.mkdir(parents=True)

    # Create JSON files with different scenarios
    file1 = input_root / "file1.json"
    file1.write_text(json.dumps({
        "summary": "Summary of file1",
        "url": "http://example.com/file1",
        "questions": "Question1"
    }), encoding='utf-8')

    file2 = input_root / "file2.json"
    file2.write_text(json.dumps({
        "summary": "Summary of file2",
        "url": "http://example.com/file2"
    }), encoding='utf-8')

    summary_file = input_root / "summary.json"
    summary_file.write_text(json.dumps({
        "summary": "Summary of folder",
        "url": "http://example.com/summary",
        "questions": "FolderQuestion"
    }), encoding='utf-8')

    empty_file = input_root / "empty.json"
    empty_file.write_text('', encoding='utf-8')

    no_summary_file = input_root / "no_summary.json"
    no_summary_file.write_text(json.dumps({
        "url": "http://example.com/no_summary",
        "questions": "No summary here"
    }), encoding='utf-8')

    # Mock traverse_file_system to control the flow
    with patch('doc_generator.utils.traverse_file_system.traverse_file_system') as mock_traverse:
        def traverse_side_effect(params: TraverseFileSystemParams):
            process_file = params.process_file
            # Simulate file processing
            for file_path in [file1, file2, summary_file, empty_file, no_summary_file]:
                process_file_params = ProcessFileParams(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    folder_path=str(input_root),
                    folder_name=input_root.name,
                    depth=0,
                )
                process_file(process_file_params)
        mock_traverse.side_effect = traverse_side_effect

        # Call the function under test
        convert_json_to_markdown(config)

        # Assert that output files are created
        output_root = Path(config.output)
        assert (output_root / "file1.md").exists()
        assert (output_root / "file2.md").exists()
        assert (output_root / "summary.md").exists()
        assert (output_root / "empty.md").exists()
        assert (output_root / "no_summary.md").exists()

        # Read and verify the contents of the markdown files
        file1_md = (output_root / "file1.md").read_text(encoding='utf-8')
        assert "[View code on GitHub](http://example.com/file1)" in file1_md
        assert "Summary of file1" in file1_md
        assert "## Questions: \nQuestion1" in file1_md

        file2_md = (output_root / "file2.md").read_text(encoding='utf-8')
        assert "[View code on GitHub](http://example.com/file2)" in file2_md
        assert "Summary of file2" in file2_md
        assert "## Questions" not in file2_md

        summary_md = (output_root / "summary.md").read_text(encoding='utf-8')
        assert "[View code on GitHub](http://example.com/summary)" in summary_md
        assert "Summary of folder" in summary_md
        assert "## Questions: \nFolderQuestion" in summary_md

        empty_md = (output_root / "empty.md").read_text(encoding='utf-8')
        assert empty_md == ''

        no_summary_md = (output_root / "no_summary.md").read_text(encoding='utf-8')
        assert no_summary_md == ''

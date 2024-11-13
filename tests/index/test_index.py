from unittest import mock
from pathlib import Path

from readme_ready.index.index import index
from readme_ready.types import AutodocRepoConfig


def test_index(tmp_path):
    # Create a mock configuration
    config = AutodocRepoConfig(
        name="test_repo",
        repository_url="https://example.com/repo.git",
        root=str(tmp_path),
        output=str(tmp_path),
        llms=None,
        priority=None,
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

    # Mock the imported functions
    with mock.patch(
        "readme_ready.index.index.process_repository"
    ) as mock_process_repository, mock.patch(
        "readme_ready.index.index.convert_json_to_markdown"
    ) as mock_convert_json_to_markdown, mock.patch(
        "readme_ready.index.index.create_vector_store"
    ) as mock_create_vector_store:

        # Run the index function
        index(config)

        # Assert that the directories were created
        json_path = Path(config.output) / "docs" / "json"
        markdown_path = Path(config.output) / "docs" / "markdown"
        data_path = Path(config.output) / "docs" / "data"
        assert json_path.exists()
        assert markdown_path.exists()
        assert data_path.exists()

        # Assert that the functions were called
        assert mock_process_repository.called
        assert mock_convert_json_to_markdown.called
        assert mock_create_vector_store.called

        # Optionally, check the call arguments if necessary
        mock_process_repository.assert_called_once()
        mock_convert_json_to_markdown.assert_called_once()
        mock_create_vector_store.assert_called_once_with(
            str(config.root),
            str(data_path),
            config.ignore,
            config.llms,
            config.device,
        )

import pytest
from unittest.mock import patch

from readme_ready.main import *
from readme_ready.types import LLMModels


@pytest.fixture
def mock_questionary():
    with patch("questionary.text") as mock_text, patch(
        "questionary.path"
    ) as mock_path, patch("questionary.select") as mock_select, patch(
        "questionary.confirm"
    ) as mock_confirm:
        yield mock_text, mock_path, mock_select, mock_confirm


def test_main_readme_mode(mock_questionary):
    mock_text, mock_path, mock_select, mock_confirm = mock_questionary

    # Mock the questionary inputs
    mock_text.return_value.ask.side_effect = [
        "readme_ready",  # name
        "https://github.com/username/readme_ready",  # project_url
        "# Introduction,## Usage",  # headings
    ]

    mock_path.return_value.ask.side_effect = [
        "./readme_ready/",  # project_root
        "./output/readme_ready/",  # output_dir
        None,  # peft_model_path
    ]

    mock_select.return_value.ask.side_effect = [
        "Readme",  # mode
        LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,  # model_name
        "cpu",  # device
    ]

    mock_confirm.return_value.ask.return_value = False  # peft = False

    with patch("readme_ready.index.index.index") as mock_index, patch(
        "readme_ready.query.query.generate_readme"
    ) as mock_generate_readme:

        main()

        # Assert that index and generate_readme were called
        mock_index.assert_called_once()
        mock_generate_readme.assert_called_once()


def test_main_query_mode(mock_questionary):
    mock_text, mock_path, mock_select, mock_confirm = mock_questionary

    # Mock the questionary inputs
    mock_text.return_value.ask.side_effect = [
        "readme_ready",  # name
        "https://github.com/username/readme_ready",  # project_url
    ]

    mock_path.return_value.ask.side_effect = [
        "./readme_ready/",  # project_root
        "./output/readme_ready/",  # output_dir
        None,  # peft_model_path
    ]

    mock_select.return_value.ask.side_effect = [
        "Query",  # mode
        LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,  # model_name
        "cpu",  # device
    ]

    mock_confirm.return_value.ask.return_value = False  # peft = False

    with patch("readme_ready.index.index.index") as mock_index, patch(
        "readme_ready.query.query.query"
    ) as mock_query:

        main()

        # Assert that index and query were called
        mock_index.assert_called_once()
        mock_query.assert_called_once()

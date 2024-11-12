import pytest
from unittest.mock import patch

from doc_generator.main import *
from doc_generator.types import LLMModels


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
        "doc_generator",  # name
        "https://github.com/username/doc_generator",  # project_url
        "# Introduction,## Usage",  # headings
    ]

    mock_path.return_value.ask.side_effect = [
        "./doc_generator/",  # project_root
        "./output/doc_generator/",  # output_dir
        None,  # peft_model_path
    ]

    mock_select.return_value.ask.side_effect = [
        "Readme",  # mode
        LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,  # model_name
        "cpu",  # device
    ]

    mock_confirm.return_value.ask.return_value = False  # peft = False

    with patch("doc_generator.index.index.index") as mock_index, patch(
        "doc_generator.query.query.generate_readme"
    ) as mock_generate_readme:

        main()

        # Assert that index and generate_readme were called
        mock_index.assert_called_once()
        mock_generate_readme.assert_called_once()


def test_main_query_mode(mock_questionary):
    mock_text, mock_path, mock_select, mock_confirm = mock_questionary

    # Mock the questionary inputs
    mock_text.return_value.ask.side_effect = [
        "doc_generator",  # name
        "https://github.com/username/doc_generator",  # project_url
    ]

    mock_path.return_value.ask.side_effect = [
        "./doc_generator/",  # project_root
        "./output/doc_generator/",  # output_dir
        None,  # peft_model_path
    ]

    mock_select.return_value.ask.side_effect = [
        "Query",  # mode
        LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,  # model_name
        "cpu",  # device
    ]

    mock_confirm.return_value.ask.return_value = False  # peft = False

    with patch("doc_generator.index.index.index") as mock_index, patch(
        "doc_generator.query.query.query"
    ) as mock_query:

        main()

        # Assert that index and query were called
        mock_index.assert_called_once()
        mock_query.assert_called_once()

import pytest
from unittest.mock import patch, MagicMock

from doc_generator.main import *
from doc_generator.types import LLMModels


@pytest.fixture
def mock_questionary():
    with patch('questionary.text') as mock_text, \
         patch('questionary.path') as mock_path, \
         patch('questionary.select') as mock_select, \
         patch('questionary.confirm') as mock_confirm:
        yield mock_text, mock_path, mock_select, mock_confirm


def test_main_readme_mode(mock_questionary):
    mock_text, mock_path, mock_select, mock_confirm = mock_questionary

    # Mock the questionary inputs
    mock_text.side_effect = [
        'doc_generator',  # name
        'https://github.com/username/doc_generator',  # project_url
        '# Introduction,## Usage'  # headings
    ]

    mock_path.side_effect = [
        './doc_generator/',  # project_root
        './output/doc_generator/',  # output_dir
        None  # peft_model_path
    ]

    mock_select.side_effect = [
        'Readme',  # mode
        LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,  # model_name
        'cpu'  # device
    ]

    mock_confirm.return_value = False  # peft = False

    with patch('doc_generator.index.index') as mock_index, \
         patch('doc_generator.query.generate_readme') as mock_generate_readme:

        main()

        # Assert that index and generate_readme were called
        mock_index.assert_called_once()
        mock_generate_readme.assert_called_once()


def test_main_query_mode(mock_questionary):
    mock_text, mock_path, mock_select, mock_confirm = mock_questionary

    # Mock the questionary inputs
    mock_text.side_effect = [
        'doc_generator',  # name
        'https://github.com/username/doc_generator'  # project_url
    ]

    mock_path.side_effect = [
        './doc_generator/',  # project_root
        './output/doc_generator/',  # output_dir
        None  # peft_model_path
    ]

    mock_select.side_effect = [
        'Query',  # mode
        LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,  # model_name
        'cpu'  # device
    ]

    mock_confirm.return_value = False  # peft = False

    with patch('doc_generator.index.index') as mock_index, \
         patch('doc_generator.query.query') as mock_query:

        main()

        # Assert that index and query were called
        mock_index.assert_called_once()
        mock_query.assert_called_once()


def test_main_with_peft(mock_questionary):
    mock_text, mock_path, mock_select, mock_confirm = mock_questionary

    # Mock the questionary inputs
    mock_text.side_effect = [
        'doc_generator',  # name
        'https://github.com/username/doc_generator',  # project_url
        '# Introduction,## Usage'  # headings
    ]

    mock_path.side_effect = [
        './doc_generator/',  # project_root
        './output/doc_generator/',  # output_dir
        './output/model/'  # peft_model_path
    ]

    mock_select.side_effect = [
        'Readme',  # mode
        LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,  # model_name
        'cpu'  # device
    ]

    mock_confirm.return_value = True  # peft = True

    with patch('doc_generator.index.index') as mock_index, \
         patch('doc_generator.query.generate_readme') as mock_generate_readme:

        main()

        # Assert that index and generate_readme were called
        mock_index.assert_called_once()
        mock_generate_readme.assert_called_once()


def test_main_all_models(mock_questionary):
    all_models = [model.value for model in LLMModels]

    for model_name in all_models:
        mock_text, mock_path, mock_select, mock_confirm = mock_questionary

        # Mock the questionary inputs
        mock_text.side_effect = [
            'doc_generator',  # name
            'https://github.com/username/doc_generator',  # project_url
            '# Introduction,## Usage'  # headings
        ]

        mock_path.side_effect = [
            './doc_generator/',  # project_root
            './output/doc_generator/',  # output_dir
            None  # peft_model_path
        ]

        mock_select.side_effect = [
            'Readme',  # mode
            model_name,  # model_name
            'cpu'  # device
        ]

        mock_confirm.return_value = False  # peft = False

        with patch('doc_generator.index.index') as mock_index, \
             patch('doc_generator.query.generate_readme') as mock_generate_readme:

            main()

            # Assert that index and generate_readme were called
            mock_index.assert_called_once()
            mock_generate_readme.assert_called_once()

        # Reset mocks for next iteration
        mock_index.reset_mock()
        mock_generate_readme.reset_mock()
        mock_text.reset_mock()
        mock_path.reset_mock()
        mock_select.reset_mock()
        mock_confirm.reset_mock()


def test_main_invalid_url(mock_questionary):
    mock_text, mock_path, mock_select, mock_confirm = mock_questionary

    # Mock the questionary inputs, including an invalid URL first
    mock_text.side_effect = [
        'doc_generator',  # name
        'invalid_url',  # invalid project_url
        'https://github.com/username/doc_generator',  # valid project_url
        '# Introduction,## Usage'  # headings
    ]

    mock_path.side_effect = [
        './doc_generator/',  # project_root
        './output/doc_generator/',  # output_dir
        None  # peft_model_path
    ]

    mock_select.side_effect = [
        'Readme',  # mode
        LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,  # model_name
        'cpu'  # device
    ]

    mock_confirm.return_value = False  # peft = False

    with patch('doc_generator.index.index') as mock_index, \
         patch('doc_generator.query.generate_readme') as mock_generate_readme:

        main()

        # Assert that index and generate_readme were called
        mock_index.assert_called_once()
        mock_generate_readme.assert_called_once()


def test_main_device_gpu(mock_questionary):
    mock_text, mock_path, mock_select, mock_confirm = mock_questionary

    # Mock the questionary inputs
    mock_text.side_effect = [
        'doc_generator',  # name
        'https://github.com/username/doc_generator',  # project_url
        '# Introduction,## Usage'  # headings
    ]

    mock_path.side_effect = [
        './doc_generator/',  # project_root
        './output/doc_generator/',  # output_dir
        None  # peft_model_path
    ]

    mock_select.side_effect = [
        'Readme',  # mode
        LLMModels.TINYLLAMA_1p1B_CHAT_GGUF.value,  # model_name
        'gpu'  # device
    ]

    mock_confirm.return_value = False  # peft = False

    with patch('doc_generator.index.index') as mock_index, \
         patch('doc_generator.query.generate_readme') as mock_generate_readme:

        main()

        # Assert that index and generate_readme were called
        mock_index.assert_called_once()
        mock_generate_readme.assert_called_once()


def test_main_default_model(mock_questionary):
    mock_text, mock_path, mock_select, mock_confirm = mock_questionary

    # Mock the questionary inputs with an invalid model name to trigger default case
    mock_text.side_effect = [
        'doc_generator',  # name
        'https://github.com/username/doc_generator',  # project_url
        '# Introduction,## Usage'  # headings
    ]

    mock_path.side_effect = [
        './doc_generator/',  # project_root
        './output/doc_generator/',  # output_dir
        None  # peft_model_path
    ]

    mock_select.side_effect = [
        'Readme',  # mode
        'InvalidModelName',  # invalid model_name
        'cpu'  # device
    ]

    mock_confirm.return_value = False  # peft = False

    with patch('doc_generator.index.index') as mock_index, \
         patch('doc_generator.query.generate_readme') as mock_generate_readme:

        main()

        # Assert that index and generate_readme were called
        mock_index.assert_called_once()
        mock_generate_readme.assert_called_once()

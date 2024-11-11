import pytest
from unittest import mock
from unittest.mock import MagicMock
from typing import Dict, List
from doc_generator.index.select_model import get_max_prompt_length, select_model
from doc_generator.types import LLMModelDetails, LLMModels, Priority
import tiktoken

def test_get_max_prompt_length():
    prompts = ["hello world", "test prompt"]
    model = LLMModels.GPT3

    # Mock the encoding and encoding_for_model function
    mock_encoding = MagicMock()
    mock_encoding.encode.side_effect = lambda x: [1] * len(x)

    with mock.patch('tiktoken.encoding_for_model', return_value=mock_encoding):
        max_length = get_max_prompt_length(prompts, model)

    expected_length = max(len(prompt) for prompt in prompts)
    assert max_length == expected_length

def test_select_model_cost_priority():
    prompts = ["short prompt"]
    llms = [LLMModels.GPT3, LLMModels.GPT4]
    models = {
        LLMModels.GPT3: LLMModelDetails(name='gpt-3', max_length=50),
        LLMModels.GPT4: LLMModelDetails(name='gpt-4', max_length=100),
    }
    priority = Priority.COST

    # Mock the encoding
    mock_encoding = MagicMock()
    mock_encoding.encode.side_effect = lambda x: [1] * len(x)

    with mock.patch('tiktoken.encoding_for_model', return_value=mock_encoding):
        selected_model = select_model(prompts, llms, models, priority)

    assert selected_model == models[LLMModels.GPT3]

def test_select_model_performance_priority():
    prompts = ["short prompt"]
    llms = [LLMModels.GPT4, LLMModels.GPT3]
    models = {
        LLMModels.GPT3: LLMModelDetails(name='gpt-3', max_length=50),
        LLMModels.GPT4: LLMModelDetails(name='gpt-4', max_length=100),
    }
    priority = Priority.PERFORMANCE

    # Mock the encoding
    mock_encoding = MagicMock()
    mock_encoding.encode.side_effect = lambda x: [1] * len(x)

    with mock.patch('tiktoken.encoding_for_model', return_value=mock_encoding):
        selected_model = select_model(prompts, llms, models, priority)

    assert selected_model == models[LLMModels.GPT4]

def test_select_model_no_model_found():
    prompts = ["this is a very long prompt that exceeds model max length"]
    llms = [LLMModels.GPT3]
    models = {
        LLMModels.GPT3: LLMModelDetails(name='gpt-3', max_length=10),
    }
    priority = Priority.COST

    # Mock the encoding to return a large token list
    mock_encoding = MagicMock()
    mock_encoding.encode.side_effect = lambda x: [1] * 50

    with mock.patch('tiktoken.encoding_for_model', return_value=mock_encoding):
        selected_model = select_model(prompts, llms, models, priority)

    assert selected_model is None

def test_select_model_unknown_priority():
    prompts = ["prompt"]
    llms = [LLMModels.GPT3]
    models = {
        LLMModels.GPT3: LLMModelDetails(name='gpt-3', max_length=50),
    }
    priority = None  # Unknown priority

    # Mock the encoding
    mock_encoding = MagicMock()
    mock_encoding.encode.side_effect = lambda x: [1] * len(x)

    with mock.patch('tiktoken.encoding_for_model', return_value=mock_encoding):
        selected_model = select_model(prompts, llms, models, priority)

    assert selected_model is None

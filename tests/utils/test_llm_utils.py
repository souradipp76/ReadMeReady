import os
from unittest import mock
import pytest
from unittest.mock import patch, MagicMock


from readme_ready.utils.llm_utils import (
    get_gemma_chat_model,
    get_llama_chat_model,
    get_openai_chat_model,
    get_openai_api_key,
    get_tokenizer,
    print_model_details,
    total_index_cost_estimate,
    get_embeddings,
)
from readme_ready.types import LLMModelDetails
from langchain_openai import ChatOpenAI


def test_get_gemma_chat_model_with_peft():
    model_name = "some-model"
    model_kwargs = {
        "gguf_file": "some_file.gguf",
        "device": "cpu",
        "peft_model_path": "path/to/peft/model",
    }
    with patch(
        "readme_ready.utils.llm_utils.hf_hub_download"
    ) as mock_hf_download, patch(
        "readme_ready.utils.llm_utils.get_tokenizer"
    ) as mock_get_tokenizer, patch(
        "readme_ready.utils.llm_utils.AutoModelForCausalLM.from_pretrained"
    ) as mock_auto_model, patch(
        "readme_ready.utils.llm_utils.PeftModel.from_pretrained"
    ) as mock_peft_model, patch(
        "readme_ready.utils.llm_utils.pipeline"
    ) as mock_pipeline, patch(
        "readme_ready.utils.llm_utils.HuggingFacePipeline"
    ) as mock_hf_pipeline, patch.dict(
        os.environ, {"HF_TOKEN": "test_token"}
    ):

        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_auto_model.return_value = mock_model

        mock_peft_model_instance = MagicMock()
        mock_peft_model.return_value = mock_peft_model_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_hf_pipeline_instance = MagicMock()
        mock_hf_pipeline.return_value = mock_hf_pipeline_instance

        result = get_gemma_chat_model(model_name, model_kwargs=model_kwargs)

        mock_hf_download.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        mock_get_tokenizer.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        mock_auto_model.assert_called_once_with(
            model_name,
            gguf_file=model_kwargs["gguf_file"],
            trust_remote_code=True,
            device_map=model_kwargs["device"],
            quantization_config=mock.ANY,
            token="test_token",
        )
        mock_peft_model.assert_called_once_with(
            mock_model, model_kwargs["peft_model_path"]
        )
        mock_pipeline.assert_called_once()
        mock_hf_pipeline.assert_called_once_with(
            pipeline=mock_pipeline_instance, model_kwargs=model_kwargs
        )
        assert result == mock_hf_pipeline_instance


def test_get_gemma_chat_model_without_peft():
    model_name = "some-model"
    model_kwargs = {
        "gguf_file": "some_file.gguf",
        "device": "cpu",
    }
    with patch(
        "readme_ready.utils.llm_utils.hf_hub_download"
    ) as mock_hf_download, patch(
        "readme_ready.utils.llm_utils.get_tokenizer"
    ) as mock_get_tokenizer, patch(
        "readme_ready.utils.llm_utils.AutoModelForCausalLM.from_pretrained"
    ) as mock_auto_model, patch(
        "readme_ready.utils.llm_utils.PeftModel.from_pretrained"
    ) as mock_peft_model, patch(
        "readme_ready.utils.llm_utils.pipeline"
    ) as mock_pipeline, patch(
        "readme_ready.utils.llm_utils.HuggingFacePipeline"
    ) as mock_hf_pipeline, patch.dict(
        os.environ, {"HF_TOKEN": "test_token"}
    ):

        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_auto_model.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_hf_pipeline_instance = MagicMock()
        mock_hf_pipeline.return_value = mock_hf_pipeline_instance

        result = get_gemma_chat_model(model_name, model_kwargs=model_kwargs)

        mock_hf_download.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        mock_get_tokenizer.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        mock_auto_model.assert_called_once_with(
            model_name,
            gguf_file=model_kwargs["gguf_file"],
            trust_remote_code=True,
            device_map=model_kwargs["device"],
            quantization_config=mock.ANY,
            token="test_token",
        )
        mock_peft_model.assert_not_called()
        mock_pipeline.assert_called_once()
        mock_hf_pipeline.assert_called_once_with(
            pipeline=mock_pipeline_instance, model_kwargs=model_kwargs
        )
        assert result == mock_hf_pipeline_instance


def test_get_gemma_chat_model_with_bnb_config():
    model_name = "some-model"
    model_kwargs = {
        "gguf_file": "some_file.gguf",
        "device": "cpu",
    }
    with patch("sys.platform", "linux"), patch(
        "readme_ready.utils.llm_utils.hf_hub_download"
    ) as mock_hf_download, patch(
        "readme_ready.utils.llm_utils.get_tokenizer"
    ) as mock_get_tokenizer, patch(
        "readme_ready.utils.llm_utils.AutoModelForCausalLM.from_pretrained"
    ) as mock_auto_model, patch(
        "readme_ready.utils.llm_utils.PeftModel.from_pretrained"
    ) as mock_peft_model, patch(
        "readme_ready.utils.llm_utils.pipeline"
    ) as mock_pipeline, patch(
        "readme_ready.utils.llm_utils.HuggingFacePipeline"
    ) as mock_hf_pipeline, patch.dict(
        os.environ, {"HF_TOKEN": "test_token"}
    ):

        mock_bnb_config = MagicMock()

        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_auto_model.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_hf_pipeline_instance = MagicMock()
        mock_hf_pipeline.return_value = mock_hf_pipeline_instance

        result = get_gemma_chat_model(model_name, model_kwargs=model_kwargs)

        mock_hf_download.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        mock_get_tokenizer.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        mock_auto_model.assert_called_once_with(
            model_name,
            gguf_file=model_kwargs["gguf_file"],
            trust_remote_code=True,
            device_map=model_kwargs["device"],
            quantization_config=mock.ANY,
            token="test_token",
        )
        mock_peft_model.assert_not_called()
        mock_pipeline.assert_called_once()
        mock_hf_pipeline.assert_called_once_with(
            pipeline=mock_pipeline_instance, model_kwargs=model_kwargs
        )
        assert result == mock_hf_pipeline_instance


def test_get_llama_chat_model_with_peft():
    model_name = "some-model"
    model_kwargs = {
        "gguf_file": "some_file.gguf",
        "device": "cpu",
        "peft_model": "path/to/peft/model",
    }
    with patch(
        "readme_ready.utils.llm_utils.hf_hub_download"
    ) as mock_hf_download, patch(
        "readme_ready.utils.llm_utils.get_tokenizer"
    ) as mock_get_tokenizer, patch(
        "readme_ready.utils.llm_utils.AutoModelForCausalLM.from_pretrained"
    ) as mock_auto_model, patch(
        "readme_ready.utils.llm_utils.PeftModel.from_pretrained"
    ) as mock_peft_model, patch(
        "readme_ready.utils.llm_utils.pipeline"
    ) as mock_pipeline, patch(
        "readme_ready.utils.llm_utils.HuggingFacePipeline"
    ) as mock_hf_pipeline, patch.dict(
        os.environ, {"HF_TOKEN": "test_token"}
    ):

        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model = MagicMock()
        mock_auto_model.return_value = mock_model

        mock_peft_model_instance = MagicMock()
        mock_peft_model.return_value = mock_peft_model_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_hf_pipeline_instance = MagicMock()
        mock_hf_pipeline.return_value = mock_hf_pipeline_instance

        result = get_llama_chat_model(model_name, model_kwargs=model_kwargs)

        mock_hf_download.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        mock_get_tokenizer.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        assert mock_tokenizer.pad_token == mock_tokenizer.eos_token
        mock_auto_model.assert_called_once_with(
            model_name,
            gguf_file=model_kwargs["gguf_file"],
            trust_remote_code=True,
            device_map=model_kwargs["device"],
            quantization_config=mock.ANY,
        )
        mock_peft_model.assert_called_once_with(
            mock_model, model_kwargs["peft_model"]
        )
        mock_pipeline.assert_called_once()
        mock_hf_pipeline.assert_called_once_with(
            pipeline=mock_pipeline_instance, model_kwargs=model_kwargs
        )
        assert result == mock_hf_pipeline_instance


def test_get_llama_chat_model_without_peft():
    model_name = "some-model"
    model_kwargs = {
        "gguf_file": "some_file.gguf",
        "device": "cpu",
    }
    with patch(
        "readme_ready.utils.llm_utils.hf_hub_download"
    ) as mock_hf_download, patch(
        "readme_ready.utils.llm_utils.get_tokenizer"
    ) as mock_get_tokenizer, patch(
        "readme_ready.utils.llm_utils.AutoModelForCausalLM.from_pretrained"
    ) as mock_auto_model, patch(
        "readme_ready.utils.llm_utils.PeftModel.from_pretrained"
    ) as mock_peft_model, patch(
        "readme_ready.utils.llm_utils.pipeline"
    ) as mock_pipeline, patch(
        "readme_ready.utils.llm_utils.HuggingFacePipeline"
    ) as mock_hf_pipeline, patch.dict(
        os.environ, {"HF_TOKEN": "test_token"}
    ):

        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model = MagicMock()
        mock_auto_model.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_hf_pipeline_instance = MagicMock()
        mock_hf_pipeline.return_value = mock_hf_pipeline_instance

        result = get_llama_chat_model(model_name, model_kwargs=model_kwargs)

        mock_hf_download.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        mock_get_tokenizer.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        assert mock_tokenizer.pad_token == mock_tokenizer.eos_token
        mock_auto_model.assert_called_once_with(
            model_name,
            gguf_file=model_kwargs["gguf_file"],
            trust_remote_code=True,
            device_map=model_kwargs["device"],
            quantization_config=mock.ANY,
        )
        mock_peft_model.assert_not_called()
        mock_pipeline.assert_called_once()
        mock_hf_pipeline.assert_called_once_with(
            pipeline=mock_pipeline_instance, model_kwargs=model_kwargs
        )
        assert result == mock_hf_pipeline_instance


def test_get_llama_chat_model_with_bnb_config():
    model_name = "some-model"
    model_kwargs = {
        "gguf_file": "some_file.gguf",
        "device": "cpu",
    }
    with patch("sys.platform", "linux"), patch(
        "readme_ready.utils.llm_utils.hf_hub_download"
    ) as mock_hf_download, patch(
        "readme_ready.utils.llm_utils.get_tokenizer"
    ) as mock_get_tokenizer, patch(
        "readme_ready.utils.llm_utils.AutoModelForCausalLM.from_pretrained"
    ) as mock_auto_model, patch(
        "readme_ready.utils.llm_utils.PeftModel.from_pretrained"
    ) as mock_peft_model, patch(
        "readme_ready.utils.llm_utils.pipeline"
    ) as mock_pipeline, patch(
        "readme_ready.utils.llm_utils.HuggingFacePipeline"
    ) as mock_hf_pipeline, patch.dict(
        os.environ, {"HF_TOKEN": "test_token"}
    ):

        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model = MagicMock()
        mock_auto_model.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_hf_pipeline_instance = MagicMock()
        mock_hf_pipeline.return_value = mock_hf_pipeline_instance

        result = get_llama_chat_model(model_name, model_kwargs=model_kwargs)

        mock_hf_download.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        mock_get_tokenizer.assert_called_once_with(
            model_name, model_kwargs["gguf_file"]
        )
        assert mock_tokenizer.pad_token == mock_tokenizer.eos_token
        mock_auto_model.assert_called_once_with(
            model_name,
            gguf_file=model_kwargs["gguf_file"],
            trust_remote_code=True,
            device_map=model_kwargs["device"],
            quantization_config=mock.ANY,
        )
        mock_peft_model.assert_not_called()
        mock_pipeline.assert_called_once()
        mock_hf_pipeline.assert_called_once_with(
            pipeline=mock_pipeline_instance, model_kwargs=model_kwargs
        )
        assert result == mock_hf_pipeline_instance


def test_get_openai_chat_model():
    model = "gpt-3.5-turbo"
    temperature = 0.7
    streaming = True
    model_kwargs = {
        "frequency_penalty": "0.0",
        "presence_penalty": "0.0",
    }

    result = get_openai_chat_model(model, temperature, streaming, model_kwargs)

    assert isinstance(result, ChatOpenAI)
    assert result.temperature == temperature
    assert result.streaming == streaming
    assert result.model_name == model


def test_get_openai_api_key_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
    api_key = get_openai_api_key()
    assert api_key == "test_api_key"


def test_get_openai_api_key_not_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    api_key = get_openai_api_key()
    assert api_key == ""


def test_get_tokenizer_with_hf_token(monkeypatch):
    model_name = "some-model"
    gguf_file = "some_file.gguf"
    with patch(
        "readme_ready.utils.llm_utils.AutoTokenizer.from_pretrained"
    ) as mock_from_pretrained:
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer

        monkeypatch.setenv("HF_TOKEN", "test_token")
        tokenizer = get_tokenizer(model_name, gguf_file)

        mock_from_pretrained.assert_called_once_with(
            model_name,
            gguf_file=gguf_file,
            token="test_token",
        )
        assert tokenizer == mock_tokenizer


def test_get_tokenizer_without_hf_token(monkeypatch):
    model_name = "some-model"
    gguf_file = "some_file.gguf"
    with patch(
        "readme_ready.utils.llm_utils.AutoTokenizer.from_pretrained"
    ) as mock_from_pretrained:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(KeyError):
            get_tokenizer(model_name, gguf_file)


def test_print_model_details(capsys):
    test_models = {
        "model1": LLMModelDetails(
            name="model1",
            input_cost_per_1k_tokens=0.01,
            output_cost_per_1k_tokens=0.02,
            max_length=1000,
            llm=None,
            input_tokens=5000,
            output_tokens=3000,
            succeeded=5,
            failed=2,
            total=7,
        ),
        "model2": LLMModelDetails(
            name="model2",
            input_cost_per_1k_tokens=0.015,
            output_cost_per_1k_tokens=0.025,
            max_length=2000,
            llm=None,
            input_tokens=7000,
            output_tokens=4000,
            succeeded=6,
            failed=1,
            total=7,
        ),
    }

    print_model_details(test_models)

    captured = capsys.readouterr()
    output_lines = captured.out.strip().split("\n")

    expected_outputs = []
    for model_details in test_models.values():
        tokens = model_details.input_tokens + model_details.output_tokens
        cost = (
            (model_details.input_tokens / 1000)
            * model_details.input_cost_per_1k_tokens
        ) + (
            (model_details.output_tokens / 1000)
            * model_details.output_cost_per_1k_tokens
        )
        result = {
            "Model": model_details.name,
            "File Count": model_details.total,
            "Succeeded": model_details.succeeded,
            "Failed": model_details.failed,
            "Tokens": tokens,
            "Cost": cost,
        }
        expected_outputs.append(result)

    totals = {
        "Model": "Total",
        "File Count": sum(item["File Count"] for item in expected_outputs),
        "Succeeded": sum(item["Succeeded"] for item in expected_outputs),
        "Failed": sum(item["Failed"] for item in expected_outputs),
        "Tokens": sum(item["Tokens"] for item in expected_outputs),
        "Cost": sum(item["Cost"] for item in expected_outputs),
    }

    all_results = expected_outputs + [totals]

    for i, line in enumerate(output_lines):
        expected_line = str(all_results[i])
        assert line.strip() == expected_line


def test_print_model_details_empty(capsys):
    test_models = {}
    print_model_details(test_models)
    captured = capsys.readouterr()
    output_lines = captured.out.strip().split("\n")
    assert output_lines == [
        "{'Model': 'Total', 'File Count': 0, 'Succeeded': 0, 'Failed': 0, 'Tokens': 0, 'Cost': 0}"
    ]


def test_total_index_cost_estimate():
    test_models = {
        "model1": LLMModelDetails(
            name="model1",
            input_cost_per_1k_tokens=0.01,
            output_cost_per_1k_tokens=0.02,
            max_length=1000,
            llm=None,
            input_tokens=5000,
            output_tokens=3000,
            succeeded=5,
            failed=2,
            total=7,
        ),
        "model2": LLMModelDetails(
            name="model2",
            input_cost_per_1k_tokens=0.015,
            output_cost_per_1k_tokens=0.025,
            max_length=2000,
            llm=None,
            input_tokens=7000,
            output_tokens=4000,
            succeeded=6,
            failed=1,
            total=7,
        ),
    }

    with patch("readme_ready.utils.llm_utils.models", test_models):
        total_cost = total_index_cost_estimate(None)

    expected_cost = sum(
        (model_details.input_tokens / 1000)
        * model_details.input_cost_per_1k_tokens
        + (model_details.output_tokens / 1000)
        * model_details.output_cost_per_1k_tokens
        for model_details in test_models.values()
    )
    assert total_cost == expected_cost


def test_get_embeddings_llama_model():
    model = "llama-something"
    device = "cpu"

    with patch(
        "readme_ready.utils.llm_utils.HuggingFaceEmbeddings"
    ) as mock_hf_embeddings:
        embeddings = get_embeddings(model, device)
        mock_hf_embeddings.assert_called_once_with(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        assert embeddings == mock_hf_embeddings.return_value


def test_get_embeddings_non_llama_model():
    model = "gpt-3.5-turbo"
    device = "cpu"

    with patch(
        "readme_ready.utils.llm_utils.OpenAIEmbeddings"
    ) as mock_openai_embeddings:
        embeddings = get_embeddings(model, device)
        mock_openai_embeddings.assert_called_once_with()
        assert embeddings == mock_openai_embeddings.return_value

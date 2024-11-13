from unittest.mock import MagicMock, patch
from readme_ready.query.create_chat_chain import (
    make_qa_prompt,
    make_readme_prompt,
    make_qa_chain,
    make_readme_chain,
    condense_qa_prompt,
    condense_readme_prompt,
)
from langchain.prompts import PromptTemplate
from readme_ready.utils.llm_utils import (
    models,
)


def test_make_qa_prompt_with_chat_prompt():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Please provide detailed explanations."
    target_audience = "developers"

    prompt = make_qa_prompt(
        project_name,
        repository_url,
        content_type,
        chat_prompt,
        target_audience,
    )

    assert isinstance(prompt, PromptTemplate)
    assert "{input}" in prompt.template
    assert "{context}" in prompt.template
    assert "Please provide detailed explanations." in prompt.template
    assert prompt.input_variables == ["context", "input"]


def test_make_qa_prompt_without_chat_prompt():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = ""
    target_audience = "developers"

    prompt = make_qa_prompt(
        project_name,
        repository_url,
        content_type,
        chat_prompt,
        target_audience,
    )

    assert isinstance(prompt, PromptTemplate)
    assert "{input}" in prompt.template
    assert "{context}" in prompt.template
    assert "Here are some additional instructions" not in prompt.template
    assert prompt.input_variables == ["context", "input"]


def test_make_readme_prompt_with_chat_prompt():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Ensure examples are included."
    target_audience = "developers"

    prompt = make_readme_prompt(
        project_name,
        repository_url,
        content_type,
        chat_prompt,
        target_audience,
    )

    assert isinstance(prompt, PromptTemplate)
    assert "{input}" in prompt.template
    assert "{context}" in prompt.template
    assert "Ensure examples are included." in prompt.template
    assert prompt.input_variables == ["context", "input"]


def test_make_readme_prompt_without_chat_prompt():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = ""
    target_audience = "developers"

    prompt = make_readme_prompt(
        project_name,
        repository_url,
        content_type,
        chat_prompt,
        target_audience,
    )

    assert isinstance(prompt, PromptTemplate)
    assert "{input}" in prompt.template
    assert "{context}" in prompt.template
    assert "Here are some additional instructions" not in prompt.template
    assert prompt.input_variables == ["context", "input"]


def test_make_qa_chain_llama():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Provide code examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm = MagicMock()
    llm.value = "llama"
    llm.name = "llama"
    llms = [llm]
    device = "cpu"

    with patch(
        "readme_ready.query.create_chat_chain.get_llama_chat_model"
    ) as mock_get_llama_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_history_aware_retriever"
    ) as mock_llm_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_chat_vector_chain:
        mock_question_chat_model = MagicMock()
        mock_get_llama_chat_model.return_value = mock_question_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_question_generator = MagicMock()
        mock_llm_chain.return_value = mock_question_generator

        mock_chat_chain_instance = MagicMock()
        mock_chat_vector_chain.return_value = mock_chat_chain_instance

        chain = make_qa_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            device=device,
            on_token_stream=False,
        )

        mock_get_llama_chat_model.assert_called()
        mock_create_stuff_chain.assert_called()
        mock_llm_chain.assert_called()
        mock_chat_vector_chain.assert_called_with(
            retriever=mock_question_generator,
            combine_docs_chain=mock_doc_chain,
        )
        assert chain == mock_chat_chain_instance


def test_make_qa_chain_gemma():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Provide code examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm = MagicMock()
    llm.value = "gemma"
    llm.name = "gemma"
    llms = [llm]
    device = "cpu"

    with patch(
        "readme_ready.query.create_chat_chain.get_gemma_chat_model"
    ) as mock_get_gemma_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_history_aware_retriever"
    ) as mock_llm_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_chat_vector_chain:
        mock_question_chat_model = MagicMock()
        mock_get_gemma_chat_model.return_value = mock_question_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_question_generator = MagicMock()
        mock_llm_chain.return_value = mock_question_generator

        mock_chat_chain_instance = MagicMock()
        mock_chat_vector_chain.return_value = mock_chat_chain_instance

        chain = make_qa_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            device=device,
            on_token_stream=False,
        )

        mock_get_gemma_chat_model.assert_called()
        mock_create_stuff_chain.assert_called()
        mock_llm_chain.assert_called()
        mock_chat_vector_chain.assert_called_with(
            retriever=mock_question_generator,
            combine_docs_chain=mock_doc_chain,
        )
        assert chain == mock_chat_chain_instance


def test_make_qa_chain_openai():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Provide code examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm = MagicMock()
    llm.value = "gpt-4"
    llm.name = "openai"
    llms = [llm]
    device = "cpu"

    with patch(
        "readme_ready.query.create_chat_chain.get_openai_chat_model"
    ) as mock_get_openai_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_history_aware_retriever"
    ) as mock_llm_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_chat_vector_chain:
        mock_question_chat_model = MagicMock()
        mock_get_openai_chat_model.return_value = mock_question_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_question_generator = MagicMock()
        mock_llm_chain.return_value = mock_question_generator

        mock_chat_chain_instance = MagicMock()
        mock_chat_vector_chain.return_value = mock_chat_chain_instance

        chain = make_qa_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            device=device,
            on_token_stream=False,
        )

        mock_get_openai_chat_model.assert_called()
        mock_create_stuff_chain.assert_called()
        mock_llm_chain.assert_called()
        mock_chat_vector_chain.assert_called_with(
            retriever=mock_question_generator,
            combine_docs_chain=mock_doc_chain,
        )
        assert chain == mock_chat_chain_instance


def test_make_readme_chain_llama():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Include examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm = MagicMock()
    llm.value = "llama"
    llm.name = "llama"
    llms = [llm]
    device = "cpu"
    peft_model = None

    with patch(
        "readme_ready.query.create_chat_chain.get_llama_chat_model"
    ) as mock_get_llama_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_create_retrieval_chain:
        mock_doc_chat_model = MagicMock()
        mock_get_llama_chat_model.return_value = mock_doc_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_retrieval_chain_instance = MagicMock()
        mock_create_retrieval_chain.return_value = (
            mock_retrieval_chain_instance
        )

        chain = make_readme_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            peft_model=peft_model,
            device=device,
            on_token_stream=False,
        )

        mock_get_llama_chat_model.assert_called()
        mock_create_stuff_chain.assert_called()
        mock_create_retrieval_chain.assert_called_with(
            retriever=vectorstore.as_retriever(),
            combine_docs_chain=mock_doc_chain,
        )
        assert chain == mock_retrieval_chain_instance


def test_make_readme_chain_gemma():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Include examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm = MagicMock()
    llm.value = "gemma"
    llm.name = "gemma"
    llms = [llm]
    device = "cpu"
    peft_model = None

    with patch(
        "readme_ready.query.create_chat_chain.get_gemma_chat_model"
    ) as mock_get_gemma_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_create_retrieval_chain:
        mock_doc_chat_model = MagicMock()
        mock_get_gemma_chat_model.return_value = mock_doc_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_retrieval_chain_instance = MagicMock()
        mock_create_retrieval_chain.return_value = (
            mock_retrieval_chain_instance
        )

        chain = make_readme_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            peft_model=peft_model,
            device=device,
            on_token_stream=False,
        )

        mock_get_gemma_chat_model.assert_called()
        mock_create_stuff_chain.assert_called()
        mock_create_retrieval_chain.assert_called_with(
            retriever=vectorstore.as_retriever(),
            combine_docs_chain=mock_doc_chain,
        )
        assert chain == mock_retrieval_chain_instance


def test_make_readme_chain_openai():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Include examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm = MagicMock()
    llm.value = "gpt-4"
    llm.name = "openai"
    llms = [llm]
    device = "cpu"
    peft_model = None

    with patch(
        "readme_ready.query.create_chat_chain.get_openai_chat_model"
    ) as mock_get_openai_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_create_retrieval_chain:
        mock_doc_chat_model = MagicMock()
        mock_get_openai_chat_model.return_value = mock_doc_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_retrieval_chain_instance = MagicMock()
        mock_create_retrieval_chain.return_value = (
            mock_retrieval_chain_instance
        )

        chain = make_readme_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            peft_model=peft_model,
            device=device,
            on_token_stream=False,
        )

        mock_get_openai_chat_model.assert_called()
        mock_create_stuff_chain.assert_called()
        mock_create_retrieval_chain.assert_called_with(
            retriever=vectorstore.as_retriever(),
            combine_docs_chain=mock_doc_chain,
        )
        assert chain == mock_retrieval_chain_instance


def test_condense_qa_prompt():
    assert isinstance(condense_qa_prompt, PromptTemplate)
    assert "chat_history" in condense_qa_prompt.input_variables
    assert "input" in condense_qa_prompt.input_variables


def test_condense_readme_prompt():
    assert isinstance(condense_readme_prompt, PromptTemplate)
    assert "input" in condense_readme_prompt.input_variables


def test_make_qa_chain_with_multiple_llms():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Provide code examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm1 = MagicMock()
    llm1.value = "llama"
    llm1.name = "llama"
    llm2 = MagicMock()
    llm2.value = "gpt-4"
    llm2.name = "openai"
    llms = [llm1, llm2]
    device = "cpu"

    with patch(
        "readme_ready.query.create_chat_chain.get_openai_chat_model"
    ) as mock_get_openai_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_history_aware_retriever"
    ) as mock_llm_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_chat_vector_chain:
        mock_question_chat_model = MagicMock()
        mock_get_openai_chat_model.return_value = mock_question_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_question_generator = MagicMock()
        mock_llm_chain.return_value = mock_question_generator

        mock_chat_chain_instance = MagicMock()
        mock_chat_vector_chain.return_value = mock_chat_chain_instance

        chain = make_qa_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            device=device,
            on_token_stream=False,
        )

        mock_get_openai_chat_model.assert_called()
        assert chain == mock_chat_chain_instance


def test_make_qa_chain_with_llama_gguf_file():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Include examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm = MagicMock()
    llm.value = "llama-gguf"
    llms = [llm]
    device = "cpu"
    peft_model = None

    models[llm] = MagicMock()
    models[llm].gguf_file = "path/to/gguf_file"

    with patch(
        "readme_ready.query.create_chat_chain.get_llama_chat_model"
    ) as mock_get_llama_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_history_aware_retriever"
    ) as mock_llm_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_chat_vector_chain:
        mock_question_chat_model = MagicMock()
        mock_get_llama_chat_model.return_value = mock_question_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_question_generator = MagicMock()
        mock_llm_chain.return_value = mock_question_generator

        mock_chat_chain_instance = MagicMock()
        mock_chat_vector_chain.return_value = mock_chat_chain_instance

        chain = make_qa_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            device=device,
            on_token_stream=False,
        )

        mock_get_llama_chat_model.assert_called()
        model_kwargs = mock_get_llama_chat_model.call_args[1]["model_kwargs"]
        assert "gguf_file" in model_kwargs
        assert model_kwargs["gguf_file"] == "path/to/gguf_file"
        assert chain == mock_chat_chain_instance


def test_make_qa_chain_with_gemma_gguf_file():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Include examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm = MagicMock()
    llm.value = "gemma-gguf"
    llms = [llm]
    device = "cpu"
    peft_model = None

    models[llm] = MagicMock()
    models[llm].gguf_file = "path/to/gguf_file"

    with patch(
        "readme_ready.query.create_chat_chain.get_gemma_chat_model"
    ) as mock_get_gemma_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_history_aware_retriever"
    ) as mock_llm_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_chat_vector_chain:
        mock_question_chat_model = MagicMock()
        mock_get_gemma_chat_model.return_value = mock_question_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_question_generator = MagicMock()
        mock_llm_chain.return_value = mock_question_generator

        mock_chat_chain_instance = MagicMock()
        mock_chat_vector_chain.return_value = mock_chat_chain_instance

        chain = make_qa_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            device=device,
            on_token_stream=False,
        )

        mock_get_gemma_chat_model.assert_called()
        model_kwargs = mock_get_gemma_chat_model.call_args[1]["model_kwargs"]
        assert "gguf_file" in model_kwargs
        assert model_kwargs["gguf_file"] == "path/to/gguf_file"
        assert chain == mock_chat_chain_instance


def test_make_readme_chain_with_llama_gguf_file():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Include examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm = MagicMock()
    llm.value = "llama-gguf"
    llms = [llm]
    device = "cpu"
    peft_model = None

    models[llm] = MagicMock()
    models[llm].gguf_file = "path/to/gguf_file"

    with patch(
        "readme_ready.query.create_chat_chain.get_llama_chat_model"
    ) as mock_get_llama_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_create_retrieval_chain:
        mock_doc_chat_model = MagicMock()
        mock_get_llama_chat_model.return_value = mock_doc_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_retrieval_chain_instance = MagicMock()
        mock_create_retrieval_chain.return_value = (
            mock_retrieval_chain_instance
        )

        chain = make_readme_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            peft_model=peft_model,
            device=device,
            on_token_stream=False,
        )

        mock_get_llama_chat_model.assert_called()
        model_kwargs = mock_get_llama_chat_model.call_args[1]["model_kwargs"]
        assert "gguf_file" in model_kwargs
        assert model_kwargs["gguf_file"] == "path/to/gguf_file"
        assert chain == mock_retrieval_chain_instance


def test_make_readme_chain_with_gemma_gguf_file():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Include examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm = MagicMock()
    llm.value = "gemma-gguf"
    llms = [llm]
    device = "cpu"
    peft_model = None

    models[llm] = MagicMock()
    models[llm].gguf_file = "path/to/gguf_file"

    with patch(
        "readme_ready.query.create_chat_chain.get_gemma_chat_model"
    ) as mock_get_gemma_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_create_retrieval_chain:
        mock_doc_chat_model = MagicMock()
        mock_get_gemma_chat_model.return_value = mock_doc_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_retrieval_chain_instance = MagicMock()
        mock_create_retrieval_chain.return_value = (
            mock_retrieval_chain_instance
        )

        chain = make_readme_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            peft_model=peft_model,
            device=device,
            on_token_stream=False,
        )

        mock_get_gemma_chat_model.assert_called()
        model_kwargs = mock_get_gemma_chat_model.call_args[1]["model_kwargs"]
        assert "gguf_file" in model_kwargs
        assert model_kwargs["gguf_file"] == "path/to/gguf_file"
        assert chain == mock_retrieval_chain_instance


def test_make_qa_chain_with_on_token_stream_true():
    project_name = "TestProject"
    repository_url = "https://github.com/test/testproject"
    content_type = "codebase"
    chat_prompt = "Provide code examples."
    target_audience = "developers"
    vectorstore = MagicMock()
    llm = MagicMock()
    llm.value = "gpt-4"
    llms = [llm]
    device = "cpu"

    with patch(
        "readme_ready.query.create_chat_chain.get_openai_chat_model"
    ) as mock_get_openai_chat_model, patch(
        "readme_ready.query.create_chat_chain.create_stuff_documents_chain"
    ) as mock_create_stuff_chain, patch(
        "readme_ready.query.create_chat_chain.create_history_aware_retriever"
    ) as mock_llm_chain, patch(
        "readme_ready.query.create_chat_chain.create_retrieval_chain"
    ) as mock_chat_vector_chain:
        mock_question_chat_model = MagicMock()
        mock_get_openai_chat_model.return_value = mock_question_chat_model

        mock_doc_chain = MagicMock()
        mock_create_stuff_chain.return_value = mock_doc_chain

        mock_question_generator = MagicMock()
        mock_llm_chain.return_value = mock_question_generator

        mock_chat_chain_instance = MagicMock()
        mock_chat_vector_chain.return_value = mock_chat_chain_instance

        chain = make_qa_chain(
            project_name,
            repository_url,
            content_type,
            chat_prompt,
            target_audience,
            vectorstore,
            llms,
            device=device,
            on_token_stream=True,
        )

        mock_get_openai_chat_model.assert_called()
        streaming = mock_get_openai_chat_model.call_args[1]["streaming"]
        assert streaming is True
        assert chain == mock_chat_chain_instance

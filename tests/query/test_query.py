from unittest.mock import MagicMock, patch, call
import os
from readme_ready.query import query


def test_display_welcome_message(capsys):
    project_name = "TestProject"
    query.display_welcome_message(project_name)
    captured = capsys.readouterr()
    assert f"Welcome to the {project_name} chatbot." in captured.out
    assert (
        f"Ask any questions related to the {project_name} codebase"
        in captured.out
    )


def test_init_qa_chain():
    repo_config = MagicMock()
    repo_config.output = "output_dir"
    repo_config.llms = [MagicMock()]
    repo_config.llms[0].value = "llm_value"
    repo_config.device = "cpu"
    repo_config.name = "TestProject"
    repo_config.repository_url = "https://github.com/test/testproject"
    repo_config.content_type = "code"
    repo_config.chat_prompt = "chat_prompt"
    repo_config.target_audience = "developers"

    user_config = MagicMock()
    user_config.llms = ["llm1"]
    user_config.streaming = False

    with patch(
        "readme_ready.query.query.get_embeddings"
    ) as mock_get_embeddings, patch(
        "readme_ready.query.query.HNSWLib.load"
    ) as mock_hnswlib_load, patch(
        "readme_ready.query.query.make_qa_chain"
    ) as mock_make_qa_chain:

        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings

        mock_vector_store = MagicMock()
        mock_hnswlib_load.return_value = mock_vector_store

        mock_chain = MagicMock()
        mock_make_qa_chain.return_value = mock_chain

        chain = query.init_qa_chain(repo_config, user_config)

        mock_get_embeddings.assert_called_once_with(
            repo_config.llms[0].value, repo_config.device
        )
        data_path = os.path.join(repo_config.output, "docs", "data")
        mock_hnswlib_load.assert_called_once_with(data_path, mock_embeddings)
        mock_make_qa_chain.assert_called_once_with(
            repo_config.name,
            repo_config.repository_url,
            repo_config.content_type,
            repo_config.chat_prompt,
            repo_config.target_audience,
            mock_vector_store,
            user_config.llms,
            on_token_stream=user_config.streaming,
        )
        assert chain == mock_chain


def test_init_readme_chain():
    repo_config = MagicMock()
    repo_config.output = "output_dir"
    repo_config.llms = [MagicMock()]
    repo_config.llms[0].value = "llm_value"
    repo_config.device = "cpu"
    repo_config.name = "TestProject"
    repo_config.repository_url = "https://github.com/test/testproject"
    repo_config.content_type = "code"
    repo_config.chat_prompt = "chat_prompt"
    repo_config.target_audience = "developers"
    repo_config.peft_model_path = "peft_model_path"

    user_config = MagicMock()
    user_config.llms = ["llm1"]
    user_config.streaming = False

    with patch(
        "readme_ready.query.query.get_embeddings"
    ) as mock_get_embeddings, patch(
        "readme_ready.query.query.HNSWLib.load"
    ) as mock_hnswlib_load, patch(
        "readme_ready.query.query.make_readme_chain"
    ) as mock_make_readme_chain:

        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings

        mock_vector_store = MagicMock()
        mock_hnswlib_load.return_value = mock_vector_store

        mock_chain = MagicMock()
        mock_make_readme_chain.return_value = mock_chain

        chain = query.init_readme_chain(repo_config, user_config)

        mock_get_embeddings.assert_called_once_with(
            repo_config.llms[0].value, repo_config.device
        )
        data_path = os.path.join(repo_config.output, "docs", "data")
        mock_hnswlib_load.assert_called_once_with(data_path, mock_embeddings)
        mock_make_readme_chain.assert_called_once_with(
            repo_config.name,
            repo_config.repository_url,
            repo_config.content_type,
            repo_config.chat_prompt,
            repo_config.target_audience,
            mock_vector_store,
            user_config.llms,
            repo_config.peft_model_path,
            on_token_stream=user_config.streaming,
        )
        assert chain == mock_chain


def test_query_normal():
    query.chat_history = []
    repo_config = MagicMock()
    repo_config.name = "TestProject"
    user_config = MagicMock()

    with patch(
        "readme_ready.query.query.init_qa_chain"
    ) as mock_init_qa_chain, patch(
        "readme_ready.query.query.clear"
    ) as mock_clear, patch(
        "readme_ready.query.query.display_welcome_message"
    ) as mock_display_welcome_message, patch(
        "readme_ready.query.query.prompt"
    ) as mock_prompt, patch(
        "readme_ready.query.query.markdown"
    ) as mock_markdown:

        mock_chain = MagicMock()
        mock_init_qa_chain.return_value = mock_chain

        mock_prompt.side_effect = ["What is the purpose?", "exit"]

        mock_chain.invoke.return_value = {"answer": "This is the answer"}

        mock_markdown.return_value = "Formatted Answer"

        query.query(repo_config, user_config)

        mock_init_qa_chain.assert_called_once_with(repo_config, user_config)
        mock_clear.assert_called_once()
        mock_display_welcome_message.assert_called_once_with(repo_config.name)
        assert mock_prompt.call_count == 2
        mock_chain.invoke.assert_called_once_with(
            {
                "input": "What is the purpose?",
                "chat_history": query.chat_history,
            }
        )
        mock_markdown.assert_called_once_with("This is the answer")


def test_query_exception():
    query.chat_history = []
    repo_config = MagicMock()
    repo_config.name = "TestProject"
    user_config = MagicMock()

    with patch(
        "readme_ready.query.query.init_qa_chain"
    ) as mock_init_qa_chain, patch(
        "readme_ready.query.query.clear"
    ) as mock_clear, patch(
        "readme_ready.query.query.display_welcome_message"
    ) as mock_display_welcome_message, patch(
        "readme_ready.query.query.prompt"
    ) as mock_prompt, patch(
        "readme_ready.query.query.print"
    ) as mock_print, patch(
        "readme_ready.query.query.traceback.print_exc"
    ) as mock_print_exc:

        mock_chain = MagicMock()
        mock_init_qa_chain.return_value = mock_chain

        mock_prompt.side_effect = ["What causes error?", "exit"]

        mock_chain.invoke.side_effect = RuntimeError("Test Error")

        query.query(repo_config, user_config)

        mock_init_qa_chain.assert_called_once_with(repo_config, user_config)
        mock_clear.assert_called_once()
        mock_display_welcome_message.assert_called_once_with(repo_config.name)
        assert mock_prompt.call_count == 2
        mock_chain.invoke.assert_called_once_with(
            {
                "input": "What causes error?",
                "chat_history": query.chat_history,
            }
        )
        mock_print.assert_any_call("Thinking...")
        mock_print.assert_any_call("Something went wrong: Test Error")
        mock_print_exc.assert_called_once()


def test_generate_readme_normal():
    repo_config = MagicMock()
    repo_config.output = "output_dir"
    repo_config.name = "TestProject"
    repo_config.llms = [MagicMock()]
    repo_config.llms[0].name = "LLMName"

    user_config = MagicMock()

    readme_config = MagicMock()
    readme_config.headings = ["Introduction", "Usage"]

    with patch(
        "readme_ready.query.query.init_readme_chain"
    ) as mock_init_readme_chain, patch(
        "readme_ready.query.query.clear"
    ) as mock_clear, patch(
        "builtins.open", new_callable=MagicMock()
    ) as mock_open, patch(
        "readme_ready.query.query.markdown"
    ) as mock_markdown, patch(
        "readme_ready.query.query.print"
    ) as mock_print:

        mock_chain = MagicMock()
        mock_init_readme_chain.return_value = mock_chain

        mock_chain.invoke.return_value = {"answer": "Answer to heading"}

        mock_markdown.return_value = "Formatted Answer"

        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle

        query.generate_readme(repo_config, user_config, readme_config)

        mock_init_readme_chain.assert_called_once_with(
            repo_config, user_config
        )
        mock_clear.assert_called_once()

        data_path = os.path.join(repo_config.output, "docs", "data")
        readme_path = os.path.join(
            data_path, f"README_{repo_config.llms[0].name}.md"
        )

        calls = [
            call(readme_path, "w", encoding="utf-8"),
            call().__enter__(),
            call().__enter__().write(f"# {repo_config.name}"),
            call().__exit__(None, None, None),
            call(readme_path, "a", encoding="utf-8"),
            call().__enter__(),
            call().__enter__().write("Formatted Answer"),
            call().__enter__().write("Formatted Answer"),
            call().__exit__(None, None, None),
        ]
        mock_open.assert_has_calls(calls, any_order=False)

        assert mock_chain.invoke.call_count == len(readme_config.headings)
        mock_markdown.assert_called_with("Answer to heading")


def test_generate_readme_exception():
    repo_config = MagicMock()
    repo_config.output = "output_dir"
    repo_config.name = "TestProject"
    repo_config.llms = [MagicMock()]
    repo_config.llms[0].name = "LLMName"

    user_config = MagicMock()

    readme_config = MagicMock()
    readme_config.headings = ["Introduction", "Usage"]

    with patch(
        "readme_ready.query.query.init_readme_chain"
    ) as mock_init_readme_chain, patch(
        "readme_ready.query.query.clear"
    ) as mock_clear, patch(
        "builtins.open", new_callable=MagicMock()
    ) as mock_open, patch(
        "readme_ready.query.query.markdown"
    ) as mock_markdown, patch(
        "readme_ready.query.query.print"
    ) as mock_print, patch(
        "readme_ready.query.query.traceback.print_exc"
    ) as mock_print_exc:

        mock_chain = MagicMock()
        mock_init_readme_chain.return_value = mock_chain

        mock_chain.invoke.side_effect = [
            {"answer": "Answer to heading"},
            RuntimeError("Test Error"),
        ]

        mock_markdown.return_value = "Formatted Answer"

        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle

        query.generate_readme(repo_config, user_config, readme_config)

        mock_init_readme_chain.assert_called_once_with(
            repo_config, user_config
        )
        mock_clear.assert_called_once()

        data_path = os.path.join(repo_config.output, "docs", "data")
        readme_path = os.path.join(
            data_path, f"README_{repo_config.llms[0].name}.md"
        )

        calls = [
            call(readme_path, "w", encoding="utf-8"),
            call().__enter__(),
            call().__enter__().write(f"# {repo_config.name}"),
            call().__exit__(None, None, None),
            call(readme_path, "a", encoding="utf-8"),
            call().__enter__(),
            call().__enter__().write("Formatted Answer"),
            call().__exit__(None, None, None),
        ]
        mock_open.assert_has_calls(calls, any_order=False)

        assert mock_chain.invoke.call_count == 2
        mock_markdown.assert_called_with("Answer to heading")
        mock_print_exc.assert_called_once()

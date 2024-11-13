from unittest.mock import mock_open, patch, MagicMock
from langchain_core.documents import Document
import os

from readme_ready.index.create_vector_store import (
    create_vector_store,
    RepoLoader,
    process_directory,
    process_file,
    should_ignore,
)
from readme_ready.types import LLMModels
from langchain.text_splitter import RecursiveCharacterTextSplitter


def test_should_ignore():
    ignore_patterns = ["*.txt", "ignore_this*"]
    assert should_ignore("test.txt", ignore_patterns) is True
    assert should_ignore("ignore_this_file.py", ignore_patterns) is True
    assert should_ignore("keep_this.py", ignore_patterns) is False


def test_process_file():
    with patch("builtins.open", mock_open(read_data="file contents")):
        with patch(
            "readme_ready.index.create_vector_store.should_ignore",
            return_value=False,
        ):
            doc = process_file("test.py", [])
            assert isinstance(doc, Document)
            assert doc.page_content == "file contents"
            assert doc.metadata == {"source": "test.py"}


def test_process_file_ignore():
    with patch(
        "readme_ready.index.create_vector_store.should_ignore",
        return_value=True,
    ):
        doc = process_file("ignore.py", [])
        assert doc is None


def test_process_file_exception():
    with patch("builtins.open", side_effect=Exception("Read error")):
        with patch(
            "readme_ready.index.create_vector_store.should_ignore",
            return_value=False,
        ):
            with patch("builtins.print") as mock_print:
                doc = process_file("error.py", [])
                assert doc is None
                mock_print.assert_called_with(
                    "Error reading file error.py: Read error"
                )


def test_process_directory(tmp_path):
    # Create test files and directories
    (tmp_path / "file1.py").write_text('print("Hello World")')
    (tmp_path / "file2.py").write_text('print("Hello Again")')
    (tmp_path / "ignore.txt").write_text("Ignore this file")
    os.mkdir(tmp_path / "subdir")
    (tmp_path / "subdir" / "file3.py").write_text('print("Hello from subdir")')

    with patch(
        "readme_ready.index.create_vector_store.should_ignore",
        side_effect=lambda x, y: x.endswith(".txt"),
    ):
        docs = process_directory(str(tmp_path), ["*.txt"])
        assert len(docs) == 3  # ignore.txt should be ignored
        doc_sources = [doc.metadata["source"] for doc in docs]
        assert str(tmp_path / "file1.py") in doc_sources
        assert str(tmp_path / "file2.py") in doc_sources
        assert str(tmp_path / "subdir" / "file3.py") in doc_sources


def test_repo_loader_load():
    with patch(
        "readme_ready.index.create_vector_store.process_directory",
        return_value=["doc1", "doc2"],
    ) as mock_process_directory:
        loader = RepoLoader("path/to/repo", [])
        docs = loader.load()
        mock_process_directory.assert_called_once_with("path/to/repo", [])
        assert docs == ["doc1", "doc2"]


def test_create_vector_store(tmp_path):
    # Prepare test documents
    raw_docs = [
        Document(page_content="Content 1", metadata={"source": "file1.py"}),
        Document(page_content="Content 2", metadata={"source": "file2.py"}),
    ]

    # Mock RepoLoader
    with patch.object(RepoLoader, "load", return_value=raw_docs):
        # Mock text splitter
        with patch.object(
            RecursiveCharacterTextSplitter,
            "split_documents",
            return_value=raw_docs,
        ) as mock_split:
            # Mock HNSWLib and embeddings
            with patch(
                "readme_ready.index.create_vector_store.get_embeddings"
            ) as mock_get_embeddings:
                mock_get_embeddings.return_value = MagicMock()
                with patch(
                    "readme_ready.index.create_vector_store.HNSWLib"
                ) as mock_hnswlib:
                    mock_vector_store = MagicMock()
                    mock_hnswlib.from_documents.return_value = (
                        mock_vector_store
                    )

                    # Call the function under test
                    create_vector_store(
                        root="path/to/root",
                        output=str(tmp_path / "output"),
                        ignore=["*.txt"],
                        llms=[LLMModels.GPT3, LLMModels.GPT4],
                        device="cpu",
                    )

                    # Assertions
                    mock_split.assert_called_once_with(raw_docs)
                    mock_hnswlib.from_documents.assert_called_once()
                    mock_vector_store.save.assert_called_once_with(
                        str(tmp_path / "output")
                    )


# def test_create_vector_store_no_docs(tmp_path):
#     # Mock RepoLoader to return no documents
#     with patch.object(RepoLoader, 'load', return_value=[]):
#         with patch('builtins.print') as mock_print:
#             create_vector_store(
#                 root='path/to/root',
#                 output=str(tmp_path / 'output'),
#                 ignore=[],
#                 llms=[LLMModels.GPT3],
#                 device='cpu'
#             )
#             mock_print.assert_any_call('Splitting text into chunks for 0 docs')
#             mock_print.assert_any_call('Creating vector store....')
#             mock_print.assert_any_call('Saving vector store output....')
#             mock_print.assert_any_call('Done creating vector store....')

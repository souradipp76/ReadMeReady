import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from readme_ready.utils.HNSWLib import (
    HNSWLib,
    HNSWLibArgs,
)
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


def test_init():
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=128)
    hnswlib_instance = HNSWLib(embeddings, args)
    assert hnswlib_instance._embeddings == embeddings
    assert hnswlib_instance.args == args
    assert isinstance(hnswlib_instance.docstore, InMemoryDocstore)


@patch("readme_ready.utils.HNSWLib.hnswlib.Index")
def test_add_texts_success(mock_index_class):
    embeddings = MagicMock()
    embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    args = HNSWLibArgs(space="cosine", num_dimensions=2)
    hnswlib_instance = HNSWLib(embeddings, args)

    mock_index = MagicMock()
    mock_index.element_count = 0
    mock_index.init_index.return_value = None
    mock_index.get_max_elements.return_value = 1000
    args.index = mock_index
    hnswlib_instance._index = mock_index

    texts = ["Document 1", "Document 2"]
    ids = hnswlib_instance.add_texts(texts)

    embeddings.embed_documents.assert_called_with(texts)
    mock_index.init_index.assert_called_with(2)
    assert len(ids) == 2


def test_get_hierarchical_nsw_no_space():
    args = HNSWLibArgs(space=None, num_dimensions=128)
    with pytest.raises(ValueError) as excinfo:
        HNSWLib.get_hierarchical_nsw(args)
    assert "hnswlib requires a space argument" in str(excinfo.value)


def test_get_hierarchical_nsw_no_num_dimensions():
    args = HNSWLibArgs(space="cosine", num_dimensions=None)
    with pytest.raises(ValueError) as excinfo:
        HNSWLib.get_hierarchical_nsw(args)
    assert "hnswlib requires a num_dimensions argument" in str(excinfo.value)


def test_add_vectors_mismatched_lengths():
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=2)
    hnswlib_instance = HNSWLib(embeddings, args)
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    documents = [Document("Doc 1")]

    with pytest.raises(ValueError) as excinfo:
        hnswlib_instance.add_vectors(vectors, documents)
    assert "Vectors and documents must have the same length" in str(
        excinfo.value
    )


def test_add_vectors_empty():
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=2)
    hnswlib_instance = HNSWLib(embeddings, args)
    vectors = None
    documents = [Document("Doc 1")]

    hnswlib_instance.add_vectors(vectors, documents)
    assert hnswlib_instance._index is None


def test_add_vectors_wrong_dimension():
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=3)
    hnswlib_instance = HNSWLib(embeddings, args)
    vectors = [[0.1, 0.2]]  # Only 2 dimensions
    documents = [Document("Doc 1")]

    with pytest.raises(ValueError) as excinfo:
        hnswlib_instance.add_vectors(vectors, documents)
    assert (
        "Vectors must have the same length as the number of dimensions"
        in str(excinfo.value)
    )


def test_add_vectors_resize_index():
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=2)
    hnswlib_instance = HNSWLib(embeddings, args)
    hnswlib_instance._index = MagicMock()
    hnswlib_instance._index.element_count = 100
    hnswlib_instance._index.get_max_elements.return_value = 100
    hnswlib_instance._index.resize_index.return_value = None

    vectors = [[0.1, 0.2], [0.3, 0.4]]
    documents = [Document("Doc 1"), Document("Doc 2")]

    hnswlib_instance.add_vectors(vectors, documents)

    hnswlib_instance._index.resize_index.assert_called_with(
        102
    )  # 100 existing + 2 new


def test_add_documents():
    embeddings = MagicMock()
    embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    args = HNSWLibArgs(space="cosine", num_dimensions=2)
    hnswlib_instance = HNSWLib(embeddings, args)
    hnswlib_instance.add_vectors = MagicMock()
    documents = [Document("Doc 1"), Document("Doc 2")]

    hnswlib_instance.add_documents(documents)

    embeddings.embed_documents.assert_called_with(["Doc 1", "Doc 2"])
    hnswlib_instance.add_vectors.assert_called()


def test_from_texts():
    embeddings = MagicMock()
    embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    with patch(
        "readme_ready.utils.HNSWLib.HNSWLib.from_documents"
    ) as mock_from_documents:
        HNSWLib.from_texts(["Text 1", "Text 2"], embeddings)
        mock_from_documents.assert_called()


def test_from_documents_with_docstore():
    embeddings = MagicMock()
    documents = [Document("Doc 1"), Document("Doc 2")]
    docstore = MagicMock()
    with patch(
        "readme_ready.utils.HNSWLib.HNSWLib.add_documents"
    ) as mock_add_documents:
        hnsw = HNSWLib.from_documents(documents, embeddings, docstore=docstore)
        mock_add_documents.assert_called_once_with(documents)


def test_from_documents_without_docstore():
    embeddings = MagicMock()
    documents = [Document("Doc 1"), Document("Doc 2")]
    with pytest.raises(KeyError):
        HNSWLib.from_documents(documents, embeddings)


def test_similarity_search_by_vector_wrong_dimension():
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=3)
    hnswlib_instance = HNSWLib(embeddings, args)
    hnswlib_instance._index = MagicMock()
    embedding = [0.1, 0.2]  # Only 2 dimensions

    with pytest.raises(ValueError) as excinfo:
        hnswlib_instance.similarity_search_by_vector(embedding)
    assert (
        f"Query vector must have the same length as the number of dimensions ({args.num_dimensions})"
        in str(excinfo.value)
    )


def test_similarity_search_by_vector_k_greater_than_total(capsys):
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=2)
    hnswlib_instance = HNSWLib(embeddings, args)
    hnswlib_instance._index = MagicMock()
    hnswlib_instance._index.element_count = 1  # Total elements is 1
    hnswlib_instance._index.knn_query.return_value = (
        np.array([[0]]),
        np.array([[0.0]]),
    )
    hnswlib_instance.docstore._dict = {"0": Document("Doc 0")}

    embedding = [0.1, 0.2]
    hnswlib_instance.similarity_search_by_vector(embedding, k=5)
    captured = capsys.readouterr()
    assert (
        "k (5) is greater than the number of elements in the index (1), setting k to 1"
        in captured.out
    )


def test_similarity_search():
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2]
    args = HNSWLibArgs(space="cosine", num_dimensions=2)
    hnswlib_instance = HNSWLib(embeddings, args)
    hnswlib_instance.similarity_search_by_vector = MagicMock(
        return_value=[(Document("Doc 1"), 0.0)]
    )

    results = hnswlib_instance.similarity_search("query", k=2)
    embeddings.embed_query.assert_called_with("query")
    hnswlib_instance.similarity_search_by_vector.assert_called_with(
        [0.1, 0.2], 2
    )
    assert len(results) == 1
    assert results[0].page_content == "Doc 1"


def test_save():
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=2)
    hnswlib_instance = HNSWLib(embeddings, args)
    hnswlib_instance._index = MagicMock()
    hnswlib_instance._index.save_index.return_value = None
    hnswlib_instance.docstore = InMemoryDocstore()
    hnswlib_instance.docstore._dict = {
        "1": Document("Doc 1"),
        "2": Document("Doc 2"),
    }

    with patch("builtins.open", new_callable=MagicMock()) as mock_open:
        with patch("os.path.exists") as mock_exists:
            with patch("os.makedirs") as mock_makedirs:
                mock_exists.return_value = False
                hnswlib_instance.save("test_directory")
                mock_makedirs.assert_called_with("test_directory")
                hnswlib_instance._index.save_index.assert_called_with(
                    os.path.join("test_directory", "hnswlib.index")
                )
                assert (
                    mock_open.call_count == 2
                )  # For docstore.json and args.json


def test_load():
    embeddings = MagicMock()
    with patch("builtins.open", new_callable=MagicMock()) as mock_open:
        with patch("json.load") as mock_json_load:
            mock_json_load.side_effect = [
                {"space": "cosine", "num_dimensions": 2},  # For args.json
                [
                    ["1", {"page_content": "Doc 1", "metadata": {}}],
                    ["2", {"page_content": "Doc 2", "metadata": {}}],
                ],  # For docstore.json
            ]
            with patch(
                "readme_ready.utils.HNSWLib.hnswlib.Index"
            ) as mock_index_class:
                mock_index = MagicMock()
                mock_index.load_index.return_value = None
                mock_index_class.return_value = mock_index
                hnswlib_instance = HNSWLib.load("test_directory", embeddings)
                assert hnswlib_instance.args.space == "cosine"
                assert hnswlib_instance.args.num_dimensions == 2
                assert hnswlib_instance._index == mock_index
                assert "1" in hnswlib_instance.docstore._dict
                assert (
                    hnswlib_instance.docstore._dict["1"].page_content
                    == "Doc 1"
                )


def test_init_index_no_index():
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=None)
    hnswlib_instance = HNSWLib(embeddings, args)
    vectors = [[0.1, 0.2, 0.3]]
    with patch(
        "readme_ready.utils.HNSWLib.HNSWLib.get_hierarchical_nsw"
    ) as mock_get_hnsw:
        mock_get_hnsw.return_value = MagicMock()
        hnswlib_instance.init_index(vectors)
        assert hnswlib_instance.args.num_dimensions == 3
        mock_get_hnsw.assert_called_with(hnswlib_instance.args)


def test_init_index_with_index():
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=3)
    mock_index = MagicMock()
    mock_index.element_count = None
    args.index = mock_index
    hnswlib_instance = HNSWLib(embeddings, args)
    hnswlib_instance._index = mock_index
    vectors = [[0.1, 0.2, 0.3]]
    hnswlib_instance.init_index(vectors)
    mock_index.init_index.assert_called_with(1)


def test_save_directory_exists():
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=2)
    hnswlib_instance = HNSWLib(embeddings, args)
    hnswlib_instance._index = MagicMock()
    hnswlib_instance.docstore = MagicMock()
    hnswlib_instance.docstore._dict = {}

    with patch("os.path.exists") as mock_exists:
        with patch("os.makedirs") as mock_makedirs:
            mock_exists.return_value = True
            hnswlib_instance.save(".")
            mock_makedirs.assert_not_called()


def test_runtime_error_caught(capsys):
    embeddings = MagicMock()
    args = HNSWLibArgs(space="cosine", num_dimensions=2)
    hnswlib_instance = HNSWLib(embeddings, args)
    hnswlib_instance._index = None

    with pytest.raises(AttributeError):
        hnswlib_instance._index.save_index("some_path")
    # Since there's no exception handling in the save method, we can't catch a RuntimeError here.
    # This test is just to show that if an error occurs, it will propagate.

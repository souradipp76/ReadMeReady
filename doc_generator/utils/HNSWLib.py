"""
HNSWLib Wrapper
"""

import json
import os
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import hnswlib
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class SaveableVectorStore(VectorStore):
    """Saveable Vector Store"""

    @abstractmethod
    def save(self, directory):
        """Save"""


class HNSWLibBase:
    """HNSWLibBase"""

    def __init__(self, space: str, num_dimensions: Optional[int] = None):
        self.space = space
        self.num_dimensions = num_dimensions


class HNSWLibArgs(HNSWLibBase):
    """HNSWLibArgs"""

    def __init__(
        self,
        space: str,
        num_dimensions: Optional[int] = None,
        docstore: Optional[InMemoryDocstore] = None,
        index: Optional[hnswlib.Index] = None,
    ):
        super().__init__(space, num_dimensions)
        self.docstore = docstore
        self.index = index


class HNSWLib(SaveableVectorStore):
    def __init__(self, embeddings: Embeddings, args: HNSWLibArgs):
        super().__init__()
        self.args = args
        self._embeddings = embeddings
        self._index = args.index
        self.docstore = args.docstore if args.docstore else InMemoryDocstore()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        vectors = self._embeddings.embed_documents(list(texts))
        documents = [Document(page_content=text) for text in texts]
        ids = self.add_vectors(vectors, documents)
        return ids

    @staticmethod
    def get_hierarchical_nsw(args: HNSWLibArgs) -> hnswlib.Index:
        """Get Hierarchical NSW"""
        if args.space is None:
            raise ValueError("hnswlib requires a space argument")
        if args.num_dimensions is None:
            raise ValueError("hnswlib requires a num_dimensions argument")
        return hnswlib.Index(args.space, args.num_dimensions)

    def init_index(self, vectors: List[List[float]]):
        """Init Index"""
        if not self._index:
            if self.args.num_dimensions is None:
                self.args.num_dimensions = len(vectors[0])
            self._index = HNSWLib.get_hierarchical_nsw(self.args)
        if not self._index.element_count:
            self._index.init_index(len(vectors))

    def add_vectors(
        self, vectors: List[List[float]], documents: List[Document]
    ):
        """Add Vectors"""
        if not vectors:
            return
        self.init_index(vectors)
        if len(vectors) != len(documents):
            raise ValueError("Vectors and documents must have the same length")
        if len(vectors[0]) != self.args.num_dimensions:
            raise ValueError(
                "Vectors must have the same length as the "
                + f"number of dimensions ({self.args.num_dimensions})"
            )
        assert self._index is not None
        capacity = self._index.get_max_elements()
        needed = self._index.element_count + len(vectors)
        if needed > capacity:
            self._index.resize_index(needed)

        docstore_size = len(self.docstore._dict)
        ids = []
        for i, vector in enumerate(vectors):
            self._index.add_items(
                np.array(vector), np.array([docstore_size + i])
            )
            self.docstore.add({str(docstore_size + i): documents[i]})
            ids.append(str(docstore_size + i))
        return ids

    def add_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        texts = [doc.page_content for doc in documents]
        embeds = self._embeddings.embed_documents(texts)
        self.add_vectors(embeds, documents)
        return []

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "HNSWLib":
        documents = [Document(text) for text in texts]
        return cls.from_documents(documents, embedding, **kwargs)

    @staticmethod
    def from_documents(
        documents: List[Document], embedding: Embeddings, **kwargs: Any
    ) -> "HNSWLib":
        args = HNSWLibArgs(space="cosine", docstore=kwargs["docstore"])
        hnsw = HNSWLib(embedding, args)
        hnsw.add_documents(documents)
        return hnsw

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List:
        if len(embedding) != self.args.num_dimensions:
            raise ValueError(
                "Query vector must have the same length as the "
                + f"number of dimensions ({self.args.num_dimensions})"
            )
        assert self._index is not None
        total = self._index.element_count
        if k > total:
            print(
                f"k ({k}) is greater than the number of elements in the "
                + f"index ({total}), setting k to {total}"
            )
            k = total
        labels, distances = self._index.knn_query(embedding, k)
        return [
            (self.docstore._dict[str(label)], distance)
            for label, distance in zip(labels[0], distances[0])
        ]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        query_embed = self._embeddings.embed_query(query)
        results = self.similarity_search_by_vector(query_embed, k, **kwargs)
        return [result[0] for result in results]

    def save(self, directory: str):
        print(f"Saving in directory {directory}")
        if not os.path.exists(directory):
            os.makedirs(directory)
        assert self._index is not None
        self._index.save_index(os.path.join(directory, "hnswlib.index"))
        with open(os.path.join(directory, "docstore.json"), "w") as f:
            docstore_data = []
            for key, val in self.docstore._dict.items():
                docstore_data.append([key, val.model_dump()])
            json.dump(docstore_data, f)
        with open(os.path.join(directory, "args.json"), "w") as f:
            json.dump(
                {
                    "space": self.args.space,
                    "num_dimensions": self.args.num_dimensions,
                },
                f,
            )

    @staticmethod
    def load(directory: str, embeddings: Embeddings):
        with open(os.path.join(directory, "args.json"), "r") as f:
            args_data = json.load(f)
        args = HNSWLibArgs(
            space=args_data["space"],
            num_dimensions=args_data["num_dimensions"],
        )
        index = hnswlib.Index(space=args.space, dim=args.num_dimensions)
        index.load_index(os.path.join(directory, "hnswlib.index"))
        args.docstore = InMemoryDocstore()
        with open(os.path.join(directory, "docstore.json"), "r") as f:
            doc_data = json.load(f)
        for id, value in doc_data:
            args.docstore.add(
                {
                    str(id): Document(
                        page_content=value["page_content"],
                        metadata=value["metadata"],
                    )
                }
            )

        args.index = index
        return HNSWLib(embeddings, args)

"""
Here are the changes made to fix the linter errors:
Added type hints for function parameters and return values where necessary.

Fixed the signature of add_texts to match the superclass VectorStore.

Handled the case where self._index is None in the __init__ method.

Replaced Optional[Any] with the actual type where possible to avoid the union-attr error.

Fixed the signatures of from_texts, from_documents, similarity_search_by_vector, and similarity_search to match the superclass VectorStore.

Added **kwargs to the add_documents method to match the superclass signature.
"""

import os
import json
import hnswlib
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Union

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

class SaveableVectorStore(VectorStore):

    def save(self, directory: str):
        pass

class HNSWLibBase:
    def __init__(self, space: str, num_dimensions: Optional[int] = None):
        self.space = space
        self.num_dimensions = num_dimensions

class HNSWLibArgs(HNSWLibBase):
    def __init__(self, space: str, num_dimensions: Optional[int] = None, docstore: Optional[InMemoryDocstore] = None, index: Optional[hnswlib.Index] = None):
        super().__init__(space, num_dimensions)
        self.docstore = docstore
        self.index = index

class HNSWLib(SaveableVectorStore):
    def __init__(self, embeddings: Embeddings, args: HNSWLibArgs):
        super().__init__()
        self.args = args
        self._embeddings = embeddings
        self._index = args.index or hnswlib.Index(args.space, args.num_dimensions)
        self.docstore = args.docstore if args.docstore else InMemoryDocstore()

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> List[str]:
        vectors = self._embeddings.embed_documents(texts)
        documents = [Document(page_content=text) for text in texts]
        self.add_vectors(vectors, documents)
        return []

    @staticmethod
    def get_hierarchical_nsw(args: HNSWLibArgs) -> hnswlib.Index:
        if args.space is None:
            raise ValueError('hnswlib requires a space argument')
        if args.num_dimensions is None:
            raise ValueError('hnswlib requires a num_dimensions argument')
        return hnswlib.Index(args.space, args.num_dimensions)

    def init_index(self, vectors: List[List[float]]):
        if not self._index.element_count:
            if self.args.num_dimensions is None:
                self.args.num_dimensions = len(vectors[0])
            self._index.init_index(len(vectors))

    def add_vectors(self, vectors: List[List[float]], documents: List[Document]):
        if not vectors:
            return
        self.init_index(vectors)
        if len(vectors) != len(documents):
            raise ValueError("Vectors and documents must have the same length")
        if len(vectors[0]) != self.args.num_dimensions:
            raise ValueError(f"Vectors must have the same length as the number of dimensions ({self.args.num_dimensions})")
        capacity = self._index.get_max_elements()
        needed = self._index.element_count + len(vectors)
        if needed > capacity:
            self._index.resize_index(needed)

        docstore_size = len(self.docstore._dict)
        for i, vector in enumerate(vectors):
            self._index.add_items(np.array(vector), np.array([docstore_size + i]))
            self.docstore.add({str(docstore_size + i): documents[i]})

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        texts = [doc.page_content for doc in documents]
        embeds = self._embeddings.embed_documents(texts)
        self.add_vectors(embeds, documents)
        return []

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> "HNSWLib":
        documents = [Document(text) for text in texts]
        return cls.from_documents(documents, embedding, **kwargs)

    @classmethod
    def from_documents(cls, documents: List[Document], embedding: Embeddings, **kwargs: Any) -> "HNSWLib":
        args = HNSWLibArgs(space='cosine')
        hnsw = cls(embedding, args)
        hnsw.add_documents(documents)
        return hnsw

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs: Any) -> List[Union[Document, float]]:
        if len(embedding) != self.args.num_dimensions:
            raise ValueError(f"Query vector must have the same length as the number of dimensions ({self.args.num_dimensions})")
        total = self._index.element_count
        if k > total:
            print(f"k ({k}) is greater than the number of elements in the index ({total}), setting k to {total}")
            k = total
        labels, distances = self._index.knn_query(embedding, k)
        return [(self.docstore._dict[str(label)], distance) for label, distance in zip(labels[0], distances[0])]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        results = self.similarity_search_by_vector(self._embeddings.embed_query(query), k, **kwargs)
        return [result[0] for result in results]

    def save(self, directory: str):
        print(f"Saving in directory {directory}")
        if not os.path.exists(directory):
            os.makedirs(directory)
        self._index.save_index(os.path.join(directory, 'hnswlib.index'))
        with open(os.path.join(directory, 'docstore.json'), 'w') as f:
            docstore_data = []
            for key, val in self.docstore._dict.items():
                docstore_data.append([key, val.dict()])
            json.dump(docstore_data, f)
        with open(os.path.join(directory, 'args.json'), 'w') as f:
            json.dump({'space': self.args.space, 'num_dimensions': self.args.num_dimensions}, f)

    @staticmethod
    def load(directory: str, embeddings: Embeddings):
        with open(os.path.join(directory, 'args.json'), 'r') as f:
            args_data = json.load(f)
        args = HNSWLibArgs(space=args_data['space'], num_dimensions=args_data['num_dimensions'])
        index = hnswlib.Index(space=args.space, dim=args.num_dimensions)
        index.load_index(os.path.join(directory, 'hnswlib.index'))
        args.docstore = InMemoryDocstore()
        with open(os.path.join(directory, 'docstore.json'), 'r') as f:
            doc_data = json.load(f)
        for id, value in doc_data:
            args.docstore.add({str(id): Document(
                page_content=value['page_content'],
                metadata=value['metadata']
            )})

        args.index = index
        return HNSWLib(embeddings, args)

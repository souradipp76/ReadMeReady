import os
import json
import hnswlib

from abc import abstractmethod
from typing import List, Optional
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document


class SaveableVectorStore(VectorStore):

    @abstractmethod
    def save(self, directory):
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
    def __init__(self, embeddings: Embeddings, args: 'HNSWLibArgs'):
        super().__init__()
        self.args = args
        self._embeddings = embeddings
        self._index = args.index
        self.docstore = args.docstore if args.docstore else InMemoryDocstore()

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        vectors = self._embeddings.embed_documents(texts)
        documents = [Document(page_content=text, metadata=None) for text in texts]
        self.add_vectors(vectors, documents)

    @staticmethod
    def get_hierarchical_nsw(args: 'HNSWLibArgs'):
        if args.space is None:
            raise ValueError('hnswlib requires a space argument')
        if args.num_dimensions is None:
            raise ValueError('hnswlib requires a num_dimensions argument')
        return hnswlib.Index(space=args.space, dim=args.num_dimensions)

    def init_index(self, vectors: List[List[float]]):
        if not self._index:
            if self.args.num_dimensions is None:
                self.args.num_dimensions = len(vectors[0])
            self.index = HNSWLib.get_hierarchical_nsw(self.args)
        if not self.index.get_current_count():
            self.index.init_index(len(vectors))

    @property
    def index(self) -> hnswlib.Index:
        if not self._index:
            raise Exception('Vector store not initialized yet. Try calling `add_documents` first.')
        return self._index

    @index.setter
    def index(self, value: hnswlib.Index):
        self._index = value

    def add_vectors(self, vectors: List[List[float]], documents: List[Document]):
        if not vectors:
            return
        self.init_index(vectors)
        if len(vectors) != len(documents):
            raise ValueError("Vectors and documents must have the same length")
        if len(vectors[0]) != self.args.num_dimensions:
            raise ValueError(f"Vectors must have the same length as the number of dimensions ({self.args.num_dimensions})")
        capacity = self.index.get_max_elements()
        needed = self.index.get_current_count() + len(vectors)
        if needed > capacity:
            self.index.resize_index(needed)
        for i, vector in enumerate(vectors):
            self.index.add_items([vector], [self.docstore.count + i])
            self.docstore.add(self.docstore.count + i, documents[i])
            
    @staticmethod
    def from_texts(texts: List[str], embeddings: Embeddings, docstore: Optional[InMemoryDocstore] = None):
        documents = [Document(text) for text in texts]
        return HNSWLib.from_documents(documents, embeddings, docstore)

    @staticmethod
    def from_documents(documents: List[Document], embeddings: Embeddings, docstore: Optional[InMemoryDocstore] = None):
        args = HNSWLibArgs(space='cosine', docstore=docstore)
        hnsw = HNSWLib(embeddings, args)
        hnsw.add_documents(documents)
        return hnsw
    
    def similarity_search_by_vector(self, query: List[float], k: int) -> List:
        if len(query) != self.args.num_dimensions:
            raise ValueError(f"Query vector must have the same length as the number of dimensions ({self.args.num_dimensions})")
        total = self.index.get_current_count()
        if k > total:
            print(f"k ({k}) is greater than the number of elements in the index ({total}), setting k to {total}")
            k = total
        labels, distances = self.index.knn_query(query, k)
        return [(self.docstore.search(str(label)), distance) for label, distance in zip(labels, distances)]

    def similarity_search(self, query: str, k: int) -> List[Document]:
        return self.similarity_search_by_vector(self._embeddings.embed_query(query), k)
    
    def save(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.index.save_index(os.path.join(directory, 'hnswlib.index'))
        with open(os.path.join(directory, 'docstore.json'), 'w') as f:
            json.dump(self.docstore._docs, f)
        with open(os.path.join(directory, 'args.json'), 'w') as f:
            json.dump({'space': self.args.space, 'num_dimensions': self.args.num_dimensions}, f)

    @staticmethod
    def load(directory: str, embeddings: Embeddings):
        with open(os.path.join(directory, 'args.json'), 'r') as f:
            args_data = json.load(f)
        args = HNSWLibArgs(space=args_data['space'], num_dimensions=args_data['num_dimensions'])
        index = hnswlib.Index(space=args.space, dim=args.num_dimensions)
        index.load_index(os.path.join(directory, 'hnswlib.index'))
        with open(os.path.join(directory, 'docstore.json'), 'r') as f:
            doc_data = json.load(f)
        args.docstore = InMemoryDocstore()
        doc_dict = {}
        for doc in doc_data:
            key, value = doc
            doc_dict[key] = value
        args.docstore.add(doc_dict)
        args.index = index
        return HNSWLib(embeddings, args)

import json
import os
from pathlib import Path
from abc import abstractmethod
import hnswlib
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStore


class SaveableVectorStore(VectorStore):

  @abstractmethod
  def save(self, directory):
    pass


class HNSW(SaveableVectorStore):

  def __init__(self,
               embeddings,
               space,
               num_dimensions=None,
               docstore=None,
               index=None):
    super().__init__(embeddings)
    self.space = space
    self.num_dimensions = num_dimensions or embeddings.get_dimensions()
    self.docstore = docstore or InMemoryDocstore()
    self.index = index or self.init_index()

  def init_index(self):
    index = hnswlib.Index(space=self.space, dim=self.num_dimensions)
    index.init_index(max_elements=10000, ef_construction=200, M=16)
    return index

  def add_documents(self, documents):
    texts = [doc.page_content for doc in documents]
    vectors = self.embeddings.embed_documents(texts)
    self.add_vectors(vectors, documents)

  def add_vectors(self, vectors, documents):
    if not self.index:
      self.init_index()
    if len(vectors) != len(documents):
      raise ValueError("Vectors and documents must have the same length")
    self.index.add_items(vectors, ids=[i for i in range(len(documents))])
    for i, doc in enumerate(documents):
      self.docstore.add(doc, i)

  def similarity_search_vector_with_score(self, query, k):
    labels, distances = self.index.knn_query(query, k=k)
    return [(self.docstore.get(label[0]), distance[0])
            for label, distance in zip(labels, distances)]

  def save(self, directory):
    os.makedirs(directory, exist_ok=True)
    self.index.save_index(str(Path(directory) / "hnswlib.index"))
    with open(Path(directory) / "args.json", 'w') as f:
      json.dump({
          'space': self.space,
          'num_dimensions': self.num_dimensions
      }, f)
    with open(Path(directory) / "docstore.json", 'w') as f:
      json.dump(self.docstore.dump(), f)

  @staticmethod
  def load(directory, embeddings):
    with open(Path(directory) / "args.json", 'r') as f:
      args = json.load(f)
    with open(Path(directory) / "docstore.json", 'r') as f:
      documents = json.load(f)
    docstore = InMemoryDocstore.load(documents)
    index = hnswlib.Index(space=args['space'], dim=args['num_dimensions'])
    index.load_index(str(Path(directory) / "hnswlib.index"),
                     max_elements=10000)
    return HNSW(embeddings, args['space'], args['num_dimensions'], docstore,
                index)


# Example usage
if __name__ == "__main__":
  embeddings = Embeddings()  # Placeholder for actual embeddings implementation
  hnsw = HNSW(embeddings, 'cosine')
  # Assume Document class and documents initialization here
  documents = []  # This should be a list of Document instances
  hnsw.add_documents(documents)
  query_vector = [0.1] * hnsw.num_dimensions  # Example query vector
  print(hnsw.similarity_search_vector_with_score(query_vector, 5))

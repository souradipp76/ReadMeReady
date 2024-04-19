import os
from pathlib import Path
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from doc_generator.utils.HNSWLib import HNSWLib

def processFile(filePath: str) -> Document:
    def read_file(path):
        with open(path, 'r', encoding='utf8') as file:
            return file.read()

    try:
        fileContents = read_file(filePath)
        metadata = {'source': filePath}
        doc = Document(page_content=fileContents, metadata=metadata)
        return doc
    except Exception as e:
        print(f"Error reading file {filePath}: {str(e)}")
        return None

def processDirectory(directoryPath: str) -> List[Document]:
    docs = []
    try:
        files = os.listdir(directoryPath)
    except Exception as e:
        print(e)
        raise Exception(f"Could not read directory: {directoryPath}. Did you run `sh download.sh`?")
    
    for file in files:
        filePath = Path(directoryPath) / file
        if filePath.is_dir():
            nestedDocs = processDirectory(str(filePath))
            docs.extend(nestedDocs)
        else:
            doc = processFile(str(filePath))
            docs.append(doc)
    
    return docs

class RepoLoader(BaseLoader):
    def __init__(self, filePath: str):
        super().__init__()
        self.filePath = filePath
    
    def load(self) -> List[Document]:
        return processDirectory(self.filePath)

def createVectorStore(root: str, output: str) -> None:
    loader = RepoLoader(root)
    rawDocs = loader.load()
    rawDocs = [doc for doc in rawDocs if doc is not None]
    # Split the text into chunks
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)
    docs = textSplitter.split_documents(rawDocs)
    # Create the vectorstore
    vectorStore = HNSWLib.from_documents(docs, OpenAIEmbeddings())
    vectorStore.save(output)

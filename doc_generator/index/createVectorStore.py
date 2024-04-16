import asyncio
import os
from pathlib import Path

# Placeholder imports for custom Python classes and functions
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document import Document
from langchain.document_loaders import BaseDocumentLoader
from langchain.hnswlib import HNSWLib

async def processFile(filePath: str) -> Document:
    async def read_file(path):
        with open(path, 'r', encoding='utf8') as file:
            return file.read()
    
    try:
        fileContents = await read_file(filePath)
        metadata = {'source': filePath}
        doc = Document(pageContent=fileContents, metadata=metadata)
        return doc
    except Exception as e:
        raise Exception(f"Error reading file {filePath}: {str(e)}")

async def processDirectory(directoryPath: str) -> [Document]:
    docs = []
    try:
        files = os.listdir(directoryPath)
    except Exception as e:
        print(e)
        raise Exception(f"Could not read directory: {directoryPath}. Did you run `sh download.sh`?")
    
    for file in files:
        filePath = Path(directoryPath) / file
        if filePath.is_dir():
            nestedDocs = await processDirectory(str(filePath))
            docs.extend(nestedDocs)
        else:
            doc = await processFile(str(filePath))
            docs.append(doc)
    
    return docs

class RepoLoader(BaseDocumentLoader):
    def __init__(self, filePath: str):
        super().__init__()
        self.filePath = filePath
    
    async def load(self) -> [Document]:
        return await processDirectory(self.filePath)

async def createVectorStore(root: str, output: str) -> None:
    loader = RepoLoader(root)
    rawDocs = await loader.load()
    # Split the text into chunks
    textSplitter = RecursiveCharacterTextSplitter(chunkSize=8000, chunkOverlap=100)
    docs = await textSplitter.splitDocuments(rawDocs)
    # Create the vectorstore
    vectorStore = await HNSWLib.fromDocuments(docs, OpenAIEmbeddings())
    await vectorStore.save(output)

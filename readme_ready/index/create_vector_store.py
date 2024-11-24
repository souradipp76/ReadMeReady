"""
Utilities to Create Vector Store
"""

import fnmatch
import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from readme_ready.utils.HNSWLib import HNSWLib, InMemoryDocstore
from readme_ready.utils.llm_utils import LLMModels, get_embeddings


def should_ignore(file_name: str, ignore: List[str]):
    return any(fnmatch.fnmatch(file_name, pattern) for pattern in ignore)


def process_file(file_path: str, ignore: List[str]) -> Document | None:
    """
    Processes a file

    Processes a specified file and converts the content
    of the file into Document format. Ignores any file matching
    the patterns provided in the ignore list.

    Args:
        file_path: The file to be processed.
        ignore: A list of file patterns to ignore during processing.

    Returns:
        doc: A Document with file contents and metatdata

    """

    def read_file(path):
        with open(path, "r", encoding="utf8") as file:
            return file.read()

    if should_ignore(file_path, ignore):
        return None

    try:
        file_contents = read_file(file_path)
        metadata = {"source": file_path}
        doc = Document(page_content=file_contents, metadata=metadata)
        return doc
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None


def process_directory(
    directory_path: str, ignore: List[str]
) -> List[Document]:
    """
    Processes a directory

    Processes a specified directory, and converts all the content
    of the files in the directory into Document format. Ignores
    files matching the patterns provided in the ignore list.

    Args:
        directory_path: The root directory containing the files to be
            processed.
        ignore: A list of file patterns to ignore during processing.

    Returns:
        docs: List of Documents with file contents and metatdata

    """
    docs = []
    try:
        files = os.listdir(directory_path)
    except Exception as e:
        print(e)
        raise FileNotFoundError(
            f"Could not read directory: {directory_path}. \
                                Did you run `sh download.sh`?"
        ) from e

    for file in files:
        if should_ignore(file, ignore):
            continue
        file_path = Path(directory_path) / file
        if file_path.is_dir():
            nested_docs = process_directory(str(file_path), ignore)
            docs.extend(nested_docs)
        else:
            doc = process_file(str(file_path), ignore)
            docs.append(doc)

    return docs


class RepoLoader(BaseLoader):
    """
    Class to load and process a repository

    A loader class which loads and processes a repsitory given
    the root directory path and list of file patterns to ignore.

    Typical usage example:

        loader = RepoLoader(path, ignore)
        docs = loader.load()

    """

    def __init__(self, file_path: str, ignore: List[str]):
        super().__init__()
        self.file_path = file_path
        self.ignore = ignore

    def load(self) -> List[Document]:
        return process_directory(self.file_path, self.ignore)


def create_vector_store(
    root: str,
    output: str,
    ignore: List[str],
    llms: List[LLMModels],
    device: str,
) -> None:
    """
    Creates a vector store from Markdown documents

    Loads documents from the specified root directory, splits the text into
    chunks, creates a vector store using the selected LLM, and saves the
    vector store to the output path. Ignores files matching the patterns
    provided in the ignore list.

    Args:
        root: The root directory containing the documents to be processed.
        output: The directory where the vector store will be saved.
        ignore: A list of file patterns to ignore during document loading.
        llms: A list of LLMModels to use for generating embeddings.
        device: The device to use for embedding generation
            (e.g., 'cpu' or 'auto').

    """

    llm = llms[1] if len(llms) > 1 else llms[0]
    loader = RepoLoader(root, ignore)
    raw_docs = loader.load()
    raw_docs = [doc for doc in raw_docs if doc is not None]
    # Split the text into chunks
    print(f"Splitting text into chunks for {len(raw_docs)} docs")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    docs = text_splitter.split_documents(raw_docs)
    # Create the vectorstore
    print("Creating vector store....")
    vector_store = HNSWLib.from_documents(
        docs, get_embeddings(llm.name, device), docstore=InMemoryDocstore()
    )

    print("Saving vector store output....")
    vector_store.save(output)

    print("Done creating vector store....")

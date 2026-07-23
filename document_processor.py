"""
document_processor.py
----------------------
Handles everything related to turning raw uploaded PDF bytes into clean,
chunked LangChain `Document` objects ready for embedding.

Responsibilities:
1. Persist uploaded files (Streamlit gives us in-memory bytes) to a temp
   directory so PyPDFLoader can read them from disk.
2. Extract text page-by-page, preserving `source` (filename) and `page`
   metadata on every chunk -- this is what allows us to cite sources later.
3. Split extracted text into overlapping chunks for better retrieval recall.
"""

import os
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


def _save_uploaded_file(uploaded_file) -> str:
    """
    Persist a Streamlit UploadedFile object to disk so PyPDFLoader (which
    expects a file path) can read it.

    Args:
        uploaded_file: A Streamlit `UploadedFile` from st.file_uploader.

    Returns:
        The absolute path to the saved temp file.
    """
    os.makedirs(config.UPLOAD_TEMP_DIR, exist_ok=True)
    file_path = os.path.join(config.UPLOAD_TEMP_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def load_and_split_pdf(uploaded_file) -> List[Document]:
    """
    Load a single uploaded PDF and split it into overlapping text chunks.

    Each resulting chunk retains metadata:
        - "source": original filename (e.g. "annual_report.pdf")
        - "page":   the 1-indexed page number the chunk came from

    This metadata is what powers the "Source: filename.pdf, Page: 4"
    citations shown alongside every answer in the UI.

    Args:
        uploaded_file: A Streamlit `UploadedFile` object.

    Returns:
        List of chunked `Document` objects.
    """
    file_path = _save_uploaded_file(uploaded_file)

    try:
        # PyPDFLoader creates one Document per PDF page, with
        # metadata={"source": file_path, "page": <int, 0-indexed>}
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(pages)

        # Normalize metadata: use the clean original filename (not the temp
        # path) and convert to 1-indexed page numbers for human readability.
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name
            chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1

        logger.info(
            "Processed '%s': %d pages -> %d chunks",
            uploaded_file.name, len(pages), len(chunks),
        )
        return chunks

    finally:
        # Clean up the temp file regardless of success/failure.
        if os.path.exists(file_path):
            os.remove(file_path)


def process_uploaded_files(uploaded_files) -> List[Document]:
    """
    Process multiple uploaded PDFs into a single flat list of chunks.

    Args:
        uploaded_files: List of Streamlit `UploadedFile` objects.

    Returns:
        Combined list of chunked `Document` objects across all files.
    """
    all_chunks: List[Document] = []
    for uploaded_file in uploaded_files:
        try:
            chunks = load_and_split_pdf(uploaded_file)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error("Failed to process %s: %s", uploaded_file.name, e)
            raise RuntimeError(
                f"Failed to process '{uploaded_file.name}': {e}"
            ) from e

    return all_chunks

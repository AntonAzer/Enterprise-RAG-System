"""
rag_pipeline.py
---------------
Core RAG logic: embeddings, vector store management, and the retrieval +
generation chain itself, built with modern LangChain Expression Language
(LCEL) rather than the older, more opaque `RetrievalQA` class.

Why LCEL instead of RetrievalQA:
- Full control over the prompt and how retrieved docs are formatted.
- Easy to expose BOTH the generated answer AND the raw source documents
  (RetrievalQA's return_source_documents works, but LCEL makes the data
  flow explicit and easy to extend, e.g. adding re-ranking later).
"""

import logging
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings(openai_api_key: str = None, use_local_fallback: bool = False):
    """
    Build an embeddings object. Defaults to OpenAI embeddings; falls back to
    a local HuggingFace sentence-transformer model when no API key is
    available (e.g. for offline demos or to avoid API costs entirely).

    Args:
        openai_api_key: OpenAI API key. If None/empty, forces local fallback.
        use_local_fallback: Explicitly force the local HF model even if a
            key is present (useful for cost-free portfolio demos).

    Returns:
        A LangChain-compatible embeddings object.
    """
    if use_local_fallback or not openai_api_key:
        logger.info("Using local HuggingFace embeddings (%s)",
                    config.HUGGINGFACE_EMBEDDING_MODEL)
        # Imported lazily so the app doesn't require sentence-transformers
        # / torch installed unless this fallback path is actually used.
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_EMBEDDING_MODEL)

    logger.info("Using OpenAI embeddings (%s)", config.OPENAI_EMBEDDING_MODEL)
    return OpenAIEmbeddings(
        model=config.OPENAI_EMBEDDING_MODEL,
        openai_api_key=openai_api_key,
    )


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

def build_vectorstore(chunks: List[Document], embeddings) -> Chroma:
    """
    Build a fresh Chroma vector store from document chunks. Each call
    creates/overwrites the collection so that re-uploading documents gives
    a clean index rather than an ever-growing one with stale data.

    Args:
        chunks: List of chunked Document objects (with source/page metadata).
        embeddings: Embeddings object from get_embeddings().

    Returns:
        A populated Chroma vector store instance.
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.PERSIST_DIRECTORY,
    )
    logger.info("Vector store built with %d chunks", len(chunks))
    return vectorstore


# ---------------------------------------------------------------------------
# RAG Chain (LCEL)
# ---------------------------------------------------------------------------

def _format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Concatenate retrieved chunks into a single context string, tagging each
    chunk with its source and page so the LLM has provenance information
    available if needed (though citations are ultimately rendered by the UI
    from raw metadata, not parsed from the LLM's text).
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore: Chroma, openai_api_key: str, model_name: str = None):
    """
    Construct the full RAG chain using LCEL.

    The chain:
      1. Takes a user question.
      2. Retrieves top-k relevant chunks from the vector store.
      3. Formats those chunks into the system prompt's {context} slot.
      4. Sends the prompt + question to the LLM.
      5. Returns BOTH the generated answer text and the raw source
         Documents (so the UI can render citations independent of what
         the model chose to say).

    Args:
        vectorstore: A populated Chroma vector store.
        openai_api_key: OpenAI API key for the chat model.
        model_name: Override for the chat model (defaults to config value).

    Returns:
        A Runnable that accepts {"question": str} and returns
        {"answer": str, "source_documents": List[Document]}.
    """
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.RETRIEVER_TOP_K}
    )

    llm = ChatOpenAI(
        model=model_name or config.OPENAI_CHAT_MODEL,
        temperature=config.LLM_TEMPERATURE,
        openai_api_key=openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", config.SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def _build_prompt_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "context": _format_docs_for_prompt(inputs["source_documents"]),
            "question": inputs["question"],
        }

    answer_chain = (
        _build_prompt_inputs
        | prompt
        | llm
        | StrOutputParser()
    )

    # Final composed chain: run retrieval once, reuse its output both for
    # generating the answer and for returning citations to the caller.
    full_chain = RunnableParallel(
        source_documents=retriever,
        question=RunnablePassthrough(),
    ) | RunnableParallel(
        answer=answer_chain,
        source_documents=lambda x: x["source_documents"],
    )

    return full_chain


def query_rag_chain(chain, question: str) -> Dict[str, Any]:
    """
    Thin wrapper around invoking the chain, with basic error handling so
    the Streamlit layer can display a friendly message on failure.

    Args:
        chain: The Runnable returned by build_rag_chain().
        question: The user's natural-language question.

    Returns:
        Dict with "answer" (str) and "source_documents" (List[Document]).
    """
    try:
        result = chain.invoke(question)
        return result
    except Exception as e:
        logger.error("RAG chain invocation failed: %s", e)
        raise

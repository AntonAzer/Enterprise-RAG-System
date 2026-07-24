"""
rag_pipeline.py
---------------
Core RAG logic: embeddings, vector store management, and the retrieval +
generation chain itself, built with modern LangChain Expression Language
(LCEL) and powered by Groq API.
"""

import logging
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embeddings (Forced Local HF for Groq)
# ---------------------------------------------------------------------------

def get_embeddings():
    """
    Build an embeddings object. Since Groq focuses on LLM inference and does not
    provide text embeddings, we strictly use a local HuggingFace sentence-transformer.
    """
    hf_model = getattr(config, "HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    logger.info("Using local HuggingFace embeddings (%s) for Groq pipeline", hf_model)
    
    # Lazy import so dependencies are only loaded if needed
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=hf_model)


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

def build_vectorstore(chunks: List[Document], embeddings) -> Chroma:
    """
    Build a fresh Chroma vector store from document chunks.
    """
    collection_name = getattr(config, "COLLECTION_NAME", "pdf_documents")
    persist_dir = getattr(config, "PERSIST_DIRECTORY", None)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    logger.info("Vector store built with %d chunks", len(chunks))
    return vectorstore


# ---------------------------------------------------------------------------
# RAG Chain (LCEL)
# ---------------------------------------------------------------------------

def _format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Concatenate retrieved chunks into a single context string, tagging each
    chunk with its source and page.
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore: Chroma, groq_api_key: str, model_name: str = "llama-3.1-8b-instant"):
    """
    Construct the full RAG chain using LCEL and Groq API.
    """
    retriever_k = getattr(config, "RETRIEVER_TOP_K", 4)
    temp = getattr(config, "LLM_TEMPERATURE", 0.2)
    sys_prompt = getattr(
        config,
        "SYSTEM_PROMPT",
        "You are an expert AI assistant for document Q&A. Answer strictly based on the provided context.\n\nContext:\n{context}"
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": retriever_k}
    )

    # استخدام ChatGroq بدلاً من جوجل
    llm = ChatGroq(
        model_name=model_name,
        temperature=temp,
        groq_api_key=groq_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
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
    Thin wrapper around invoking the chain with basic error handling.
    """
    try:
        result = chain.invoke(question)
        return result
    except Exception as e:
        logger.error("RAG chain invocation failed: %s", e)
        raise
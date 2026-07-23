"""
config.py
---------
Central configuration for the Enterprise Document Q&A RAG System.

Keeping all tunable parameters, model names, and paths in a single file
is a production best-practice: it means you never have to hunt through
business logic to change a model name, chunk size, or prompt.
"""

import os

# ---------------------------------------------------------------------------
# LLM & Embedding Model Configuration
# ---------------------------------------------------------------------------

# Chat model used to generate answers. gpt-4o-mini is a strong, low-cost
# default for RAG use cases. Swap to "gpt-4o" for higher quality if needed.
OPENAI_CHAT_MODEL = "gpt-4o-mini"

# OpenAI embedding model. "text-embedding-3-small" is cheap and performs
# very well for document retrieval tasks.
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Lightweight local fallback embedding model (no API key required).
# Useful for demos, offline development, or when an OpenAI key isn't set.
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Temperature controls creativity. For factual document Q&A we want the
# model to be as deterministic and grounded as possible.
LLM_TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Text Chunking Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------------------------------------------------------------------
# Vector Store Configuration
# ---------------------------------------------------------------------------

# Directory where Chroma persists its embeddings on disk. Using a persistent
# store (rather than pure in-memory) means the app survives a restart
# without needing to re-embed documents.
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

# Name of the Chroma collection. Kept constant here; app.py resets/rebuilds
# it whenever new documents are uploaded.
COLLECTION_NAME = "enterprise_docs"

# Number of chunks to retrieve per query.
RETRIEVER_TOP_K = 4

# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

# The single most important guardrail in this whole app: it forces the LLM
# to answer ONLY from retrieved context, and gives it an explicit escape
# hatch ("I cannot find this information...") instead of hallucinating.
SYSTEM_PROMPT = """You are an enterprise document assistant. Your job is to \
answer the user's question using ONLY the context provided below, which was \
retrieved from the user's uploaded documents.

Rules you must follow strictly:
1. Answer ONLY using information found in the provided context.
2. If the context does not contain enough information to answer the \
question, respond exactly with: "I cannot find this information in the \
provided document."
3. Do NOT use any outside knowledge, assumptions, or information not \
present in the context.
4. Be concise and factual. Do not speculate.
5. Do not mention that you were given a "context" - answer naturally, as \
if you had read the documents yourself.

Context:
{context}
"""

# ---------------------------------------------------------------------------
# Misc App Settings
# ---------------------------------------------------------------------------

APP_TITLE = "Enterprise Document Q&A (RAG)"
APP_ICON = "📄"

# Temp directory for storing uploaded PDFs before they are parsed.
UPLOAD_TEMP_DIR = os.path.join(os.getcwd(), "tmp_uploads")

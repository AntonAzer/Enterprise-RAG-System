# 📄 Enterprise Document Q&A — RAG System

A production-style Retrieval-Augmented Generation (RAG) application that
lets you upload PDF documents and ask natural-language questions about
them, with answers **strictly grounded in the source documents** and
every answer cited back to its source file and page number.

Built with **Python, Streamlit, LangChain (LCEL), ChromaDB, and the
OpenAI API**.

---

## ✨ Features

- **Strict grounding**: a custom system prompt forces the LLM to answer
  only from retrieved context, and to explicitly say
  *"I cannot find this information in the provided document"* rather
  than hallucinate.
- **Source citations**: every answer is accompanied by the exact
  filename and page number(s) it was derived from.
- **Modern LCEL pipeline**: built with LangChain Expression Language
  instead of the legacy `RetrievalQA` chain, giving full control over
  the retrieval → prompt → generation → citation data flow.
- **Pluggable embeddings**: OpenAI embeddings by default, with a free
  local HuggingFace sentence-transformer fallback for zero-cost demos.
- **Clean modular architecture**: config, document processing, RAG
  pipeline, and UI are fully separated for readability and easy
  extension.

---

## 🗂️ Project Structure

```
.
├── app.py                  # Streamlit UI (chat interface, sidebar controls)
├── rag_pipeline.py         # Embeddings, vector store, LCEL RAG chain
├── document_processor.py   # PDF loading, chunking, metadata tagging
├── config.py                # Central configuration (models, prompts, params)
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

### 1. Clone and set up a virtual environment

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** the local embedding fallback pulls in `sentence-transformers`
> and `torch` (~1-2 GB). If you always have an OpenAI key available and
> don't need the free fallback, you can remove those two lines from
> `requirements.txt` for a lighter install.

### 3. Provide your OpenAI API key

You have two options:

**Option A — enter it in the app** (fastest for local testing): just
paste it into the "OpenAI API Key" field in the sidebar when the app
launches.

**Option B — use Streamlit secrets** (recommended, matches deployment):
create a file at `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
```

The app checks `st.secrets` first and will auto-fill the key if found.

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

### 5. Use it

1. Upload one or more PDF files in the sidebar.
2. Click **Process Documents** (this chunks, embeds, and indexes them
   into ChromaDB).
3. Ask questions in the chat box at the bottom.
4. Expand **📚 Sources** under any answer to see exactly which file and
   page it came from.
5. Use **Clear Chat History** to reset the conversation (documents stay
   indexed).

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this project to a public (or private, with Cloud access) GitHub
   repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click
   **New app**.
3. Select your repository, branch, and set the main file path to
   `app.py`.
4. Before deploying, open **Advanced settings → Secrets** and add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
   This populates `st.secrets["OPENAI_API_KEY"]`, which `app.py` reads
   automatically — end users won't need to paste a key themselves.
5. Click **Deploy**. Streamlit Cloud will install everything from
   `requirements.txt` and launch the app.

### Notes on Cloud deployment
- Streamlit Community Cloud's filesystem is ephemeral — the Chroma
  `persist_directory` will reset on redeploys/restarts. That's expected;
  users simply re-upload and re-process their PDFs after a cold start.
- If you removed the HuggingFace fallback dependencies to keep the app
  lightweight, make sure `OPENAI_API_KEY` is always set (via secrets),
  since embeddings will then require it.

---

## 🧠 How It Works (Architecture Overview)

```
PDF Upload
   │
   ▼
document_processor.py
   • PyPDFLoader extracts text per page
   • RecursiveCharacterTextSplitter chunks text (1000 chars, 200 overlap)
   • Each chunk tagged with metadata: {source: filename, page: N}
   │
   ▼
rag_pipeline.py
   • Chunks embedded (OpenAI or local HuggingFace model)
   • Stored in a persistent ChromaDB collection
   • LCEL chain: question → retriever (top-k) → context formatting →
     system prompt → LLM → answer
   • Source documents returned alongside the answer (independent of
     what the LLM says, so citations are always accurate)
   │
   ▼
app.py
   • Renders chat UI, session state, sidebar controls
   • Displays answer + expandable "Sources" citation list
```

---

## 🔧 Configuration

All tunable parameters live in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model used for answer generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `HUGGINGFACE_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local fallback embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Text splitting parameters |
| `RETRIEVER_TOP_K` | `4` | Number of chunks retrieved per query |
| `LLM_TEMPERATURE` | `0.0` | Kept at 0 for deterministic, factual answers |

---

## 📌 Portfolio Talking Points

If you're presenting this project, here are the design decisions worth
highlighting:

- **Why LCEL over `RetrievalQA`**: LCEL exposes the full data flow as
  composable `Runnable` objects, making it straightforward to return
  both the generated answer *and* the raw retrieved documents in one
  chain invocation — critical for reliable source citation.
- **Why metadata-based citation instead of asking the LLM to cite**:
  LLMs can fabricate or misstate citations. By tagging every chunk with
  its true source/page at ingestion time and reading that metadata back
  from the retriever's output (not from the LLM's generated text),
  citations are guaranteed accurate.
- **Why a strict "context-only" system prompt**: this is the core
  anti-hallucination guardrail for enterprise document Q&A, where wrong
  answers on internal policy/compliance/financial documents carry real
  risk.

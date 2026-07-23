"""
app.py
------
Streamlit entry point for the Enterprise Document Q&A RAG System.

Run with:
    streamlit run app.py

Responsibilities:
- Render the sidebar (API key input, PDF upload, processing controls).
- Render the main chat interface (st.chat_message / st.chat_input).
- Wire together document_processor.py and rag_pipeline.py, storing the
  built vector store and chain in st.session_state so they persist across
  reruns (Streamlit reruns the whole script on every interaction).
"""

import logging
import streamlit as st

import config
from document_processor import process_uploaded_files
from rag_pipeline import get_embeddings, build_vectorstore, build_rag_chain, query_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title=config.APP_TITLE, page_icon=config.APP_ICON, layout="wide")


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
# Streamlit reruns this whole script top-to-bottom on every user
# interaction, so anything that needs to survive between interactions
# (chat history, the vector store, the compiled chain) must live in
# st.session_state.

def init_session_state():
    defaults = {
        "chat_history": [],       # List[Dict]: {"role": "user"/"assistant", "content": str, "sources": [...] }
        "vectorstore": None,      # Chroma instance once documents are processed
        "rag_chain": None,        # Compiled LCEL chain
        "docs_processed": False, # Whether at least one successful processing run has occurred
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# Sidebar: API Key + Document Upload
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    """
    Resolve the OpenAI API key with the following priority:
      1. st.secrets["OPENAI_API_KEY"]  (used in Streamlit Community Cloud
         deployments -- set via the app's "Secrets" settings panel)
      2. Manual text input in the sidebar (for local use / quick testing)

    Returns:
        The resolved API key string, or "" if none is available.
    """
    secret_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    if secret_key:
        st.sidebar.success("✅ API key loaded from st.secrets")
        return secret_key

    return st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Your key is only used for this session and is never stored.",
    )


with st.sidebar:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.markdown("---")

    st.subheader("🔑 Configuration")
    api_key = get_api_key()

    use_local_embeddings = st.checkbox(
        "Use free local embeddings (no API key needed for embedding step)",
        value=not bool(api_key),
        help="Uses a local HuggingFace sentence-transformer instead of "
             "OpenAI embeddings. The chat model still requires an API key.",
    )

    st.markdown("---")
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_clicked = st.button("Process Documents", type="primary", use_container_width=True)

    st.markdown("---")
    clear_clicked = st.button("🗑️ Clear Chat History", use_container_width=True)

    if st.session_state.docs_processed:
        st.success("Documents indexed and ready for questions.")


# ---------------------------------------------------------------------------
# Sidebar Actions
# ---------------------------------------------------------------------------

if clear_clicked:
    st.session_state.chat_history = []
    st.rerun()

if process_clicked:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF first.")
    elif not api_key:
        st.sidebar.error("Please provide an OpenAI API key (required for the chat model).")
    else:
        with st.spinner("Extracting text, chunking, and building the vector index..."):
            try:
                # Step 1: Extract + chunk all uploaded PDFs.
                chunks = process_uploaded_files(uploaded_files)

                # Step 2: Embed chunks and build the Chroma vector store.
                embeddings = get_embeddings(
                    openai_api_key=api_key,
                    use_local_fallback=use_local_embeddings,
                )
                vectorstore = build_vectorstore(chunks, embeddings)

                # Step 3: Compile the LCEL RAG chain against the new store.
                rag_chain = build_rag_chain(vectorstore, openai_api_key=api_key)

                # Persist in session state for use across reruns.
                st.session_state.vectorstore = vectorstore
                st.session_state.rag_chain = rag_chain
                st.session_state.docs_processed = True

                st.sidebar.success(f"✅ Processed {len(uploaded_files)} file(s) into {len(chunks)} chunks.")
            except Exception as e:
                logger.exception("Document processing failed")
                st.sidebar.error(f"❌ Processing failed: {e}")


# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------

st.header(f"{config.APP_ICON} Enterprise Document Q&A")
st.caption(
    "Upload PDFs in the sidebar, click **Process Documents**, then ask "
    "questions below. Answers are grounded strictly in your documents."
)

# Render existing chat history.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.markdown(f"- **{src['source']}**, page {src['page']}")

# Chat input box.
user_question = st.chat_input("Ask a question about your uploaded documents...")

if user_question:
    # Guard rails: don't let the user query before the pipeline is ready.
    if not st.session_state.rag_chain:
        st.error("Please upload and process at least one document before asking questions.")
    else:
        # Display the user's message immediately.
        st.session_state.chat_history.append({"role": "user", "content": user_question, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate and display the assistant's answer.
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = query_rag_chain(st.session_state.rag_chain, user_question)
                    answer = result["answer"]
                    source_docs = result["source_documents"]

                    # De-duplicate (source, page) pairs since multiple
                    # chunks can come from the same page.
                    seen = set()
                    sources = []
                    for doc in source_docs:
                        key = (doc.metadata.get("source", "unknown"), doc.metadata.get("page", "?"))
                        if key not in seen:
                            seen.add(key)
                            sources.append({"source": key[0], "page": key[1]})

                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 Sources"):
                            for src in sources:
                                st.markdown(f"- **{src['source']}**, page {src['page']}")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    logger.exception("Query failed")
                    error_msg = f"❌ Something went wrong while answering: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": error_msg, "sources": [],
                    })

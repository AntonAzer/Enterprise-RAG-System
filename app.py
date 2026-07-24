"""
app.py
------
Streamlit entry point for the Enterprise Document Q&A RAG System (Powered by Groq API).
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

def init_session_state():
    defaults = {
        "chat_history": [],       
        "vectorstore": None,      
        "rag_chain": None,        
        "docs_processed": False,  
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
    Safely fetch Groq API Key without throwing errors if secrets.toml is missing.
    """
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
        if secret_key:
            st.sidebar.success("✅ API key loaded from st.secrets")
            return secret_key
    except Exception:
        pass

    return st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Get your free key from Groq Console: https://console.groq.com/keys",
    )


with st.sidebar:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.markdown("---")

    st.subheader("🔑 Configuration")
    api_key = get_api_key()

    # رسالة توضيحية للمستخدم بدلاً من الـ Checkbox لأن الـ Embeddings ستكون محلية إجبارياً
    st.info("ℹ️ Local HuggingFace models will be used automatically for Document Embeddings. API Key is only used for Groq Chat inference.")

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
        st.sidebar.error("Please provide a Groq API key.")
    else:
        with st.spinner("Extracting text, chunking, and building the vector index..."):
            try:
                # Step 1: Extract + chunk all uploaded PDFs.
                chunks = process_uploaded_files(uploaded_files)

                # Step 2: Embed chunks (Forced local HF) and build Chroma store.
                embeddings = get_embeddings()
                vectorstore = build_vectorstore(chunks, embeddings)

                # Step 3: Compile the LCEL RAG chain against the new store using Groq.
                rag_chain = build_rag_chain(vectorstore, groq_api_key=api_key)

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

st.header(f"{config.APP_ICON} Enterprise Document Q&A (Groq)")
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
    if not st.session_state.rag_chain:
        st.error("Please upload and process at least one document before asking questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_question, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking (Groq is very fast!)..."):
                try:
                    result = query_rag_chain(st.session_state.rag_chain, user_question)
                    answer = result["answer"]
                    source_docs = result["source_documents"]

                    # De-duplicate (source, page) pairs
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
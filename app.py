"""Streamlit chatbot UI for the document RAG system."""
import streamlit as st

from rag.ingest import SUPPORTED, process_document
from rag.pipeline import answer
from rag.store import KnowledgeBase

st.set_page_config(page_title="DocChat RAG", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and knowledge base...")
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()


kb = get_kb()

# ---------------- Sidebar: upload + knowledge base ----------------
with st.sidebar:
    st.header("📁 Documents")
    files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD", type=[e.strip(".") for e in SUPPORTED], accept_multiple_files=True
    )
    if files and st.button("Process & add to knowledge base", type="primary"):
        for f in files:
            with st.spinner(f"Processing {f.name}..."):
                try:
                    doc = process_document(f.name, f.getvalue())
                    if kb.add(doc):
                        st.success(f"{f.name}: {doc['num_chunks']} chunks from {doc['num_pages']} page(s)")
                    elif not doc["chunks"]:
                        st.warning(f"{f.name}: no extractable text (scanned PDF?)")
                    else:
                        st.info(f"{f.name}: already in knowledge base")
                except Exception as e:
                    st.error(f"{f.name}: {e}")

    st.subheader("Knowledge base")
    docs = kb.documents()
    if not docs:
        st.caption("No documents yet.")
    for doc_id, d in docs.items():
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{d['filename']}**  \n{d['num_pages']} pages · {d['num_chunks']} chunks")
        if c2.button("🗑", key=f"del-{doc_id}", help="Remove"):
            kb.delete(doc_id)
            st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat ----------------
st.title("📄 DocChat")
st.caption("Ask questions — answers come only from your uploaded documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for i, s in enumerate(sources, 1):
                m = s["metadata"]
                st.markdown(f"**[{i}] {m['filename']} — page {m['page']}** (relevance {s['score']:.2f})")
                st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"))

if question := st.chat_input("Ask about your documents..."):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not kb.documents():
            result = {"answer": "Please upload a document first.", "sources": []}
        else:
            with st.spinner("Thinking..."):
                try:
                    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    result = answer(kb, question, history)
                except Exception as e:
                    result = {"answer": f"⚠️ Error: {e}", "sources": []}
        st.markdown(result["answer"])
        show_sources(result["sources"])
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )

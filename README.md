# DocChat — RAG Chatbot over Uploaded Documents

Upload documents, build a searchable knowledge base, and chat with an LLM that answers from those documents, with citations. All facts and figures in an answer must come from the uploaded documents.

## Requirements coverage

| Requirement | Implementation | File |
|---|---|---|
| Document upload | Streamlit multi-file uploader (PDF, DOCX, TXT, MD) | `app.py` |
| Extraction & structuring | Page-wise text extraction (incl. DOCX tables), header/footer removal, cleaning, overlapping chunks with metadata, separate chunks for glossary definitions | `rag/ingest.py` |
| Knowledge base | ChromaDB persistent store + JSON document registry; duplicate detection by content hash; delete support | `rag/store.py` |
| Embeddings & vector DB | `BAAI/bge-small-en-v1.5` (runs locally, free) + ChromaDB (cosine) | `rag/store.py` |
| RAG with LLM | Query rewriting → hybrid search → re-ranking → grounded prompt → Groq `openai/gpt-oss-120b` | `rag/pipeline.py`, `rag/store.py`, `rag/llm.py` |
| Chatbot interface | Streamlit chat with history, inline citations, source panel and document management sidebar | `app.py` |

## Architecture

### Ingestion

```
Upload (Streamlit)
    │
    ▼
Extract text page by page (pypdf / python-docx / plain text)
    │
    ▼
Remove repeated headers/footers ──► Clean
    │
    ▼
Chunk (~400 chars, 80 overlap, boundary-aware) + definition chunks
    │   structured records: {id, text, doc_id, filename, page, chunk, type}
    ▼
Embed (bge-small-en-v1.5, local) ──► ChromaDB (persistent, cosine)
                                     + documents.json registry
                                     + in-memory BM25 keyword index
```

### Answering a question

```
Question + recent chat
    │
    ▼
LLM rewrites it into 1–4 search queries (the original question is also searched)
    │
    ▼  for each query
Vector top 30 + BM25 top 30 ──► Reciprocal rank fusion ──► Cross-encoder re-rank ──► Top 6
    │
    ▼
Merge results, remove duplicates, filter by score (max 10 chunks)
    │
    ▼
Grounded system prompt + numbered excerpts + last 4 messages ──► Groq LLM
    │
    ▼
Answer with [n] citations + source panel (file, page, relevance score)
```

## Project files

```
app.py              Streamlit chat interface
rag/ingest.py       Text extraction, cleaning and chunking
rag/store.py        Embeddings, ChromaDB, hybrid search and re-ranking
rag/pipeline.py     Query rewriting, prompt and answer generation
rag/llm.py          Groq API call
requirements.txt    Python packages
```

The app creates a local `data/` folder for the knowledge base when it first runs.

## Setup

You need Python (tested on 3.12), Git and a free Groq API key from https://console.groq.com.

**1. Clone the repository**

```bash
git clone <repository-url>
cd <repository-folder>
```

**2. Create and activate a virtual environment**

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install the packages**

```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a file named `.env` in the project folder containing this line, with your own key:

```
GROQ_API_KEY=your_key_here
```

**5. Run the app**

```bash
streamlit run app.py
```

The app opens in your browser. The embedding and re-ranking models are downloaded on first run, so this needs an internet connection and may take a few minutes.

## Usage

1. Upload one or more files in the sidebar and click **Process & add to knowledge base**.
2. Ask a question in the chat.
3. Open **Sources** under an answer to see the excerpts used, with file, page and relevance score.

## Design choices

- **Groq free tier**: no paid LLM or API key was available, and Groq offers a free, fast option. The LLM call is isolated in `rag/llm.py`, so the provider can be changed there. The model can be overridden with `GROQ_MODEL`.
- **Local embeddings and re-ranking**: no API cost or rate limits for these steps.
- **ChromaDB**: embedded and persistent, with metadata filters (used for per-document deletion). No server needed.
- **Hybrid search**: vector search finds passages with similar meaning, while BM25 catches exact terms such as names, acronyms and numbers.
- **Cross-encoder re-ranking**: scores each question and passage together, which gives a more precise final ordering of the candidates.
- **Small chunks (400/80)**: keep individual facts and figures from being diluted by surrounding text.
- **Query rewriting**: a question like "How much did revenue grow from 2019 to 2021?" needs facts that may be on different pages, so each fact gets its own search. This also resolves follow-ups like "what about 2024?".

## How answers stay grounded in the documents

Strict "only repeat the text" instructions stopped the chatbot from doing calculations or interpreting results, so the rule separates facts from reasoning:

- **Facts come only from the documents.** Excerpts are the only allowed source of facts, figures, names and dates.
- **General knowledge is for understanding only.** The LLM may use it to explain a term, work out how to compute a metric or interpret a trend, but never to supply missing facts or numbers.
- **Calculations use document values.** The LLM shows the values it used and cites them.
- **Gaps are stated.** If only part of a question can be answered, the LLM says what is missing. If nothing relevant is found, it replies with a fixed "not found" message.
- **Answers can be checked.** Inline citations and the source panel show where each answer came from.

The top 3 chunks are always sent to the LLM, even when their scores are low, so open-ended questions aren't rejected by the score filter. This means the LLM, guided by the prompt, makes the final decision to refuse an off-topic question.

## Limitations and future work

- Scanned PDFs need OCR (e.g. Tesseract). They are currently reported as having no extractable text.
- Retrieved excerpts and recent chat messages are sent to the Groq API, so documents are not fully processed on-device.
- Each question makes at least two LLM calls (rewriting and answering), which uses more of the free-tier quota.

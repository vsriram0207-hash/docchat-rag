"""Knowledge base: embeddings + persistent ChromaDB, with hybrid (vector + BM25) search and re-ranking."""
import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION = "documents_bge"
CANDIDATES = 30  # chunks passed from hybrid search to the re-ranker


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class KnowledgeBase:
    def __init__(self, path: Path = DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)
        self.registry_path = path / "documents.json"
        self.model = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.client = chromadb.PersistentClient(path=str(path / "chroma"))
        self.collection = self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        self._build_bm25()

    # --- keyword index (rebuilt from ChromaDB so both stay in sync) ---
    def _build_bm25(self):
        res = self.collection.get(include=["documents", "metadatas"])
        self._ids, self._texts, self._metas = res["ids"], res["documents"], res["metadatas"]
        self._bm25 = BM25Okapi([_tokens(t) for t in self._texts]) if self._texts else None

    # --- document registry (structured store of what has been ingested) ---
    def documents(self) -> dict:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {}

    def _save_registry(self, reg: dict):
        self.registry_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    # --- write ---
    def add(self, doc: dict) -> bool:
        """Embed and store a processed document. Returns False if already present or empty."""
        reg = self.documents()
        if doc["doc_id"] in reg or not doc["chunks"]:
            return False
        texts = [c["text"] for c in doc["chunks"]]
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()
        self.collection.add(
            ids=[c["id"] for c in doc["chunks"]],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in doc["chunks"]],
        )
        reg[doc["doc_id"]] = {k: doc[k] for k in ("filename", "num_pages", "num_chunks")}
        self._save_registry(reg)
        self._build_bm25()
        return True

    def delete(self, doc_id: str):
        self.collection.delete(where={"doc_id": doc_id})
        reg = self.documents()
        reg.pop(doc_id, None)
        self._save_registry(reg)
        self._build_bm25()

    # --- read ---
    def search(self, query: str, k: int = 5) -> list[dict]:
        """Hybrid retrieval (vector + BM25, fused by reciprocal rank), then cross-encoder re-ranking.
        Returned `score` is the re-ranker relevance score (higher = more relevant)."""
        n = len(self._ids)
        if n == 0:
            return []
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
        vec = self.collection.query(query_embeddings=q, n_results=min(CANDIDATES, n))["ids"][0]
        scores = self._bm25.get_scores(_tokens(query))
        kw = [self._ids[i] for i in sorted(range(n), key=lambda i: -scores[i])[:CANDIDATES] if scores[i] > 0]

        fused = {}
        for ranking in (vec, kw):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
        cand = sorted(fused, key=fused.get, reverse=True)[:CANDIDATES]

        pos = {cid: i for i, cid in enumerate(self._ids)}
        ce = self.reranker.predict([(query, self._texts[pos[c]]) for c in cand])
        ranked = sorted(zip(cand, ce), key=lambda x: -x[1])[:k]
        return [{"text": self._texts[pos[c]], "metadata": self._metas[pos[c]], "score": float(s)} for c, s in ranked]

"""RAG pipeline: rewrite question -> retrieve per sub-query -> build grounded prompt -> generate."""
import json
import re

from .llm import generate
from .store import KnowledgeBase

TOP_K = 6          # chunks retrieved per sub-query
MAX_CONTEXT = 10   # chunks passed to the LLM after merging sub-queries
MIN_SCORE = -3.0   # re-ranker score below this = weak match (unrelated questions score around -10)
MIN_KEEP = 3       # always give the LLM at least this many chunks; it decides if they are relevant
NOT_FOUND = "I couldn't find information about that in the uploaded documents."

SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions about the user's uploaded documents.
The context excerpts below are your ONLY source of facts, figures, names and dates.

How to answer:
- When required/asked to, reason over the context like an analyst: combine information from several excerpts, compare values, and do calculations (differences, totals, averages, percentage change, growth) when the question needs them. Show the values you used.
- Use general knowledge and common sense only to understand and explain (e.g. what a term means, how to compute a metric, interpreting a trend). Never use it to supply facts or numbers that are not in the context.
- If the context only partly answers the question, answer the part you can and say clearly what is missing.
- If the context contains nothing relevant, reply: "{NOT_FOUND}"
- Cite sources inline like [1], [2] matching the excerpt numbers.
- If the answer contains numbers, preserve units if they existed in the source.
- If the document cites that figures are synthetic, cite that in your answer when using those figures.
- Write calculations in plain text (e.g. 145.0 - 131.5 = 13.5), not LaTeX.
- Be concise and clear."""

REWRITE_PROMPT = """Rewrite the user's question into 1-4 short search queries for retrieving passages from a document.
- Resolve references to earlier conversation (e.g. "what about 2024?" -> full question).
- If the question needs several separate facts (comparisons, changes between periods, multiple items), make one query per fact.
- Otherwise return the question itself, cleaned up.
Examples:
"How much did revenue grow from 2019 to 2021?" -> ["revenue in 2019", "revenue in 2021"]
"Compare the battery life of model A and model B" -> ["battery life of model A", "battery life of model B"]
"Who is the CEO?" -> ["Who is the CEO?"]
Return ONLY a JSON list of strings."""


def rewrite_queries(question: str, history: list[dict]) -> list[str]:
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    try:
        out = generate([
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"Conversation so far:\n{convo or '(none)'}\n\nQuestion: {question}"},
        ], temperature=0)
        queries = json.loads(re.search(r"\[.*\]", out, re.S).group(0))
        queries = [q for q in queries if isinstance(q, str) and q.strip()][:4]
    except Exception:
        queries = []
    return [question] + [q for q in queries if q.lower() != question.lower()]


def retrieve(kb: KnowledgeBase, queries: list[str]) -> list[dict]:
    """Search each query, merge results, drop duplicate text, keep the most relevant."""
    best = {}
    for q in queries:
        for h in kb.search(q, k=TOP_K):
            key = re.sub(r"\s+", " ", h["text"]).strip().lower()
            if key not in best or h["score"] > best[key]["score"]:
                best[key] = h
    ranked = sorted(best.values(), key=lambda h: -h["score"])
    # keep confident matches; always keep a few top chunks so the LLM can judge open-ended questions itself
    return [h for i, h in enumerate(ranked) if i < MIN_KEEP or h["score"] >= MIN_SCORE][:MAX_CONTEXT]


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (Source: {h['metadata']['filename']}, page {h['metadata']['page']})\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )


def answer(kb: KnowledgeBase, question: str, history: list[dict] | None = None) -> dict:
    history = history or []
    queries = rewrite_queries(question, history)
    hits = retrieve(kb, queries)
    if not hits:
        return {"answer": NOT_FOUND, "sources": [], "queries": queries}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-4:]:  # short conversational memory
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({
        "role": "user",
        "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}",
    })
    text = generate(messages).replace("【", "[").replace("】", "]")
    text = re.sub(r"\[(\d+)†[^\]]*\]", r"[\1]", text)  # normalise "[1†L1-L3]" style citations to "[1]"
    return {"answer": text, "sources": hits, "queries": queries}

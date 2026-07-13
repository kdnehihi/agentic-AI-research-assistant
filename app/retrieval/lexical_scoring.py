from __future__ import annotations

import math
import re
from collections import Counter


def bm25_scores_for_query(
    query: str,
    documents: dict[str, str],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    if not documents:
        return {}

    query_terms = tokenize(query)
    if not query_terms:
        return {document_id: 0.0 for document_id in documents}

    document_ids = list(documents)
    tokenized_docs = [tokenize(documents[document_id]) for document_id in document_ids]
    doc_count = len(tokenized_docs)
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_length = sum(doc_lengths) / doc_count if doc_count else 0.0

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    scores: dict[str, float] = {}
    for document_id, tokens, doc_length in zip(document_ids, tokenized_docs, doc_lengths):
        term_counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_counts.get(term, 0)
            if frequency == 0:
                continue

            idf = math.log(
                1.0
                + (doc_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * doc_length / max(avg_doc_length, 1e-9)
            )
            score += idf * frequency * (k1 + 1.0) / denominator
        scores[document_id] = score

    return scores


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    max_score = max(scores.values())
    if max_score <= 0.0:
        return {key: 0.0 for key in scores}

    return {
        key: value / max_score
        for key, value in scores.items()
    }


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

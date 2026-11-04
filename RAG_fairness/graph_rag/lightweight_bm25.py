"""Small deterministic BM25 retriever for annotation-linked topic pools."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_TOPIC_RE = re.compile(r"topic\s+['\"](.+?)['\"]", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric/possessive tokens; no stemming or stopword removal."""
    return [token.casefold() for token in _TOKEN_RE.findall(text or "")]


class LightweightBM25Retriever:
    """Pure-Python BM25Okapi over per-topic pools; no index or GPU is required."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.topk = int(config["retrieval_topk"] or 5)
        if self.topk < 1:
            raise ValueError("retrieval_topk must be positive")
        graph_config = config["graph_rag"] or {}
        manifest_path = config["pilot_manifest"] or graph_config.get("pilot_manifest")
        if not manifest_path:
            raise ValueError("pilot_manifest must identify the prepared topic pools")
        manifest_file = Path(manifest_path)
        if not manifest_file.is_file():
            raise FileNotFoundError(f"pilot manifest not found: {manifest_file}")
        try:
            self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read pilot manifest: {manifest_file}") from exc
        pool_paths = self.manifest.get("pool_files")
        if not isinstance(pool_paths, dict) or not pool_paths:
            raise ValueError("pilot manifest must contain non-empty pool_files")
        self._documents: Dict[str, List[Dict[str, Any]]] = {}
        self._models: Dict[str, Dict[str, Any]] = {}
        for topic, path_value in pool_paths.items():
            path = Path(path_value)
            if not path.is_file():
                raise FileNotFoundError(f"candidate pool for {topic!r} not found: {path}")
            documents = []
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid candidate JSONL at {path}:{line_number}") from exc
                    if not isinstance(doc, dict) or not doc.get("id") or not isinstance(doc.get("contents"), str):
                        raise ValueError(f"candidate at {path}:{line_number} needs id and contents")
                    if "is_relevant" in doc or "relevance_label" in doc:
                        raise ValueError("runtime candidate pools must not contain ground-truth relevance labels")
                    if str(doc.get("topic_id")) != str(topic):
                        raise ValueError(f"candidate topic mismatch in {path}:{line_number}")
                    documents.append(dict(doc))
            if not documents:
                raise ValueError(f"candidate pool is empty for topic {topic!r}")
            self._documents[str(topic)] = documents
            tokenized = [tokenize(doc["contents"]) for doc in documents]
            lengths = [len(tokens) for tokens in tokenized]
            frequencies = [Counter(tokens) for tokens in tokenized]
            doc_freq: Counter[str] = Counter()
            for term_counts in frequencies:
                doc_freq.update(term_counts.keys())
            idf = {
                term: math.log(len(documents) - count + 0.5) - math.log(count + 0.5)
                for term, count in doc_freq.items()
            }
            average_idf = sum(idf.values()) / len(idf) if idf else 0.0
            floor = 0.25 * average_idf
            for term, value in idf.items():
                if value < 0:
                    idf[term] = floor
            self._models[str(topic)] = {
                "frequencies": frequencies,
                "lengths": lengths,
                "average_length": sum(lengths) / len(lengths),
                "idf": idf,
            }
        self._topic_casefold = {topic.casefold(): topic for topic in self._documents}
        self._active_topic: Optional[str] = None

    def set_query_context(self, query: str) -> str:
        topic = self._topic_from_query(query)
        if topic is None:
            raise ValueError(f"could not identify a known topic in query: {query[:160]!r}")
        self._active_topic = topic
        return topic

    def _topic_from_query(self, query: str) -> Optional[str]:
        match = _TOPIC_RE.search(query or "")
        if match:
            topic = self._topic_casefold.get(match.group(1).casefold())
            if topic:
                return topic
        return self._active_topic

    def _search(self, query: str, num: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[float]]:
        topic = self._topic_from_query(query)
        if topic is None:
            raise ValueError("BM25 query has no topic context; refusing cross-topic retrieval")
        docs = self._documents[topic]
        model = self._models[topic]
        scores = [0.0] * len(docs)
        for term in tokenize(query):
            idf = model["idf"].get(term, 0.0)
            for index, term_counts in enumerate(model["frequencies"]):
                frequency = term_counts.get(term, 0)
                length_norm = 1.5 * (1 - 0.75 + 0.75 * model["lengths"][index] / model["average_length"])
                scores[index] += idf * (frequency * 2.5 / (frequency + length_norm))
        order = sorted(range(len(docs)), key=lambda index: (-scores[index], index))
        count = self.topk if num is None else int(num)
        if count < 0:
            raise ValueError("num must not be negative")
        selected = []
        selected_scores = []
        for index in order[:count]:
            doc = dict(docs[index])
            doc["score"] = scores[index]
            selected.append(doc)
            selected_scores.append(scores[index])
        return selected, selected_scores

    def search(self, query: str, num: Optional[int] = None, return_score: bool = False):
        docs, scores = self._search(query, num)
        return (docs, scores) if return_score else docs

    def batch_search(self, query_list: Sequence[str], num: Optional[int] = None, return_score: bool = False):
        all_docs: List[List[Dict[str, Any]]] = []
        all_scores: List[List[float]] = []
        for query in query_list:
            # Per-query topic parsing takes precedence over the context retained for
            # graph expansion queries, keeping batch search deterministic.
            topic = self._topic_from_query(query)
            if topic is None:
                raise ValueError(f"query has no topic context: {query[:160]!r}")
            self._active_topic = topic
            docs, scores = self._search(query, num)
            all_docs.append(docs)
            all_scores.append(scores)
        return (all_docs, all_scores) if return_score else all_docs

    def _save_cache(self) -> None:
        # This pilot uses an in-memory index and does not persist retrieval caches.
        return None

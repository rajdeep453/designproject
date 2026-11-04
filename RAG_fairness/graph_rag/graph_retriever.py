"""Graph-based evidence ranking and bounded, relevance-filtered expansion."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .build_graph import EvidenceGraph, _tokens, build_evidence_graph, extract_entities


class GraphRetriever:
    def __init__(
        self,
        retriever: Any = None,
        relevance_threshold: float = 0.1,
        max_additional_docs: int = 3,
        max_expansion_queries: int = 2,
    ):
        if max_additional_docs < 0 or max_expansion_queries < 0:
            raise ValueError("retrieval limits cannot be negative")
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0 and 1")
        self.retriever = retriever
        self.relevance_threshold = relevance_threshold
        self.max_additional_docs = max_additional_docs
        self.max_expansion_queries = max_expansion_queries

    @staticmethod
    def _doc_id(doc: Mapping[str, Any]) -> str:
        return str(doc.get("id", doc.get("doc_id", "")))

    def _score(self, query: str, doc: Mapping[str, Any]) -> float:
        explicit = doc.get("relevance_score")
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                return 0.0
        value = doc.get("is_relevant", doc.get("relevance_label"))
        if isinstance(value, str):
            value = value.strip().casefold() in {"1", "true", "relevant", "yes"}
        if value is False:
            return 0.0
        text = str(doc.get("contents") or doc.get("text") or "")
        terms = _tokens(query)
        return len(terms & _tokens(text)) / len(terms) if terms else 0.0

    def rank(self, query: str, documents: Sequence[Mapping[str, Any]], graph: Optional[EvidenceGraph] = None) -> List[Dict[str, Any]]:
        """Rank retrieved docs by question relevance and entity connectivity.

        Demographic labels do not affect score, rank, or inclusion.
        """
        graph = graph or build_evidence_graph(documents, query, self.relevance_threshold)
        query_entities = {e.casefold() for e in extract_entities(query)}
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for index, original in enumerate(documents):
            doc = dict(original)
            explicit = doc.get("is_relevant", doc.get("relevance_label"))
            if isinstance(explicit, str):
                explicit = explicit.strip().casefold() in {"1", "true", "relevant", "yes"}
            if explicit is False:
                continue
            text = str(doc.get("contents") or doc.get("text") or "")
            lexical = self._score(query, doc)
            entities = extract_entities(text)
            entity_match = any(e.casefold() in query_entities for e in entities)
            # Connected evidence is a tie-breaker only; it cannot make irrelevant
            # text relevant or use demographic membership as a relevance signal.
            connected = 0.0
            doc_id = self._doc_id(doc)
            node = f"doc:{doc_id}"
            if node in graph.nodes and entity_match:
                connected = 0.05 * len(graph.neighbors(node))
            score = lexical + (0.1 if entity_match else 0.0) + min(0.2, connected)
            doc["graph_relevance_score"] = lexical
            doc["graph_rank_score"] = score
            scored.append((score, index, doc))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [doc for _, _, doc in scored]

    def _expansion_terms(
        self,
        query: str,
        graph: EvidenceGraph,
        limit: Optional[int] = None,
        excluded_terms: Iterable[str] = (),
    ) -> List[str]:
        query_terms = _tokens(query)
        excluded = {str(term).casefold() for term in excluded_terms}
        terms: List[Tuple[float, str]] = []
        for node_id, attrs in graph.nodes.items():
            if attrs.get("kind") != "entity":
                continue
            name = str(attrs.get("name", ""))
            if name.casefold() in excluded:
                continue
            entity_terms = _tokens(name)
            if not entity_terms or entity_terms <= query_terms:
                continue
            # Entity candidates connect to relevant question terms, never to group
            # labels. This limits expansion to question-grounded graph traversal.
            related_doc_ids = graph.doc_ids_for_entities([name])
            rel_scores = []
            for doc_id in related_doc_ids:
                doc = graph.documents.get(doc_id)
                if not doc or doc.get("graph_relevance_label") is False:
                    continue
                rel_scores.append(float(doc.get("graph_relevance_score", 0.0)))
            grounding = max(rel_scores, default=0.0)
            overlap = len(entity_terms & query_terms) / max(1, len(entity_terms))
            if grounding >= self.relevance_threshold or overlap > 0:
                terms.append((grounding + overlap, name))
        terms.sort(key=lambda x: (-x[0], x[1].casefold()))
        cap = self.max_expansion_queries if limit is None else limit
        return [term for _, term in terms[:cap]]

    def retrieve_additional(
        self,
        query: str,
        graph: EvidenceGraph,
        existing_documents: Sequence[Mapping[str, Any]],
        max_documents: Optional[int] = None,
        excluded_terms: Iterable[str] = (),
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Retrieve and filter related evidence; returns (new docs, search calls)."""
        if self.retriever is None:
            return [], 0
        doc_limit = self.max_additional_docs if max_documents is None else max_documents
        if doc_limit <= 0:
            return [], 0
        anchors = self._expansion_terms(query, graph, excluded_terms=excluded_terms)
        if not anchors:
            # Query-only fallback is useful when graph text has no named entities.
            anchors = [""]
        queries = [f"{query} {anchor}".strip() for anchor in anchors]
        known_ids = {self._doc_id(d) for d in existing_documents}
        found: List[Dict[str, Any]] = []
        operations = 0
        for expanded_query in queries:
            if len(found) >= doc_limit:
                break
            try:
                batch = self.retriever.batch_search([expanded_query], num=max(1, doc_limit))
            except TypeError:
                batch = self.retriever.batch_search([expanded_query])
            except Exception as exc:
                raise RuntimeError(f"additional retrieval failed for query {expanded_query!r}") from exc
            operations += 1
            if not batch:
                continue
            for raw_doc in batch[0] if batch and isinstance(batch[0], list) else batch:
                if not isinstance(raw_doc, Mapping):
                    continue
                doc = dict(raw_doc)
                doc_id = self._doc_id(doc)
                if not doc_id or doc_id in known_ids:
                    continue
                score = self._score(query, doc)
                # Relevant explicit labels or question-grounded lexical evidence are
                # required; demographic membership alone never passes this gate.
                explicit = doc.get("is_relevant", doc.get("relevance_label"))
                if isinstance(explicit, str):
                    accepted_by_label = explicit.strip().casefold() in {"1", "true", "relevant", "yes"}
                else:
                    accepted_by_label = bool(explicit) if explicit is not None else False
                if score < self.relevance_threshold and not accepted_by_label:
                    continue
                doc["graph_relevance_score"] = score
                doc["graph_expansion_query"] = expanded_query
                found.append(doc)
                known_ids.add(doc_id)
                if len(found) >= doc_limit:
                    break
        return found, operations

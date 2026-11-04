"""Build a small in-memory graph over retrieved evidence documents."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "what", "which", "who", "with",
}


def _doc_text(doc: Mapping[str, Any]) -> str:
    return str(doc.get("contents") or doc.get("text") or doc.get("title") or "")


def _tokens(text: str) -> Set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def extract_entities(text: str) -> List[str]:
    """Extract simple proper-name candidates without external NLP dependencies."""
    entities = []
    seen = set()
    for match in _ENTITY_RE.findall(text):
        key = match.casefold()
        if key not in seen and key not in _STOPWORDS:
            entities.append(match.strip())
            seen.add(key)
    return entities


@dataclass
class EvidenceGraph:
    """A compact graph with document, entity and relation nodes and typed edges."""

    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    documents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    adjacency: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_node(self, node_id: str, **attributes: Any) -> None:
        self.nodes.setdefault(node_id, {}).update(attributes)

    def add_edge(self, source: str, target: str, relation: str, **attributes: Any) -> None:
        edge = {"source": source, "target": target, "relation": relation, **attributes}
        self.edges.append(edge)
        self.adjacency[source].add(target)
        self.adjacency[target].add(source)

    def neighbors(self, node_id: str) -> Set[str]:
        return set(self.adjacency.get(node_id, set()))

    def doc_ids_for_entities(self, entities: Iterable[str]) -> Set[str]:
        found: Set[str] = set()
        for entity in entities:
            entity_id = f"entity:{entity.casefold()}"
            for neighbor in self.neighbors(entity_id):
                if neighbor.startswith("doc:"):
                    found.add(neighbor[4:])
        return found


def _relevance(doc: Mapping[str, Any], query: str) -> Tuple[float, Optional[bool]]:
    explicit = doc.get("is_relevant")
    if explicit is None:
        explicit = doc.get("relevance_label")
    if explicit is not None:
        if isinstance(explicit, str):
            explicit = explicit.strip().casefold() in {"1", "true", "relevant", "yes"}
        explicit = bool(explicit)
    score = doc.get("relevance_score")
    if score is not None:
        try:
            value = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            value = 0.0
        return value, explicit
    query_terms = _tokens(query)
    text_terms = _tokens(_doc_text(doc))
    if not query_terms:
        return 0.0, explicit
    # Lexical overlap is a prediction proxy, not a human relevance judgment.
    return len(query_terms & text_terms) / len(query_terms), explicit


def build_evidence_graph(
    documents: Iterable[Mapping[str, Any]],
    query: str,
    relevance_threshold: float = 0.1,
) -> EvidenceGraph:
    """Create graph nodes and edges from retrieved documents.

    Demographic membership is copied only from explicit document metadata. It does
    not contribute to the relevance score. Capitalized co-mentions produce
    lightweight ``co_mentioned`` relationship nodes and edges.
    """
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance_threshold must be between 0 and 1")
    graph = EvidenceGraph()
    graph.add_node("query", kind="query", text=query)

    for index, original in enumerate(documents):
        if not isinstance(original, Mapping):
            raise TypeError(f"document at position {index} must be a mapping")
        doc = dict(original)
        doc_id = str(doc.get("id", f"retrieved-{index}"))
        node_id = f"doc:{doc_id}"
        text = _doc_text(doc)
        relevance_score, explicit_label = _relevance(doc, query)
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), Mapping) else {}
        group = (
            doc.get("demographic_group")
            or doc.get("group")
            or doc.get("gender")
            or doc.get("geography")
            or doc.get("geographic_location")
            or doc.get("protected_group")
            or metadata.get("demographic_group")
            or metadata.get("group")
        )
        attributes = doc.get("demographics") or metadata.get("demographics") or {}
        attributes = dict(attributes) if isinstance(attributes, Mapping) else {}
        for field_name in ("gender", "geography", "geographic_location", "protected_group", "non_protected_group"):
            if doc.get(field_name) is not None:
                attributes.setdefault(field_name, doc[field_name])
            if metadata.get(field_name) is not None:
                attributes.setdefault(field_name, metadata[field_name])
        if isinstance(group, str):
            attributes["group"] = group

        doc["id"] = doc_id
        doc["contents"] = text
        doc["graph_relevance_score"] = relevance_score
        doc["graph_relevance_label"] = explicit_label
        doc["demographics"] = attributes
        graph.documents[doc_id] = doc
        graph.add_node(
            node_id,
            kind="document",
            title=doc.get("title", ""),
            relevance_score=relevance_score,
            relevance_label=explicit_label,
            demographics=attributes,
            retrieval_score=doc.get("score"),
        )
        graph.add_edge("query", node_id, "retrieved_as", score=doc.get("score"))

        entities = extract_entities(text)
        entity_ids = []
        for entity in entities:
            entity_id = f"entity:{entity.casefold()}"
            graph.add_node(entity_id, kind="entity", name=entity)
            graph.add_edge(node_id, entity_id, "mentions")
            entity_ids.append(entity_id)
        # The relation is a co-mention association, not a claim of causality.
        for left_index, left_id in enumerate(entity_ids):
            for right_id in entity_ids[left_index + 1 :]:
                relation_id = f"relation:{left_id[7:]}::{right_id[7:]}"
                graph.add_node(relation_id, kind="relationship", label="co_mentioned")
                graph.add_edge(left_id, relation_id, "subject_of")
                graph.add_edge(relation_id, right_id, "object_of")
        graph.add_edge(
            node_id,
            "query",
            "supports_query" if relevance_score >= relevance_threshold else "weak_query_match",
            relevance_score=relevance_score,
        )
    return graph

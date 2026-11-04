"""Bounded retrieval controller driven by relevance-aware graph analysis."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from .build_graph import EvidenceGraph, build_evidence_graph
from .fairness_analyzer import FairnessAnalyzer
from .graph_retriever import GraphRetriever


class AdaptiveController:
    def __init__(
        self,
        graph_retriever: GraphRetriever,
        analyzer: FairnessAnalyzer,
        max_rounds: int = 1,
        max_additional_docs: int = 3,
        protected_groups: Iterable[str] = (),
        non_protected_groups: Iterable[str] = (),
    ):
        if max_rounds < 0 or max_additional_docs < 0:
            raise ValueError("adaptive retrieval limits cannot be negative")
        self.graph_retriever = graph_retriever
        self.analyzer = analyzer
        self.max_rounds = max_rounds
        self.max_additional_docs = max_additional_docs
        self.protected_groups = list(protected_groups)
        self.non_protected_groups = list(non_protected_groups)

    def run(
        self,
        query: str,
        initial_documents: Iterable[Mapping[str, Any]],
        expected_groups: Iterable[str] = (),
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze, expand only when needed, then return the final evidence set."""
        documents: List[Dict[str, Any]] = [dict(doc) for doc in initial_documents]
        graph: EvidenceGraph = build_evidence_graph(
            documents, query, self.analyzer.relevance_threshold
        )
        selected = self.graph_retriever.rank(query, documents, graph)
        if top_k is not None:
            selected = selected[:top_k]
        report = self.analyzer.analyze(
            graph, expected_groups, self.protected_groups, self.non_protected_groups
        )
        initial_report = report
        additional_operations = 0
        additional_documents = 0
        decisions = []

        for round_index in range(self.max_rounds):
            if not report["needs_more_evidence"]:
                decisions.append({"round": round_index, "action": "stop", "reason": "evidence_sufficient"})
                break
            remaining = self.max_additional_docs - additional_documents
            if remaining <= 0:
                decisions.append({"round": round_index, "action": "stop", "reason": "document_limit"})
                break
            additions, operations = self.graph_retriever.retrieve_additional(
                query,
                graph,
                documents,
                max_documents=remaining,
                excluded_terms=set(expected_groups) | set(self.protected_groups) | set(self.non_protected_groups),
            )
            additional_operations += operations
            if not additions:
                decisions.append({"round": round_index, "action": "stop", "reason": "no_relevant_candidates"})
                break
            documents.extend(additions)
            additional_documents += len(additions)
            graph = build_evidence_graph(
                documents, query, self.analyzer.relevance_threshold
            )
            selected = self.graph_retriever.rank(query, documents, graph)
            if top_k is not None:
                selected = selected[:top_k]
            report = self.analyzer.analyze(
                graph, expected_groups, self.protected_groups, self.non_protected_groups
            )
            decisions.append({
                "round": round_index,
                "action": "retrieve",
                "added_documents": len(additions),
                "search_operations": operations,
                "missing_groups_after": report["missing_groups"],
            })

        if self.max_rounds == 0 and report["needs_more_evidence"]:
            decisions.append({"round": 0, "action": "stop", "reason": "round_limit"})
        if report["needs_more_evidence"] and additional_documents >= self.max_additional_docs:
            decisions.append({"round": self.max_rounds, "action": "stop", "reason": "document_limit"})

        return {
            "documents": documents,
            "selected_documents": selected,
            "graph": graph,
            "initial_analysis": initial_report,
            "analysis": report,
            "additional_retrieval_operations": additional_operations,
            "additional_documents": additional_documents,
            "decisions": decisions,
        }

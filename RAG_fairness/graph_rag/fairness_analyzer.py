"""Relevance-aware group evidence diagnostics and fairness summaries."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from .build_graph import EvidenceGraph


class FairnessAnalyzer:
    """Analyze group coverage and relevance without assuming equal group counts."""

    def __init__(self, relevance_threshold: float = 0.1, disparity_threshold: float = 0.35):
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0 and 1")
        if not 0.0 <= disparity_threshold <= 1.0:
            raise ValueError("disparity_threshold must be between 0 and 1")
        self.relevance_threshold = relevance_threshold
        self.disparity_threshold = disparity_threshold

    @staticmethod
    def _group(doc: Mapping[str, Any]) -> Optional[str]:
        demographics = doc.get("demographics") or {}
        if isinstance(demographics, Mapping):
            value = (
                demographics.get("group")
                or demographics.get("gender")
                or demographics.get("geography")
                or demographics.get("geographic_location")
            )
            if value is not None:
                return str(value)
        return None

    def analyze(
        self,
        graph: EvidenceGraph,
        expected_groups: Iterable[str] = (),
        protected_groups: Iterable[str] = (),
        non_protected_groups: Iterable[str] = (),
    ) -> Dict[str, Any]:
        groups = {str(g) for g in expected_groups if g is not None}
        protected = {str(g) for g in protected_groups if g is not None}
        non_protected = {str(g) for g in non_protected_groups if g is not None}
        groups |= protected | non_protected
        stats: Dict[str, Dict[str, float]] = {
            g: {"documents": 0, "relevant_documents": 0, "relevance_rate": 0.0}
            for g in groups
        }
        unknown_group_docs = 0
        relevant_total = 0
        total = 0
        relevant_group_count: Dict[str, int] = {g: 0 for g in groups}

        for doc in graph.documents.values():
            total += 1
            group = self._group(doc)
            explicit_label = doc.get("graph_relevance_label")
            score = float(doc.get("graph_relevance_score", 0.0) or 0.0)
            relevant = bool(explicit_label) if explicit_label is not None else score >= self.relevance_threshold
            if relevant:
                relevant_total += 1
            if group is None:
                unknown_group_docs += 1
                continue
            stats.setdefault(group, {"documents": 0, "relevant_documents": 0, "relevance_rate": 0.0})
            relevant_group_count.setdefault(group, 0)
            stats[group]["documents"] += 1
            if relevant:
                stats[group]["relevant_documents"] += 1
                relevant_group_count[group] += 1

        for group_stats in stats.values():
            group_stats["relevance_rate"] = (
                group_stats["relevant_documents"] / group_stats["documents"]
                if group_stats["documents"] else 0.0
            )

        comparison_groups = [
            g for g in relevant_group_count
            if stats.get(g, {}).get("documents", 0) > 0 or g in groups
        ]
        shares = [relevant_group_count[g] / relevant_total for g in comparison_groups] if relevant_total else []
        group_disparity = max(shares) - min(shares) if len(shares) > 1 else 0.0
        rates = [stats[g]["relevance_rate"] for g in stats if stats[g]["documents"] > 0]
        rate_gap = max(rates) - min(rates) if len(rates) > 1 else (1.0 if rates and rates[0] == 0 else 0.0)

        # Missing groups only cause a retrieval request when the caller explicitly
        # supplied expected groups. This is a coverage signal, not a 50/50 target.
        group_labels_available = bool(total and unknown_group_docs < total)
        missing_groups = (
            [g for g in sorted(groups) if relevant_group_count.get(g, 0) == 0]
            if group_labels_available else []
        )
        return {
            "groups": stats,
            "group_roles": {
                g: "protected" if g in protected else "non_protected" if g in non_protected else "unspecified"
                for g in sorted(stats)
            },
            "total_documents": total,
            "relevant_documents": relevant_total,
            "unknown_group_documents": unknown_group_docs,
            "missing_groups": missing_groups,
            "group_disparity": group_disparity,
            "relevance_rate_gap": rate_gap,
            "needs_more_evidence": group_labels_available and (bool(missing_groups) or rate_gap > self.disparity_threshold),
            "group_labels_available": group_labels_available and unknown_group_docs == 0,
            "fairness_metric_note": "Group disparity is the max-minus-min relevant-evidence share; equal representation is not assumed.",
        }

    @staticmethod
    def equalized_odds(
        y_true: Iterable[bool],
        y_pred: Iterable[bool],
        groups: Iterable[str],
    ) -> Optional[Dict[str, Any]]:
        """Return group TPR/FPR and their gaps; None when labels are not supplied.

        This is a document relevance-classification metric. It requires externally
        labeled relevance truth and must not be inferred from demographic membership.
        """
        truth, pred, group_values = list(y_true), list(y_pred), list(groups)
        if not (len(truth) == len(pred) == len(group_values)):
            raise ValueError("y_true, y_pred, and groups must have equal lengths")
        if not truth:
            return None
        per_group: Dict[str, Dict[str, float]] = {}
        for group in sorted(set(str(g) for g in group_values)):
            rows = [(bool(t), bool(p)) for t, p, g in zip(truth, pred, group_values) if str(g) == group]
            positives = [p for t, p in rows if t]
            negatives = [p for t, p in rows if not t]
            per_group[group] = {
                "tpr": sum(positives) / len(positives) if positives else None,
                "fpr": sum(negatives) / len(negatives) if negatives else None,
            }
        tprs = [v["tpr"] for v in per_group.values() if v["tpr"] is not None]
        fprs = [v["fpr"] for v in per_group.values() if v["fpr"] is not None]
        return {
            "groups": per_group,
            "tpr_gap": max(tprs) - min(tprs) if len(tprs) > 1 else None,
            "fpr_gap": max(fprs) - min(fprs) if len(fprs) > 1 else None,
        }

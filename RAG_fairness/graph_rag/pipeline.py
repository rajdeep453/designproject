"""Composition adapter that adds graph selection to existing FlashRAG components."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Mapping, Optional

from .adaptive_controller import AdaptiveController
from .build_graph import build_evidence_graph
from .fairness_analyzer import FairnessAnalyzer
from .graph_retriever import GraphRetriever


class GraphRAGPipeline:
    """Use an existing SequentialPipeline's retriever, prompt, generator and evaluator."""

    MODES = {"graph", "adaptive_fair"}

    def __init__(self, sequential_pipeline: Any, config: Mapping[str, Any], mode: Optional[str] = None):
        self.pipeline = sequential_pipeline
        self.config = config
        if getattr(sequential_pipeline, "generator", None) is None:
            raise ValueError("graph RAG mode requires a generator on SequentialPipeline; disable refiner mode for this prototype")
        self.settings = self._get(config, "graph_rag", {}) or {}
        self.mode = mode or self._get(config, "rag_mode", "graph")
        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if not isinstance(self.settings, Mapping):
            raise ValueError("graph_rag configuration must be a mapping")
        self.relevance_threshold = float(self.settings.get("relevance_threshold", 0.1))
        self.max_rounds = int(self.settings.get("max_rounds", 1))
        self.max_additional_docs = int(self.settings.get("max_additional_docs", 3))
        self.expected_groups = self._labels(self.settings.get("expected_groups", []), "expected_groups")
        self.protected_groups = self._labels(self.settings.get("protected_groups", []), "protected_groups")
        self.non_protected_groups = self._labels(self.settings.get("non_protected_groups", []), "non_protected_groups")
        self.label_map = self._load_label_map(self.settings.get("document_group_map"))
        self.graph_retriever = GraphRetriever(
            retriever=sequential_pipeline.retriever,
            relevance_threshold=self.relevance_threshold,
            max_additional_docs=self.max_additional_docs,
            max_expansion_queries=int(self.settings.get("max_expansion_queries", 2)),
        )
        self.analyzer = FairnessAnalyzer(
            relevance_threshold=self.relevance_threshold,
            disparity_threshold=float(self.settings.get("disparity_threshold", 0.35)),
        )
        self.controller = AdaptiveController(
            self.graph_retriever,
            self.analyzer,
            max_rounds=self.max_rounds,
            max_additional_docs=self.max_additional_docs,
            protected_groups=self.protected_groups,
            non_protected_groups=self.non_protected_groups,
        )

    @staticmethod
    def _get(config: Any, key: str, default: Any = None) -> Any:
        try:
            value = config[key]
        except (KeyError, TypeError, AttributeError):
            return default
        return default if value is None else value

    @staticmethod
    def _labels(value: Any, field: str):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(label) for label in value]
        raise ValueError(f"graph_rag.{field} must be a string or list of strings")

    @staticmethod
    def _load_label_map(path: Optional[str]) -> Dict[str, str]:
        if not path:
            return {}
        if not os.path.isfile(path):
            raise FileNotFoundError(f"document_group_map does not exist: {path}")
        try:
            with open(path, "r", encoding="utf-8") as stream:
                raw = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read document_group_map: {path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("document_group_map must be a JSON object mapping doc IDs to group labels")
        return {str(key): str(value) for key, value in raw.items()}

    def _annotate(self, documents):
        annotated = []
        for original in documents:
            doc = dict(original)
            doc_id = str(doc.get("id", doc.get("doc_id", "")))
            metadata = doc.get("metadata") if isinstance(doc.get("metadata"), Mapping) else {}
            demographics = doc.get("demographics") if isinstance(doc.get("demographics"), Mapping) else {}
            has_group = any(
                doc.get(key) is not None or metadata.get(key) is not None or demographics.get(key) is not None
                for key in ("demographic_group", "group", "gender", "geography", "geographic_location")
            )
            if doc_id in self.label_map and not has_group:
                doc["demographic_group"] = self.label_map[doc_id]
            annotated.append(doc)
        return annotated

    def run(self, dataset: Any, do_eval: bool = True, pred_process_fun=None, initial_results=None):
        started = time.perf_counter()
        if initial_results is None:
            initial_results = self.pipeline.retriever.batch_search(dataset.question)
        retrieval_finished = time.perf_counter()
        if len(initial_results) != len(dataset):
            raise ValueError("initial retrieval results must match dataset length")
        dataset.update_output("initial_retrieval_result", initial_results)
        final_results = []
        analyses = []
        initial_analyses = []
        per_query_ops = []
        per_query_additions = []
        decision_logs = []

        graph_started = time.perf_counter()
        for query, retrieved in zip(dataset.question, initial_results):
            docs = self._annotate(retrieved)
            top_k = int(self._get(self.config, "retrieval_topk", len(docs)))
            if self.mode == "adaptive_fair":
                outcome = self.controller.run(query, docs, self.expected_groups, top_k=top_k)
                selected = outcome["selected_documents"]
                analysis = outcome["analysis"]
                initial_analysis = outcome["initial_analysis"]
                operations = outcome["additional_retrieval_operations"]
                additions = outcome["additional_documents"]
                decisions = outcome["decisions"]
            else:
                graph = build_evidence_graph(docs, query, self.relevance_threshold)
                selected = self.graph_retriever.rank(query, docs, graph)[:top_k]
                analysis = self.analyzer.analyze(
                    graph, self.expected_groups, self.protected_groups, self.non_protected_groups
                )
                initial_analysis = analysis
                operations = 0
                additions = 0
                decisions = [{"action": "graph_rank"}]
            final_results.append(selected)
            analyses.append(analysis)
            initial_analyses.append(initial_analysis)
            per_query_ops.append(operations)
            per_query_additions.append(additions)
            decision_logs.append(decisions)
        graph_finished = time.perf_counter()

        dataset.update_output("retrieval_result", final_results)
        dataset.update_output("graph_fairness_analysis", analyses)
        dataset.update_output("initial_graph_fairness_analysis", initial_analyses)
        dataset.update_output("additional_retrieval_operations", per_query_ops)
        dataset.update_output("additional_documents", per_query_additions)
        dataset.update_output("graph_rag_decisions", decision_logs)
        prompts = [
            self.pipeline.prompt_template.get_string(question=q, retrieval_result=r)
            for q, r in zip(dataset.question, final_results)
        ]
        dataset.update_output("prompt", prompts)
        generation_started = time.perf_counter()
        predictions = self.pipeline.generator.generate(prompts)
        generation_finished = time.perf_counter()
        dataset.update_output("pred", predictions)
        if pred_process_fun is not None:
            raw = dataset.pred
            dataset.update_output("raw_pred", raw)
            dataset.update_output("pred", [pred_process_fun(p) for p in raw])

        evaluation_started = time.perf_counter()
        eval_result = self.pipeline.evaluator.evaluate(dataset) if do_eval else {}
        evaluation_finished = time.perf_counter()
        if self.pipeline.save_retrieval_cache:
            self.pipeline.retriever._save_cache()
        elapsed = time.perf_counter() - started
        timings = {
            "initial_retrieval_seconds": retrieval_finished - started,
            "graph_and_controller_seconds": graph_finished - graph_started,
            "generation_seconds": generation_finished - generation_started,
            "evaluation_seconds": evaluation_finished - evaluation_started,
        }
        report = self._make_report(
            dataset, initial_analyses, analyses, per_query_ops, per_query_additions,
            decision_logs, eval_result, elapsed, timings
        )
        save_dir = self._get(self.config, "save_dir")
        if not save_dir:
            raise ValueError("save_dir must be configured to save graph RAG metrics")
        report_path = os.path.join(save_dir, "graph_rag_metrics.json")
        try:
            with open(report_path, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(f"failed to save graph RAG report to {report_path}") from exc
        print(report)
        return dataset

    def _make_report(
        self, dataset, initial_analyses, analyses, ops, additions, decisions,
        eval_result, elapsed, timings
    ):
        scores = [item.output.get("metric_score", {}) for item in dataset]
        means: Dict[str, float] = {}
        for metric in {key for row in scores for key in row}:
            values = [float(row[metric]) for row in scores if isinstance(row.get(metric), (int, float))]
            if values:
                means[metric] = sum(values) / len(values)
        initial_disparities = [
            a.get("group_disparity") for a in initial_analyses if a.get("group_labels_available")
        ]
        disparities = [a.get("group_disparity") for a in analyses if a.get("group_labels_available")]
        initial_disparity_mean = (
            sum(initial_disparities) / len(initial_disparities) if initial_disparities else None
        )
        final_disparity_mean = sum(disparities) / len(disparities) if disparities else None
        labels, relevance_predictions, groups = [], [], []
        for item in dataset:
            for doc in item.retrieval_result:
                explicit = doc.get("is_relevant", doc.get("relevance_label"))
                demographics = doc.get("demographics") or {}
                group = (
                    doc.get("demographic_group") or doc.get("group") or doc.get("gender")
                    or doc.get("geography") or doc.get("geographic_location")
                )
                if isinstance(demographics, Mapping):
                    group = (
                        group or demographics.get("group") or demographics.get("gender")
                        or demographics.get("geography") or demographics.get("geographic_location")
                    )
                if explicit is None or group is None:
                    continue
                if isinstance(explicit, str):
                    explicit = explicit.strip().casefold() in {"1", "true", "relevant", "yes"}
                labels.append(bool(explicit))
                relevance_predictions.append(
                    float(doc.get("graph_relevance_score", 0.0) or 0.0) >= self.relevance_threshold
                )
                groups.append(str(group))
        eo_summary = self.analyzer.equalized_odds(labels, relevance_predictions, groups) if labels else None
        retrieval_metrics = {
            key: value for key, value in means.items()
            if key.startswith("mrr") or key.startswith("retrieval_")
        }
        graph_scores = [
            float(doc.get("graph_relevance_score", 0.0) or 0.0)
            for item in dataset for doc in item.retrieval_result
        ]
        report = {
            "mode": self.mode,
            "dataset_size": len(dataset),
            "utility_metrics": means,
            "evaluator_metrics": eval_result,
            "retrieval_performance": {
                "evaluator_metrics": retrieval_metrics,
                "mean_graph_relevance_score_proxy": (
                    sum(graph_scores) / len(graph_scores) if graph_scores else None
                ),
            },
            "initial_group_disparity_mean": initial_disparity_mean,
            "group_disparity_mean": final_disparity_mean,
            "group_disparity_reduction": (
                initial_disparity_mean - final_disparity_mean
                if initial_disparity_mean is not None and final_disparity_mean is not None else None
            ),
            "equalized_odds": eo_summary,
            "mean_retrieved_documents": (
                sum(len(item.retrieval_result) for item in dataset) / len(dataset) if len(dataset) else 0.0
            ),
            "additional_retrieval_operations": sum(ops),
            "additional_documents": sum(additions),
            "adaptive_triggered_queries": sum(1 for value in ops if value > 0),
            "controller_decisions": decisions,
            "timing_breakdown_seconds": timings,
            "elapsed_seconds": elapsed,
            "note": "Equalized odds is reported only when explicit relevance labels are supplied. Group disparity is omitted when demographic labels are unavailable.",
        }
        return report

"""Run real-data retrieval/graph/controller diagnostics without generation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from .lightweight_bm25 import LightweightBM25Retriever

from .adaptive_controller import AdaptiveController
from .build_graph import build_evidence_graph
from .fairness_analyzer import FairnessAnalyzer
from .graph_retriever import GraphRetriever


def _doc_id(doc: Mapping[str, Any]) -> str:
    return str(doc.get("id", doc.get("doc_id", "")))


def _offline_scores(retrieved: Sequence[Mapping[str, Any]], topic: str, qrels: Mapping[str, Any], groups: Sequence[str]):
    topic_labels = qrels.get(topic, {})
    group_totals = {group: {"relevant": 0, "irrelevant": 0} for group in groups}
    relevant_ids = {doc_id for doc_id, label in topic_labels.items() if label.get("relevant") is True}
    reciprocal_rank = 0.0
    first_relevant_rank = None
    judged_count = 0
    for rank, doc in enumerate(retrieved, 1):
        doc_id = _doc_id(doc)
        label = topic_labels.get(doc_id)
        if label is None:
            continue
        judged_count += 1
        group = str(label["gender"])
        group_totals.setdefault(group, {"relevant": 0, "irrelevant": 0})
        bucket = "relevant" if label["relevant"] else "irrelevant"
        group_totals[group][bucket] += 1
        if label["relevant"] and first_relevant_rank is None:
            first_relevant_rank = rank
            reciprocal_rank = 1.0 / rank
    rel_by_group = {group: group_totals.get(group, {}).get("relevant", 0) for group in groups}
    total_relevant = sum(rel_by_group.values())
    shares = [rel_by_group[group] / total_relevant if total_relevant else 0.0 for group in groups]
    disparity = max(shares) - min(shares) if len(shares) > 1 else None
    return {
        "judged_retrieved_count": judged_count,
        "relevant_irrelevant_by_group": group_totals,
        "relevant_evidence_group_disparity": disparity,
        "mrr_first_relevant_at_5": reciprocal_rank,
        "first_relevant_rank": first_relevant_rank,
        "relevant_candidate_count": len(relevant_ids),
    }


def run_retrieval_check(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("pilot config must be a YAML mapping")
    split_name = config["split"][0]
    dataset_path = Path(config["data_dir"]) / config["dataset_name"] / f"{split_name}.jsonl"
    dataset = []
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid pilot question at {dataset_path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"pilot question at {dataset_path}:{line_number} must be an object")
            dataset.append(SimpleNamespace(
                id=row.get("id"),
                question=row.get("question"),
                metadata=row.get("metadata", {}),
            ))
    retriever = LightweightBM25Retriever(config)
    questions = [item.question for item in dataset]
    topics = [str(item.metadata.get("topic_id", "")) for item in dataset]
    if len(questions) != 10:
        raise ValueError(f"expected exactly 10 prepared questions, got {len(questions)}")
    for question, topic in zip(questions, topics):
        retriever.set_query_context(question)
        if retriever._active_topic != topic:
            raise ValueError(f"question topic {topic!r} routed to {retriever._active_topic!r}")

    expected_groups = list(config["graph_rag"]["expected_groups"])
    protected_groups = list(config["graph_rag"]["protected_groups"])
    non_protected_groups = list(config["graph_rag"]["non_protected_groups"])
    threshold = float(config["graph_rag"]["relevance_threshold"])
    graph_retriever = GraphRetriever(
        retriever=retriever,
        relevance_threshold=threshold,
        max_additional_docs=int(config["graph_rag"]["max_additional_docs"]),
        max_expansion_queries=int(config["graph_rag"]["max_expansion_queries"]),
    )
    analyzer = FairnessAnalyzer(
        relevance_threshold=threshold,
        disparity_threshold=float(config["graph_rag"]["disparity_threshold"]),
    )
    controller = AdaptiveController(
        graph_retriever,
        analyzer,
        max_rounds=int(config["graph_rag"]["max_rounds"]),
        max_additional_docs=int(config["graph_rag"]["max_additional_docs"]),
        protected_groups=protected_groups,
        non_protected_groups=non_protected_groups,
    )

    # Inference phase: qrels are intentionally not loaded until all retrieval and
    # controller decisions are complete.
    modes: Dict[str, Dict[str, Any]] = {
        name: {"items": [], "initial_documents": 0, "final_documents": 0,
               "additional_retrieval_operations": 0, "additional_documents": 0,
               "elapsed_seconds": 0.0}
        for name in ("baseline_retrieval", "graph_retrieval", "adaptive_retrieval")
    }
    for question, topic, item in zip(questions, topics, dataset):
        retriever.set_query_context(question)
        started = time.perf_counter()
        initial = retriever.batch_search([question], num=int(config["retrieval_topk"]))[0]
        baseline_seconds = time.perf_counter() - started
        for doc in initial:
            if "is_relevant" in doc or "relevance_label" in doc:
                raise ValueError("ground-truth relevance leaked into a runtime retrieval document")
        modes["baseline_retrieval"]["items"].append({
            "question_id": str(item.id), "topic_id": topic,
            "retrieved_document_ids": [_doc_id(doc) for doc in initial],
        })
        modes["baseline_retrieval"]["initial_documents"] += len(initial)
        modes["baseline_retrieval"]["final_documents"] += len(initial)
        modes["baseline_retrieval"]["elapsed_seconds"] += baseline_seconds

        started = time.perf_counter()
        graph = build_evidence_graph(initial, question, threshold)
        graph_docs = graph_retriever.rank(question, initial, graph)[: int(config["retrieval_topk"])]
        graph_seconds = time.perf_counter() - started
        modes["graph_retrieval"]["items"].append({
            "question_id": str(item.id), "topic_id": topic,
            "retrieved_document_ids": [_doc_id(doc) for doc in graph_docs],
        })
        modes["graph_retrieval"]["initial_documents"] += len(initial)
        modes["graph_retrieval"]["final_documents"] += len(graph_docs)
        modes["graph_retrieval"]["elapsed_seconds"] += graph_seconds

        started = time.perf_counter()
        adaptive = controller.run(question, initial, expected_groups, top_k=int(config["retrieval_topk"]))
        adaptive_seconds = time.perf_counter() - started
        final_docs = adaptive["selected_documents"]
        modes["adaptive_retrieval"]["items"].append({
            "question_id": str(item.id), "topic_id": topic,
            "initial_document_ids": [_doc_id(doc) for doc in initial],
            "additional_document_ids": sorted(
                {_doc_id(doc) for doc in adaptive["documents"]} - {_doc_id(doc) for doc in initial}
            ),
            "retrieved_document_ids": [_doc_id(doc) for doc in final_docs],
            "controller_decisions": adaptive["decisions"],
            "initial_analysis": adaptive["initial_analysis"],
            "final_analysis": adaptive["analysis"],
        })
        modes["adaptive_retrieval"]["initial_documents"] += len(initial)
        modes["adaptive_retrieval"]["final_documents"] += len(final_docs)
        modes["adaptive_retrieval"]["additional_retrieval_operations"] += adaptive["additional_retrieval_operations"]
        modes["adaptive_retrieval"]["additional_documents"] += adaptive["additional_documents"]
        modes["adaptive_retrieval"]["elapsed_seconds"] += adaptive_seconds

    # Offline evaluation phase: only now load annotations and join on topic/doc IDs.
    qrels_path = Path(config["pilot_qrels_path"])
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    if not isinstance(qrels, dict):
        raise ValueError("pilot qrels must map topics to document annotations")
    for mode in modes.values():
        per_query = []
        for row in mode["items"]:
            ids = row.get("retrieved_document_ids", [])
            docs = [{"id": doc_id} for doc_id in ids]
            per_query.append(_offline_scores(docs, row["topic_id"], qrels, expected_groups))
        mode["offline_annotation_metrics"] = {
            "mean_mrr_first_relevant_at_5": (
                sum(row["mrr_first_relevant_at_5"] for row in per_query) / len(per_query) if per_query else None
            ),
            "relevant_irrelevant_by_group": {
                group: {
                    "relevant": sum(row["relevant_irrelevant_by_group"].get(group, {}).get("relevant", 0) for row in per_query),
                    "irrelevant": sum(row["relevant_irrelevant_by_group"].get(group, {}).get("irrelevant", 0) for row in per_query),
                }
                for group in expected_groups
            },
            "relevant_evidence_group_disparity_mean": (
                sum(row["relevant_evidence_group_disparity"] for row in per_query
                    if row["relevant_evidence_group_disparity"] is not None)
                / sum(1 for row in per_query if row["relevant_evidence_group_disparity"] is not None)
                if any(row["relevant_evidence_group_disparity"] is not None for row in per_query) else None
            ),
            "equalized_odds": {
                "status": "N/A",
                "reason": "The released split has only relevant male and irrelevant female documents; each group lacks one relevance class, so group-conditional TPR/FPR gaps are not estimable.",
            },
            "per_query": per_query,
        }
        if mode is modes["adaptive_retrieval"]:
            addition_counts: Dict[str, Dict[str, int]] = {}
            for row in mode["items"]:
                for doc_id in row.get("additional_document_ids", []):
                    label = qrels.get(row["topic_id"], {}).get(doc_id)
                    if label is None:
                        continue
                    group = str(label["gender"])
                    bucket = "relevant" if label["relevant"] else "irrelevant"
                    addition_counts.setdefault(group, {"relevant": 0, "irrelevant": 0})[bucket] += 1
            mode["offline_annotation_metrics"]["additional_retrieval_annotation_counts"] = addition_counts
        mode["average_latency_seconds_per_query"] = mode["elapsed_seconds"] / len(questions)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "retrieval_only_precheck_not_a_complete_rag_experiment",
        "scope": "real TREC-derived questions and golden documents in topic-specific restricted judged pools; not full-corpus retrieval",
        "generator_ran": False,
        "ground_truth_used_for_controller": False,
        "question_count": len(questions),
        "question_ids": [str(item.id) for item in dataset],
        "topics": topics,
        "retrieval_method": "pure-Python BM25Okapi formula; lowercase regex tokens [A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*; no stemming/stopword filtering; k1=1.5, b=0.75, epsilon=0.25; stable original-order tie break",
        "top_k": int(config["retrieval_topk"]),
        "config_path": str(Path(config_path).resolve()),
        "qrels_path_used_after_inference": str(qrels_path.resolve()),
        "methods": modes,
        "unavailable_metrics": ["Exact Match", "ROUGE-1", "answer group disparity", "generation latency"],
        "note": "These measurements are retrieval diagnostics only; no baseline/graph/adaptive RAG experiment is claimed complete without generation.",
    }
    (output_dir / "retrieval_only_precheck.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "config_used.yaml").write_text(
        Path(config_path).read_text(encoding="utf-8"), encoding="utf-8"
    )
    csv_path = output_dir / "retrieval_only_precheck.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["Method", "Questions", "Initial Retrieved", "Final Retrieved", "Additional Operations",
                  "Additional Documents", "MRR@5", "Relevant Evidence Group Disparity", "Avg Retrieval Latency (s)", "Equalized Odds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, mode in modes.items():
            scores = mode["offline_annotation_metrics"]
            writer.writerow({
                "Method": name,
                "Questions": len(mode["items"]),
                "Initial Retrieved": mode["initial_documents"],
                "Final Retrieved": mode["final_documents"],
                "Additional Operations": mode["additional_retrieval_operations"],
                "Additional Documents": mode["additional_documents"],
                "MRR@5": scores["mean_mrr_first_relevant_at_5"],
                "Relevant Evidence Group Disparity": scores["relevant_evidence_group_disparity_mean"],
                "Avg Retrieval Latency (s)": mode["average_latency_seconds_per_query"],
                "Equalized Odds": scores["equalized_odds"]["status"],
            })
    (output_dir / "run_log.txt").write_text(
        "Adaptive Fair Graph RAG restricted real-data retrieval precheck\n"
        f"Status: {report['status']}\n"
        f"Questions: {report['question_count']}\n"
        f"Topics: {', '.join(report['topics'])}\n"
        f"Retrieval method: {report['retrieval_method']}\n"
        f"Top-k: {report['top_k']}\n"
        f"Generator ran: {report['generator_ran']}\n"
        "This is a retrieval-only diagnostic on restricted judged pools, not a full-corpus or generated-answer experiment.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("graph_rag/pilot10_gender.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/real_experiments/pilot10_gender/retrieval_only_precheck"))
    args = parser.parse_args()
    report = run_retrieval_check(args.config, args.output_dir)
    print(json.dumps({"status": report["status"], "question_count": report["question_count"],
                      "modes": list(report["methods"])}, indent=2))


if __name__ == "__main__":
    main()

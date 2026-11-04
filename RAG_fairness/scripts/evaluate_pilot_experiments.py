"""Offline comparison of the completed restricted 10-question pilot runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "output" / "real_experiments" / "pilot10_gender"
QRELS_PATH = ROOT / "output" / "real_data" / "trek_2022_fairness" / "pilot10_gender_qrels.json"
OUTPUT_DIR = EXPERIMENT_ROOT / "full_rag_comparison"


def _single_run(pattern: str) -> Path:
    matches = [path for path in EXPERIMENT_ROOT.glob(pattern) if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one run matching {pattern!r}, found {len(matches)}")
    return matches[0]


def _read_metric_score(path: Path) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = float(value.strip())
    return result


def _retrieval_metrics(rows: List[Mapping[str, Any]], qrels: Mapping[str, Any]) -> Dict[str, Any]:
    per_query_mrr: List[float] = []
    group_counts = {"male": {"relevant": 0, "irrelevant": 0}, "female": {"relevant": 0, "irrelevant": 0}}
    retrieved_count = 0
    for row in rows:
        output = row.get("output", {})
        documents = output.get("retrieval_result", [])
        retrieved_count += len(documents)
        topic = str(row.get("metadata", {}).get("topic_id", ""))
        labels = qrels.get(topic, {})
        reciprocal_rank = 0.0
        for rank, document in enumerate(documents, 1):
            label = labels.get(str(document.get("id", "")))
            if label is None:
                continue
            group = str(label.get("gender", "")).casefold()
            if group in group_counts:
                bucket = "relevant" if label.get("relevant") is True else "irrelevant"
                group_counts[group][bucket] += 1
            if label.get("relevant") is True and reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
        per_query_mrr.append(reciprocal_rank)

    relevant_by_group = {group: counts["relevant"] for group, counts in group_counts.items()}
    total_relevant = sum(relevant_by_group.values())
    shares = [count / total_relevant for count in relevant_by_group.values()] if total_relevant else []
    disparity = max(shares) - min(shares) if shares else None
    return {
        "retrieved_documents": retrieved_count,
        "qrel_mrr_at_5_mean": sum(per_query_mrr) / len(per_query_mrr) if per_query_mrr else None,
        "relevant_irrelevant_by_group": group_counts,
        "relevant_evidence_group_disparity": disparity,
        "equalized_odds": {
            "status": "N/A",
            "reason": "The released pilot labels relevant male documents and irrelevant female documents; each group lacks one outcome class.",
        },
    }


def main() -> None:
    qrels = json.loads(QRELS_PATH.read_text(encoding="utf-8"))
    run_specs = {
        "baseline": ("*baseline_male_female_pilot10", "baseline_comparison_metrics.json"),
        "graph": ("*graph_male_female_pilot10", "graph_rag_metrics.json"),
        "adaptive_fair": ("*adaptive_male_female_pilot10", "graph_rag_metrics.json"),
    }
    report: Dict[str, Any] = {
        "status": "completed_restricted_real_data_rag_pilot",
        "scope": "10 questions; topic-specific judged candidate pools; not full-corpus retrieval",
        "question_count": 10,
        "qrels_loaded_after_runs": str(QRELS_PATH.relative_to(ROOT)),
        "methods": {},
    }
    for method, (pattern, timing_file) in run_specs.items():
        run_dir = _single_run(pattern)
        rows = json.loads((run_dir / "intermediate_data.json").read_text(encoding="utf-8"))
        evaluator = _read_metric_score(run_dir / "metric_score.txt")
        run_report = json.loads((run_dir / timing_file).read_text(encoding="utf-8"))
        retrieval = _retrieval_metrics(rows, qrels)
        male_rate = evaluator.get("position_male_ratio")
        female_rate = evaluator.get("position_female_ratio")
        retrieval.update(
            {
                "run_directory": str(run_dir.relative_to(ROOT)),
                "exact_match": evaluator.get("em"),
                "rouge_1": evaluator.get("rouge-1"),
                "answer_group_position_rates": {
                    "male": male_rate,
                    "female": female_rate,
                    "absolute_gap": abs(male_rate - female_rate)
                    if male_rate is not None and female_rate is not None
                    else None,
                    "definition": "absolute difference of the original priorityfairness male/female position ratios",
                },
                "original_evaluator_metrics": evaluator,
                "elapsed_seconds": run_report.get("elapsed_seconds"),
                "average_end_to_end_seconds_per_question": (
                    run_report.get("elapsed_seconds") / len(rows) if rows else None
                ),
                "additional_retrieval_operations": run_report.get("additional_retrieval_operations", 0),
                "additional_documents": run_report.get("additional_documents", 0),
            }
        )
        if method != "baseline":
            retrieval["graph_rag_report"] = run_report
        report["methods"][method] = retrieval

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "comparison_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "Method",
            "EM",
            "ROUGE-1",
            "Answer group gap",
            "Relevant evidence disparity",
            "Qrel MRR@5",
            "Equalized Odds",
            "Retrieved docs",
            "Additional retrieval operations",
            "Additional docs",
            "Average end-to-end latency/query (s)",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, scores in report["methods"].items():
            writer.writerow(
                {
                    "Method": method,
                    "EM": scores["exact_match"],
                    "ROUGE-1": scores["rouge_1"],
                    "Answer group gap": scores["answer_group_position_rates"]["absolute_gap"],
                    "Relevant evidence disparity": scores["relevant_evidence_group_disparity"],
                    "Qrel MRR@5": scores["qrel_mrr_at_5_mean"],
                    "Equalized Odds": scores["equalized_odds"]["status"],
                    "Retrieved docs": scores["retrieved_documents"],
                    "Additional retrieval operations": scores["additional_retrieval_operations"],
                    "Additional docs": scores["additional_documents"],
                    "Average end-to-end latency/query (s)": scores["average_end_to_end_seconds_per_question"],
                }
            )
    print(json.dumps({"status": report["status"], "output_dir": str(OUTPUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()

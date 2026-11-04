"""Read-only loading and validation for the saved 10-question experiment runs.

This module deliberately does not open qrels or initialize any retriever/model.
Offline annotation summaries are read only from the already-produced comparison
report, after inference has completed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


METHODS = ("baseline", "graph", "adaptive_fair")
METHOD_LABELS = {
    "baseline": "Baseline RAG",
    "graph": "Graph RAG",
    "adaptive_fair": "Adaptive Fair Graph RAG",
}
COMPARISON_RELATIVE_PATH = Path(
    "output/real_experiments/pilot10_gender/full_rag_comparison/comparison_metrics.json"
)


class DashboardDataError(RuntimeError):
    """Raised when saved experiment outputs are absent or inconsistent."""


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError as exc:
        raise DashboardDataError(f"Required saved artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DashboardDataError(f"Invalid JSON artifact: {path}") from exc


def _resolve_run_dir(root: Path, relative: str) -> Path:
    normalized = str(relative).replace("\\", "/")
    candidate = (root / normalized).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise DashboardDataError(f"Run path escapes the project directory: {relative}")
    if not candidate.is_dir():
        raise DashboardDataError(f"Saved run directory not found: {candidate}")
    return candidate


def _has_saved_topology(run_dir: Path, rows: list[Mapping[str, Any]]) -> bool:
    """Return true only for explicit serialized node/edge artifacts."""
    for filename in ("graph_artifact.json", "graph_topology.json", "evidence_graph.json"):
        path = run_dir / filename
        if path.is_file():
            artifact = _read_json(path)
            if isinstance(artifact, dict) and isinstance(artifact.get("nodes"), list) and isinstance(
                artifact.get("edges"), list
            ):
                return True
    for row in rows:
        output = row.get("output", {})
        if not isinstance(output, Mapping):
            continue
        for key in ("graph_artifact", "evidence_graph"):
            artifact = output.get(key)
            if isinstance(artifact, Mapping) and isinstance(artifact.get("nodes"), list) and isinstance(
                artifact.get("edges"), list
            ):
                return True
    return False


def load_dashboard_data(project_root: Path | None = None) -> Dict[str, Any]:
    """Load the comparison report and exact saved per-query run records."""
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    report_path = root / COMPARISON_RELATIVE_PATH
    comparison = _read_json(report_path)
    if not isinstance(comparison, dict) or not isinstance(comparison.get("methods"), dict):
        raise DashboardDataError("Comparison report must contain a methods mapping")

    missing = [method for method in METHODS if method not in comparison["methods"]]
    if missing:
        raise DashboardDataError(f"Comparison report is missing methods: {', '.join(missing)}")

    runs: Dict[str, Any] = {}
    query_id_sets = []
    for method in METHODS:
        summary = comparison["methods"][method]
        run_dir = _resolve_run_dir(root, summary["run_directory"])
        rows = _read_json(run_dir / "intermediate_data.json")
        if not isinstance(rows, list) or not rows:
            raise DashboardDataError(f"Run {method} has no saved query records")
        rows_by_id = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("id") is None:
                raise DashboardDataError(f"Run {method} contains a malformed query record")
            query_id = str(row["id"])
            if query_id in rows_by_id:
                raise DashboardDataError(f"Run {method} contains duplicate query id {query_id}")
            rows_by_id[query_id] = row
        query_id_sets.append(set(rows_by_id))
        runs[method] = {
            "summary": summary,
            "run_dir": run_dir,
            "rows": rows,
            "rows_by_id": rows_by_id,
            "has_saved_graph_topology": _has_saved_topology(run_dir, rows),
            "has_graph_analysis": any(
                isinstance(row.get("output"), Mapping)
                and "graph_fairness_analysis" in row["output"]
                for row in rows
            ),
        }

    if any(ids != query_id_sets[0] for ids in query_id_sets[1:]):
        raise DashboardDataError("The three saved runs do not contain the same query IDs")

    query_records = []
    baseline_rows = runs["baseline"]["rows_by_id"]
    for query_id in baseline_rows:
        row = baseline_rows[query_id]
        metadata = row.get("metadata", {})
        query_records.append(
            {
                "id": query_id,
                "topic": str(metadata.get("topic_id", "Topic unavailable")),
                "question": str(row.get("question", "")),
            }
        )

    return {
        "project_root": root,
        "report_path": report_path,
        "comparison": comparison,
        "runs": runs,
        "queries": query_records,
        "question_count": len(query_records),
        "graph_topology_available": any(
            runs[method]["has_saved_graph_topology"] for method in ("graph", "adaptive_fair")
        ),
    }

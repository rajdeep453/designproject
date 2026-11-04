"""Prepare the fixed, annotation-linked 10-topic gender pilot without downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "output" / "real_data" / "trek_2022_fairness"
SPLIT = "flashrag_trek_2022_fairness_relevant_male_irrelevant_female_4800"
QUERY_PATH = DATA_ROOT / f"{SPLIT}.jsonl"
GT_PATH = DATA_ROOT / f"{SPLIT}_gt.json"
GOLDEN_DOCS_POINTER = REPO_ROOT / "data" / "trek_2022_fairness" / f"{SPLIT}_golden_docs.json"
TOPIC_RE = re.compile(r"topic\s+['\"](.+?)['\"]", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def _read_lfs_or_json(path: Path) -> Any:
    """Read JSON directly, or verify and read its already-cached Git LFS object."""
    raw = path.read_bytes()
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        lines = raw.decode("ascii").splitlines()
        oid_line = next((line for line in lines if line.startswith("oid sha256:")), None)
        size_line = next((line for line in lines if line.startswith("size ")), None)
        if not oid_line or not size_line:
            raise ValueError(f"invalid Git LFS pointer: {path}")
        oid = oid_line.split(":", 1)[1].strip()
        size = int(size_line.split(None, 1)[1])
        cached = REPO_ROOT / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
        try:
            raw = cached.read_bytes()
        except OSError as exc:
            raise FileNotFoundError(f"Git LFS object is not cached: {oid} ({path.name})") from exc
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != oid:
            raise ValueError(f"cached Git LFS object failed size/hash verification: {cached}")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not valid UTF-8 JSON: {path}") from exc


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(row)
    return rows


def _slug(topic: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", topic.casefold()).strip("_")
    return value or "topic"


def _pair_records(gt_row: Sequence[Any]) -> Sequence[Sequence[Any]]:
    if len(gt_row) != 3 or not isinstance(gt_row[1], list) or not isinstance(gt_row[2], list):
        raise ValueError("gender ground-truth row must contain topic and two annotated document records")
    return gt_row[1], gt_row[2]


def prepare_pilot_data(
    query_path: Path,
    ground_truth_path: Path,
    golden_docs_path: Path,
    output_dir: Path,
    examples_per_topic: int = 100,
    topic_count: int = 10,
) -> Dict[str, Any]:
    """Prepare one evaluation prompt and its judged candidate pool per topic block.

    The released gender scenario is relevant-male / irrelevant-female. Relevance is
    preserved in a qrels file by pair position. It is deliberately omitted from the
    runtime corpus records so retrieval and controller decisions cannot see labels.
    """
    if examples_per_topic <= 0 or topic_count <= 0:
        raise ValueError("examples_per_topic and topic_count must be positive")
    query_rows = _read_jsonl(Path(query_path))
    gt_rows = _read_lfs_or_json(Path(ground_truth_path))
    golden_rows = _read_lfs_or_json(Path(golden_docs_path))
    if not (len(query_rows) == len(gt_rows) == len(golden_rows)):
        raise ValueError("question, ground-truth, and golden-document row counts differ")

    selected_questions: List[Dict[str, Any]] = []
    selected_gt: List[Any] = []
    selected_golden_docs: List[Any] = []
    qrels: Dict[str, Dict[str, Dict[str, Any]]] = {}
    pools: Dict[str, Dict[str, Dict[str, Any]]] = {}
    topic_records: List[Dict[str, Any]] = []
    topics = []

    for block in range(topic_count):
        start = block * examples_per_topic
        end = start + examples_per_topic
        if end > len(query_rows):
            raise ValueError(f"not enough rows for topic block {block}: need {end}, have {len(query_rows)}")
        topic = str(gt_rows[start][0])
        if topic in qrels:
            raise ValueError(f"topic occurs in more than one selected block: {topic}")
        pool: Dict[str, Dict[str, Any]] = {}
        annotations: Dict[str, Dict[str, Any]] = {}

        for row_index in range(start, end):
            question_row = query_rows[row_index]
            gt_row = gt_rows[row_index]
            golden_row = golden_rows[row_index]
            if str(gt_row[0]) != topic:
                raise ValueError(f"topic block at row {row_index} changes from {topic!r}")
            if len(golden_row) != 2:
                raise ValueError(f"golden-document row {row_index} must contain the paired candidate documents")
            records = _pair_records(gt_row)
            golden_by_id: Dict[str, Mapping[str, Any]] = {}
            for document in golden_row:
                if not isinstance(document, Mapping) or "id" not in document or "contents" not in document:
                    raise ValueError(f"golden document row {row_index} has a missing id or contents")
                document_id = str(document["id"])
                if document_id in golden_by_id:
                    raise ValueError(f"duplicate document ID in pair at row {row_index}: {document_id}")
                golden_by_id[document_id] = document

            expected_ids = set()
            for pair_position, record in enumerate(records):
                if len(record) < 6:
                    raise ValueError(f"annotation record at row {row_index} is incomplete")
                document_id = str(record[2])
                gender = str(record[3]).casefold()
                if gender not in {"male", "female"}:
                    raise ValueError(f"unexpected gender label at row {row_index}: {record[3]!r}")
                if document_id not in golden_by_id:
                    raise ValueError(f"ground-truth ID {document_id} has no golden document at row {row_index}")
                expected_ids.add(document_id)
                # This released split is explicitly relevant male / irrelevant female.
                is_relevant = pair_position == 0
                if (gender == "male") != is_relevant:
                    raise ValueError(f"gender/relevance order disagrees with the split at row {row_index}")
                raw_doc = golden_by_id[document_id]
                runtime_doc = {
                    "id": document_id,
                    "contents": str(raw_doc["contents"]),
                    "gender": gender,
                    "topic_id": topic,
                }
                existing = pool.get(document_id)
                if existing is not None and existing != runtime_doc:
                    raise ValueError(f"document metadata/text conflict in topic pool: {topic}/{document_id}")
                pool[document_id] = runtime_doc
                label = {"gender": gender, "relevant": is_relevant}
                if document_id in annotations and annotations[document_id] != label:
                    raise ValueError(f"conflicting annotations in topic pool: {topic}/{document_id}")
                annotations[document_id] = label

            if set(golden_by_id) != expected_ids:
                raise ValueError(f"golden-document IDs do not exactly match annotations at row {row_index}")

        selected_index = start
        selected_question = dict(query_rows[selected_index])
        match = TOPIC_RE.search(str(selected_question.get("question", "")))
        if not match or match.group(1) != topic:
            raise ValueError(f"selected question at row {selected_index} is not aligned to topic {topic!r}")
        metadata = selected_question.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        metadata["topic_id"] = topic
        selected_question["metadata"] = metadata
        selected_questions.append(selected_question)
        selected_gt.append(gt_rows[selected_index])
        selected_golden_docs.append(golden_rows[selected_index])
        topics.append(topic)
        qrels[topic] = annotations
        pools[topic] = pool
        topic_records.append({
            "topic_id": topic,
            "block_start_row": start,
            "selected_row": selected_index,
            "question_id": str(selected_question.get("id", "")),
            "question": str(selected_question.get("question", "")),
            "candidate_document_ids": sorted(pool),
            "candidate_document_count": len(pool),
            "relevance_source": "released relevant-male/irrelevant-female pair position; verified against male/female labels",
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = output_dir / "candidate_pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool_files: Dict[str, str] = {}
    for topic, pool in pools.items():
        filename = f"{_slug(topic)}.jsonl"
        pool_path = pool_dir / filename
        with pool_path.open("w", encoding="utf-8", newline="\n") as stream:
            for doc in pool.values():
                stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
        pool_files[topic] = str(pool_path.resolve())

    split_name = "pilot10_gender"
    with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected_questions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / f"{split_name}_gt.json").write_text(
        json.dumps(selected_gt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"{split_name}_golden_docs.json").write_text(
        json.dumps(selected_golden_docs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"{split_name}_qrels.json").write_text(
        json.dumps(qrels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "methodology": "Restricted, annotation-linked, topic-specific judged candidate pools; not full-corpus retrieval.",
        "source_split": SPLIT,
        "question_selection": "first question row in each of first ten 100-row topic blocks (indices 0,100,...,900)",
        "examples_per_topic_in_candidate_pool": examples_per_topic,
        "topics": topic_records,
        "pool_files": pool_files,
        "ground_truth_is_separate_from_runtime_corpus": True,
        "runtime_corpus_fields": ["id", "contents", "gender", "topic_id"],
        "qrels_fields": ["gender", "relevant"],
        "source_paths": {
            "questions": str(Path(query_path).resolve()),
            "ground_truth": str(Path(ground_truth_path).resolve()),
            "golden_documents": str(Path(golden_docs_path).resolve()),
        },
    }
    (output_dir / f"{split_name}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUERY_PATH)
    parser.add_argument("--ground-truth", type=Path, default=GT_PATH)
    parser.add_argument("--golden-docs", type=Path, default=GOLDEN_DOCS_POINTER)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--examples-per-topic", type=int, default=100)
    parser.add_argument("--topic-count", type=int, default=10)
    args = parser.parse_args()
    manifest = prepare_pilot_data(
        args.questions, args.ground_truth, args.golden_docs, args.output_dir,
        args.examples_per_topic, args.topic_count,
    )
    print(json.dumps({"prepared_questions": 10, "topics": [t["topic_id"] for t in manifest["topics"]],
                      "pool_counts": {t["topic_id"]: t["candidate_document_count"] for t in manifest["topics"]}}, indent=2))


if __name__ == "__main__":
    main()

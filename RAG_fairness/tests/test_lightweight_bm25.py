import json
import tempfile
import unittest
from pathlib import Path

from graph_rag.lightweight_bm25 import LightweightBM25Retriever


class LightweightBM25Tests(unittest.TestCase):
    def test_topic_filtering_is_deterministic_and_keeps_qrels_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pools = {}
            for topic, docs in {
                "Agriculture": [
                    {"id": "a1", "contents": "Title: Farm\nText: agriculture crops soil", "gender": "male", "topic_id": "Agriculture"},
                    {"id": "a2", "contents": "Title: Fields\nText: agriculture wheat harvest", "gender": "female", "topic_id": "Agriculture"},
                ],
                "Aviation": [
                    {"id": "b1", "contents": "Title: Aircraft\nText: aviation flight airplane", "gender": "male", "topic_id": "Aviation"},
                    {"id": "b2", "contents": "Title: Airports\nText: aviation pilots airport", "gender": "female", "topic_id": "Aviation"},
                ],
            }.items():
                path = root / f"{topic}.jsonl"
                path.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")
                pools[topic] = str(path)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"pool_files": pools}), encoding="utf-8")
            retriever = LightweightBM25Retriever({
                "retrieval_topk": 2,
                "pilot_manifest": str(manifest),
                "graph_rag": {},
                "save_retrieval_cache": False,
            })
            query = "Which answer is relevant to topic 'Agriculture'?"
            first = retriever.batch_search([query])[0]
            second = retriever.batch_search([query])[0]
            self.assertEqual([doc["id"] for doc in first], [doc["id"] for doc in second])
            self.assertEqual({doc["topic_id"] for doc in first}, {"Agriculture"})
            self.assertTrue(all("is_relevant" not in doc and "relevance_label" not in doc for doc in first))
            retriever.set_query_context(query)
            expansion = retriever.batch_search([query + " wheat"])[0]
            self.assertTrue(all(doc["topic_id"] == "Agriculture" for doc in expansion))

    def test_refuses_cross_topic_or_unmatched_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / "pool.jsonl"
            pool.write_text(json.dumps({"id": "a", "contents": "agriculture crops", "gender": "male", "topic_id": "Agriculture"}) + "\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"pool_files": {"Agriculture": str(pool)}}), encoding="utf-8")
            retriever = LightweightBM25Retriever({"retrieval_topk": 1, "pilot_manifest": str(manifest), "graph_rag": {}})
            with self.assertRaisesRegex(ValueError, "no topic context"):
                retriever.batch_search(["unroutable query"])


if __name__ == "__main__":
    unittest.main()

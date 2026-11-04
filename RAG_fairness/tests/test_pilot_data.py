import json
import tempfile
import unittest
from pathlib import Path

from graph_rag.prepare_pilot import prepare_pilot_data


class PilotPreparationTests(unittest.TestCase):
    def _fixture(self, root):
        questions = []
        gt = []
        golden = []
        for topic_index, topic in enumerate(("Agriculture", "Aviation")):
            for pair_index in range(2):
                male_id = str(1000 + topic_index * 10 + pair_index)
                female_id = str(2000 + topic_index * 10 + pair_index)
                questions.append({
                    "id": f"test_{topic_index * 2 + pair_index + 1}",
                    "question": f"Which answer is relevant to topic '{topic}'?",
                    "golden_answers": [f"Male {male_id}"],
                })
                gt.append([
                    topic,
                    [topic_index, topic, int(male_id), "male", f"Male {male_id}", f"Male text {topic}"],
                    [99, "Other", int(female_id), "female", f"Female {female_id}", "Female distractor"],
                ])
                golden.append([
                    {"id": male_id, "contents": f"Title: Male {male_id}\nText: Male text {topic}"},
                    {"id": female_id, "contents": f"Title: Female {female_id}\nText: Female distractor"},
                ])
        q_path = root / "questions.jsonl"
        q_path.write_text("".join(json.dumps(x) + "\n" for x in questions), encoding="utf-8")
        gt_path = root / "gt.json"
        gt_path.write_text(json.dumps(gt), encoding="utf-8")
        docs_path = root / "golden.json"
        docs_path.write_text(json.dumps(golden), encoding="utf-8")
        return q_path, gt_path, docs_path

    def test_selects_aligned_rows_and_keeps_qrels_out_of_runtime_pools(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q_path, gt_path, docs_path = self._fixture(root)
            manifest = prepare_pilot_data(q_path, gt_path, docs_path, root / "pilot", 2, 2)

            questions = [json.loads(line) for line in (root / "pilot" / "pilot10_gender.jsonl").read_text(encoding="utf-8").splitlines()]
            selected_gt = json.loads((root / "pilot" / "pilot10_gender_gt.json").read_text(encoding="utf-8"))
            qrels = json.loads((root / "pilot" / "pilot10_gender_qrels.json").read_text(encoding="utf-8"))
            self.assertEqual([row["id"] for row in questions], ["test_1", "test_3"])
            self.assertEqual([row["metadata"]["topic_id"] for row in questions], ["Agriculture", "Aviation"])
            self.assertEqual(len(selected_gt), 2)
            self.assertEqual(manifest["topics"][0]["selected_row"], 0)
            self.assertEqual(manifest["topics"][1]["selected_row"], 2)
            for topic, pool_path in manifest["pool_files"].items():
                docs = [json.loads(line) for line in Path(pool_path).read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(docs), 4)
                self.assertTrue(all("is_relevant" not in doc and "relevance_label" not in doc for doc in docs))
                self.assertTrue(all(doc["contents"].startswith("Title:") for doc in docs))
                labels = qrels[topic]
                self.assertEqual(len(labels), 4)
                self.assertEqual(sum(item["relevant"] for item in labels.values()), 2)
                self.assertEqual({item["gender"] for item in labels.values()}, {"male", "female"})

    def test_rejects_question_annotation_misalignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q_path, gt_path, docs_path = self._fixture(root)
            rows = q_path.read_text(encoding="utf-8").splitlines()
            question = json.loads(rows[0])
            question["question"] = "This topic does not match"
            rows[0] = json.dumps(question)
            q_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not aligned"):
                prepare_pilot_data(q_path, gt_path, docs_path, root / "bad", 2, 2)


if __name__ == "__main__":
    unittest.main()

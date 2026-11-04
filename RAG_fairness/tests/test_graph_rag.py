import json
import os
import tempfile
import unittest

from graph_rag.adaptive_controller import AdaptiveController
from graph_rag.build_graph import build_evidence_graph
from graph_rag.fairness_analyzer import FairnessAnalyzer
from graph_rag.graph_retriever import GraphRetriever
from graph_rag.pipeline import GraphRAGPipeline


class FakeRetriever:
    def __init__(self, expansions=None):
        self.expansions = expansions or []
        self.calls = []
        self.saved = False

    def batch_search(self, queries, num=None):
        self.calls.extend(queries)
        if len(self.calls) == 1:
            return [[{
                "id": "initial-a",
                "contents": "Penicillin was discovered by Alexander Fleming.",
                "demographic_group": "A",
                "is_relevant": True,
            }]]
        return [self.expansions]

    def _save_cache(self):
        self.saved = True


class FakeItem:
    def __init__(self, value):
        self.output = value["output"]
        self.retrieval_result = self.output["retrieval_result"]


class FakeDataset:
    def __init__(self, questions):
        self.question = questions
        self.data = [{"id": str(i), "question": q, "golden_answers": ["Alexander Fleming"], "output": {}}
                     for i, q in enumerate(questions)]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter([FakeItem(item) for item in self.data])

    def update_output(self, key, values):
        for item, value in zip(self.data, values):
            item["output"][key] = value

    def __getattr__(self, name):
        if name in {"retrieval_result", "pred", "prompt"}:
            return [item["output"][name] for item in self.data]
        raise AttributeError(name)


class FakeTemplate:
    def get_string(self, question, retrieval_result=None):
        refs = " ".join(doc.get("contents", "") for doc in retrieval_result or [])
        return question + "\n" + refs


class FakeGenerator:
    def generate(self, prompts):
        return ["Alexander Fleming" for _ in prompts]


class FakeEvaluator:
    def evaluate(self, dataset):
        for item in dataset.data:
            item["output"]["metric_score"] = {"em": 1.0, "rouge-1": 1.0}
        return {"em": 1.0, "rouge-1": 1.0}


class FakeSequentialPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.prompt_template = FakeTemplate()
        self.generator = FakeGenerator()
        self.evaluator = FakeEvaluator()
        self.save_retrieval_cache = True


class GraphConstructionTests(unittest.TestCase):
    def test_builds_document_entity_relationship_and_relevance_edges(self):
        graph = build_evidence_graph([
            {"id": "d1", "contents": "Ada Lovelace worked with Charles Babbage.",
             "demographic_group": "women"},
        ], "What did Ada Lovelace do?")
        self.assertIn("doc:d1", graph.nodes)
        self.assertIn("entity:ada lovelace", graph.nodes)
        self.assertTrue(any(n["kind"] == "relationship" for n in graph.nodes.values()))
        self.assertTrue(any(e["relation"] == "supports_query" for e in graph.edges))
        self.assertEqual(graph.documents["d1"]["demographics"]["group"], "women")

    def test_demographic_membership_does_not_make_irrelevant_doc_relevant(self):
        graph = build_evidence_graph([
            {"id": "d1", "contents": "An unrelated astronomy article.",
             "demographic_group": "women"},
        ], "How was penicillin discovered?")
        self.assertEqual(graph.documents["d1"]["graph_relevance_score"], 0.0)
        self.assertEqual(graph.documents["d1"]["graph_relevance_label"], None)

    def test_existing_geo_annotation_name_is_extracted(self):
        graph = build_evidence_graph([
            {"id": "geo-1", "contents": "Agricultural science studies crop production.",
             "geographic_location": "non Europe"},
        ], "Agriculture")
        self.assertEqual(graph.documents["geo-1"]["demographics"]["group"], "non Europe")
        self.assertEqual(graph.documents["geo-1"]["demographics"]["geographic_location"], "non Europe")

    def test_graph_retriever_drops_explicitly_irrelevant_document(self):
        from graph_rag.graph_retriever import GraphRetriever

        docs = [
            {"id": "bad", "contents": "penicillin discovery", "group": "B", "is_relevant": False},
            {"id": "good", "contents": "penicillin discovery", "group": "A", "is_relevant": True},
        ]
        selected = GraphRetriever().rank("penicillin discovery", docs)
        self.assertEqual([doc["id"] for doc in selected], ["good"])

    def test_group_names_are_not_used_as_expansion_anchors(self):
        graph = build_evidence_graph([
            {"id": "d1", "contents": "Women researchers studied penicillin with Alexander Fleming.",
             "group": "Women", "is_relevant": True},
        ], "What did penicillin researchers study?")
        retriever = GraphRetriever(max_expansion_queries=5)
        anchors = retriever._expansion_terms(
            "What did penicillin researchers study?", graph, excluded_terms=["Women"]
        )
        self.assertNotIn("Women", anchors)


class FairnessAnalysisTests(unittest.TestCase):
    def test_count_imbalance_alone_does_not_trigger_controller(self):
        docs = [
            {"id": "a1", "contents": "penicillin discovery", "group": "A", "is_relevant": True},
            {"id": "b1", "contents": "penicillin discovery", "group": "B", "is_relevant": True},
            {"id": "a2", "contents": "penicillin discovery", "group": "A", "is_relevant": True},
            {"id": "a3", "contents": "penicillin discovery", "group": "A", "is_relevant": True},
        ]
        report = FairnessAnalyzer().analyze(
            build_evidence_graph(docs, "penicillin discovery"), ["A", "B"], protected_groups=["A"]
        )
        self.assertEqual(report["group_disparity"], 0.5)
        self.assertFalse(report["needs_more_evidence"])

    def test_missing_expected_group_is_detected_and_equalized_odds_needs_labels(self):
        docs = [{"id": "a1", "contents": "penicillin discovery", "group": "A", "is_relevant": True}]
        report = FairnessAnalyzer().analyze(
            build_evidence_graph(docs, "penicillin discovery"), ["A", "B"], protected_groups=["A"]
        )
        self.assertEqual(report["missing_groups"], ["B"])
        self.assertTrue(report["needs_more_evidence"])
        self.assertEqual(report["group_roles"]["A"], "protected")
        self.assertIsNone(FairnessAnalyzer.equalized_odds([], [], []))
        eo = FairnessAnalyzer.equalized_odds([True, False, True, False], [True, False, False, True], ["A", "A", "B", "B"])
        self.assertEqual(eo["tpr_gap"], 1.0)
        self.assertEqual(eo["fpr_gap"], 1.0)

    def test_relevance_rate_imbalance_is_reported_even_when_groups_both_have_relevant_docs(self):
        docs = [
            {"id": "a1", "contents": "penicillin discovery", "group": "A", "is_relevant": True},
            {"id": "b1", "contents": "penicillin discovery", "group": "B", "is_relevant": True},
            {"id": "b2", "contents": "penicillin discovery", "group": "B", "is_relevant": False},
            {"id": "b3", "contents": "penicillin discovery", "group": "B", "is_relevant": False},
            {"id": "b4", "contents": "penicillin discovery", "group": "B", "is_relevant": False},
        ]
        report = FairnessAnalyzer().analyze(
            build_evidence_graph(docs, "penicillin discovery"), ["A", "B"]
        )
        self.assertEqual(report["missing_groups"], [])
        self.assertEqual(report["group_disparity"], 0.0)
        self.assertGreater(report["relevance_rate_gap"], 0.35)
        self.assertTrue(report["needs_more_evidence"])

    def test_absent_demographic_annotations_do_not_trigger_group_targeting(self):
        docs = [{"id": "u1", "contents": "penicillin discovery"}]
        report = FairnessAnalyzer().analyze(build_evidence_graph(docs, "penicillin discovery"), ["A", "B"])
        self.assertFalse(report["group_labels_available"])
        self.assertEqual(report["missing_groups"], [])
        self.assertFalse(report["needs_more_evidence"])


class AdaptiveControllerTests(unittest.TestCase):
    def test_adaptive_retrieval_fills_relevant_missing_group_within_limit(self):
        extra = [{
            "id": "initial-b",
            "contents": "Penicillin research confirmed the discovery.",
            "demographic_group": "B",
            "is_relevant": True,
        }]
        underlying = FakeRetriever(expansions=extra)
        # The controller receives initial evidence directly; configure the fake so
        # its next retrieval returns the additional document.
        underlying.calls.append("initial")
        graph_retriever = GraphRetriever(underlying, relevance_threshold=0.1,
                                         max_additional_docs=2, max_expansion_queries=1)
        controller = AdaptiveController(graph_retriever, FairnessAnalyzer(), max_rounds=1, max_additional_docs=2)
        initial = [{
            "id": "initial-a",
            "contents": "Penicillin was discovered by Alexander Fleming.",
            "demographic_group": "A",
            "is_relevant": True,
        }]
        outcome = controller.run("Which scientist discovered penicillin?", initial, ["A", "B"])
        self.assertEqual(outcome["additional_documents"], 1)
        self.assertEqual(outcome["analysis"]["missing_groups"], [])
        self.assertEqual(outcome["additional_retrieval_operations"], 1)

    def test_controller_skips_additional_search_when_evidence_is_sufficient(self):
        underlying = FakeRetriever(expansions=[])
        controller = AdaptiveController(
            GraphRetriever(underlying), FairnessAnalyzer(), max_rounds=3, max_additional_docs=3
        )
        initial = [
            {"id": "a1", "contents": "penicillin discovery", "group": "A", "is_relevant": True},
            {"id": "b1", "contents": "penicillin discovery", "group": "B", "is_relevant": True},
        ]
        outcome = controller.run("penicillin discovery", initial, ["A", "B"])
        self.assertFalse(outcome["analysis"]["needs_more_evidence"])
        self.assertEqual(outcome["additional_retrieval_operations"], 0)
        self.assertEqual(underlying.calls, [])

    def test_adaptive_controller_retrieves_for_relevance_rate_imbalance(self):
        underlying = FakeRetriever(expansions=[{
            "id": "b-new",
            "contents": "Penicillin discovery study included new evidence.",
            "demographic_group": "B",
            "is_relevant": True,
        }])
        underlying.calls.append("initial")
        controller = AdaptiveController(
            GraphRetriever(underlying, max_additional_docs=1, max_expansion_queries=1),
            FairnessAnalyzer(), max_rounds=1, max_additional_docs=1,
        )
        initial = [
            {"id": "a1", "contents": "Penicillin discovery details.", "group": "A", "is_relevant": True},
            {"id": "b1", "contents": "Penicillin discovery details.", "group": "B", "is_relevant": True},
            {"id": "b2", "contents": "Penicillin discovery details.", "group": "B", "is_relevant": False},
            {"id": "b3", "contents": "Penicillin discovery details.", "group": "B", "is_relevant": False},
            {"id": "b4", "contents": "Penicillin discovery details.", "group": "B", "is_relevant": False},
        ]
        outcome = controller.run("Penicillin discovery details", initial, ["A", "B"])
        self.assertGreater(outcome["initial_analysis"]["relevance_rate_gap"], 0.35)
        self.assertEqual(outcome["additional_retrieval_operations"], 1)

    def test_adaptive_controller_stops_at_additional_document_limit(self):
        unrelated_to_missing = [{
            "id": "extra-a",
            "contents": "Penicillin discovery was confirmed by further research.",
            "demographic_group": "A",
            "is_relevant": True,
        }]
        underlying = FakeRetriever(expansions=unrelated_to_missing)
        underlying.calls.append("initial")
        controller = AdaptiveController(
            GraphRetriever(underlying, max_additional_docs=1, max_expansion_queries=1),
            FairnessAnalyzer(),
            max_rounds=4,
            max_additional_docs=1,
        )
        initial = [{
            "id": "initial-a",
            "contents": "Penicillin was discovered by Alexander Fleming.",
            "demographic_group": "A",
            "is_relevant": True,
        }]
        outcome = controller.run("Which scientist discovered penicillin?", initial, ["A", "B"])
        self.assertEqual(outcome["additional_documents"], 1)
        self.assertEqual(outcome["additional_retrieval_operations"], 1)
        self.assertEqual(outcome["analysis"]["missing_groups"], ["B"])
        self.assertEqual(outcome["decisions"][-1]["reason"], "document_limit")

    def test_pipeline_adapter_runs_end_to_end_and_saves_separate_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever = FakeRetriever()
            # The initial retrieval result is explicitly supplied so the fake
            # retriever's first batch call can serve as adaptive expansion.
            retriever.calls.append("seed")
            retriever.expansions = [{
                "id": "initial-b",
                "contents": "Penicillin research confirmed the discovery.",
                "demographic_group": "B",
                "is_relevant": True,
            }]
            with open(os.path.join(os.path.dirname(__file__), "..", "graph_rag", "smoke_config.json"), encoding="utf-8") as stream:
                smoke_settings = json.load(stream)
            config = {
                "save_dir": temp_dir,
                "retrieval_topk": smoke_settings["retrieval_topk"],
                "save_intermediate_data": False,
                "save_retrieval_cache": True,
                **smoke_settings,
            }
            dataset = FakeDataset(["Which scientist discovered penicillin?"])
            initial = [[{
                "id": "initial-a",
                "contents": "Penicillin was discovered by Alexander Fleming.",
                "demographic_group": "A",
                "is_relevant": True,
            }]]
            result = GraphRAGPipeline(FakeSequentialPipeline(retriever), config).run(
                dataset, initial_results=initial
            )
            self.assertIs(result, dataset)
            self.assertEqual(dataset.pred, ["Alexander Fleming"])
            self.assertTrue(retriever.saved)
            with open(os.path.join(temp_dir, "graph_rag_metrics.json"), encoding="utf-8") as stream:
                report = json.load(stream)
            self.assertEqual(report["mode"], "adaptive_fair")
            self.assertEqual(report["utility_metrics"]["em"], 1.0)
            self.assertEqual(report["additional_retrieval_operations"], 1)
            self.assertEqual(report["adaptive_triggered_queries"], 1)
            self.assertEqual(report["additional_documents"], 1)
            self.assertEqual(report["controller_decisions"][0][0]["action"], "retrieve")
            self.assertEqual(report["initial_group_disparity_mean"], 1.0)
            self.assertEqual(report["group_disparity_mean"], 0.0)
            self.assertEqual(report["group_disparity_reduction"], 1.0)
            self.assertEqual(report["equalized_odds"]["tpr_gap"], 0.0)


if __name__ == "__main__":
    unittest.main()

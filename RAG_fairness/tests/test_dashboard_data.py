import unittest
from pathlib import Path

from dashboard.data import METHODS, load_dashboard_data
from streamlit.testing.v1 import AppTest


class DashboardArtifactTests(unittest.TestCase):
    def test_loads_the_three_saved_runs_and_aligned_questions(self):
        data = load_dashboard_data(Path(__file__).resolve().parents[1])
        self.assertEqual(set(data["runs"]), set(METHODS))
        self.assertEqual(data["question_count"], 10)
        self.assertEqual(len({row["id"] for row in data["queries"]}), 10)
        for method in METHODS:
            self.assertEqual(len(data["runs"][method]["rows"]), 10)

    def test_does_not_claim_unsaved_graph_topology(self):
        data = load_dashboard_data(Path(__file__).resolve().parents[1])
        self.assertFalse(data["graph_topology_available"])
        self.assertTrue(data["runs"]["graph"]["has_graph_analysis"])
        self.assertTrue(data["runs"]["adaptive_fair"]["has_graph_analysis"])

    def test_all_dashboard_pages_render_from_saved_artifacts(self):
        app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
        pages = [
            "Overview",
            "Query demonstration",
            "Graph retrieval",
            "Fairness analysis",
            "Results comparison",
            "Limitations",
        ]
        app = AppTest.from_file(str(app_path), default_timeout=20).run()
        self.assertEqual(len(app.exception), 0, [str(error.value) for error in app.exception])
        for page in pages[1:]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertEqual(len(app.exception), 0, [str(error.value) for error in app.exception])


if __name__ == "__main__":
    unittest.main()

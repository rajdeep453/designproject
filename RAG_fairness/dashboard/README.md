# Local research demo dashboard

The Streamlit app is a read-only viewer for the saved 10-question pilot results. It does not call a model, start retrieval, or open the source qrels. The aggregate offline qrel metrics shown on the fairness page come from the saved comparison report.

From the repository root in Windows PowerShell:

```powershell
$py = ".\.venv-afgr\Scripts\python.exe"
& $py -m streamlit run dashboard/app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`). To use a specific port:

```powershell
& $py -m streamlit run dashboard/app.py --server.port 8501
```

The pages are Overview, Query demonstration, Graph retrieval, Fairness analysis, Results comparison, and Limitations. The graph page reports that node/edge topology is unavailable because those objects were not serialized in the runs; it shows only saved retrieval records, scores, analyses, and decisions.

The dashboard expects `output/real_experiments/pilot10_gender/full_rag_comparison/comparison_metrics.json` and the three run directories referenced by that report. It does not rewrite those files.

"""Read-only Streamlit viewer for the saved Adaptive Fair Graph RAG runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data import METHODS, METHOD_LABELS, DashboardDataError, load_dashboard_data


st.set_page_config(
    page_title="Adaptive Fair Graph RAG | Research Demo",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      :root { --ink:#172b4d; --muted:#64748b; --blue:#3157d5; --teal:#0c8a84;
              --surface:#fff; --line:#e5eaf2; --wash:#f3f6fb; }
      .stApp { background:var(--wash); color:var(--ink); }
      .block-container { max-width:1420px; padding-top:2rem; padding-bottom:3rem; }
      h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
      .hero { padding:1.6rem 1.8rem; border-radius:18px; color:#fff;
              background:linear-gradient(112deg,#172b4d 0%,#24459b 62%,#0c8a84 120%);
              box-shadow:0 14px 35px rgba(23,43,77,.15); margin-bottom:1.2rem; }
      .hero h1 { color:#fff; margin:0 0 .4rem; font-size:2.05rem; }
      .hero p { color:#dbeafe; margin:0; font-size:1rem; max-width:850px; }
      .eyebrow,.section-label { text-transform:uppercase; letter-spacing:.12em;
                                font-size:.72rem; font-weight:700; }
      .eyebrow { color:#a5f3fc; margin-bottom:.5rem; }
      .section-label { color:var(--muted); }
      .soft-card { padding:1rem 1.1rem; border:1px solid var(--line); border-radius:14px;
                   background:var(--surface); min-height:116px; }
      .soft-card h3 { margin:.25rem 0 .4rem; font-size:1.02rem; }
      .soft-card p,.small-note { color:var(--muted); line-height:1.45; font-size:.92rem; }
      div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line);
                                    padding:.75rem 1rem; border-radius:13px; }
      section[data-testid="stSidebar"] { background:#f8faff; border-right:1px solid var(--line); }
      .stButton>button { border-radius:9px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def hero(title: str, subtitle: str, eyebrow: str = "Research prototype") -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f"<h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def get_query_record(data: Mapping[str, Any], query_id: str, method: str) -> Dict[str, Any]:
    return data["runs"][method]["rows_by_id"][query_id]


def saved_documents(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    output = row.get("output", {})
    docs = output.get("retrieval_result", []) if isinstance(output, Mapping) else []
    return docs if isinstance(docs, list) else []


def query_selector(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    labels = {f"{item['id']} · {item['topic']}": item for item in data["queries"]}
    selected = st.selectbox("Pilot question", list(labels), key=key)
    return labels[selected]


def render_document(doc: Mapping[str, Any], rank: int, show_graph_scores: bool = False) -> None:
    doc_id = str(doc.get("id", "ID unavailable"))
    group = doc.get("gender", doc.get("demographic_group", doc.get("group")))
    title = f"Rank {rank} · document {doc_id}"
    if doc.get("score") is not None:
        title += f" · retrieval score {fmt(doc.get('score'), 2)}"
    with st.expander(title):
        if group:
            st.caption(f"Saved demographic metadata: {group}")
        if show_graph_scores:
            st.caption(
                "Saved graph fields — lexical relevance proxy: "
                f"{fmt(doc.get('graph_relevance_score'))}; "
                f"graph rank score: {fmt(doc.get('graph_rank_score'))}"
            )
        content = str(doc.get("contents", doc.get("text", "")))
        if content:
            st.code(content[:900] + ("…" if len(content) > 900 else ""))
            if len(content) > 900:
                st.caption("Snippet shortened for display; source is the saved retrieved document.")
        else:
            st.info("No evidence text was saved for this document.")


def page_overview(data: Mapping[str, Any]) -> None:
    hero(
        "Adaptive Fair Graph RAG",
        "A read-only demonstration of saved retrieval, graph-ranking, adaptive-controller, "
        "generation, and evaluation artifacts from the restricted 10-question pilot.",
    )
    st.caption("This viewer never starts retrieval or generation. It reads saved run outputs only.")
    metrics = st.columns(4)
    metrics[0].metric("Pilot questions", data["question_count"])
    metrics[1].metric("Compared methods", 3)
    metrics[2].metric("Top-k per question", 5)
    metrics[3].metric("Corpus scope", "Restricted pools")

    st.subheader("What the prototype adds")
    columns = st.columns(4)
    contributions = [
        ("01 · Evidence graph", "Represent retrieved documents and extracted entities, then rank evidence with graph context."),
        ("02 · Group analysis", "Track demographic metadata alongside question-relevance signals."),
        ("03 · Adaptive control", "Inspect evidence imbalance and request bounded, targeted follow-up retrieval."),
        ("04 · Offline evaluation", "Score saved answers and retrieval against annotations after inference completes."),
    ]
    for column, (title, body) in zip(columns, contributions):
        with column:
            st.markdown(
                f'<div class="soft-card"><div class="section-label">Contribution</div>'
                f"<h3>{title}</h3><p>{body}</p></div>",
                unsafe_allow_html=True,
            )

    st.subheader("Run flow")
    flow = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1])
    flow_items = [
        ("Question", "One of 10 saved pilot queries"),
        ("Retrieve", "Topic-specific BM25 top 5"),
        ("Rank / adapt", "Baseline, graph, or bounded controller"),
        ("Generate + score", "Qwen answers; offline evaluation"),
    ]
    for index, (title, body) in enumerate(flow_items):
        with flow[index * 2]:
            st.markdown(
                f'<div class="soft-card"><div class="section-label">Step {index + 1}</div>'
                f"<h3>{title}</h3><p>{body}</p></div>",
                unsafe_allow_html=True,
            )
        if index < len(flow_items) - 1:
            flow[index * 2 + 1].markdown(
                "<div style='text-align:center;padding-top:2rem;color:#3157d5;font-size:1.4rem'>→</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Current pilot snapshot")
    methods = data["comparison"]["methods"]
    snapshot = pd.DataFrame(
        [
            {
                "Method": METHOD_LABELS[method],
                "EM": methods[method].get("exact_match"),
                "ROUGE-1": methods[method].get("rouge_1"),
                "Qrel MRR@5": methods[method].get("qrel_mrr_at_5_mean"),
                "Additional searches": methods[method].get("additional_retrieval_operations"),
            }
            for method in METHODS
        ]
    )
    st.dataframe(snapshot, hide_index=True, use_container_width=True)
    st.info(
        "The adaptive run did not reduce qrel-based relevant-evidence disparity in this pilot. "
        "See Results and Limitations for measured values and scope."
    )


def page_query_demo(data: Mapping[str, Any]) -> None:
    hero(
        "Query demonstration",
        "Compare the exact saved answers and retrieved evidence for one pilot question.",
        "Saved outputs · no live inference",
    )
    query = query_selector(data, "query_demo_select")
    st.markdown(f"**{query['topic']}** · {query['id']}")
    st.write(query["question"])

    columns = st.columns(3)
    for column, method in zip(columns, METHODS):
        row = get_query_record(data, query["id"], method)
        output = row.get("output", {})
        prediction = output.get("pred")
        docs = saved_documents(row)
        with column:
            st.markdown(f"### {METHOD_LABELS[method]}")
            st.caption(f"{len(docs)} final retrieved documents")
            if prediction is None or str(prediction).strip() == "":
                st.info("No answer was saved for this method/query.")
            else:
                st.code(str(prediction).strip())
            ids = [str(doc.get("id", "N/A")) for doc in docs]
            st.caption("Retrieved IDs: " + (", ".join(ids) if ids else "none saved"))
            for rank, doc in enumerate(docs, 1):
                render_document(doc, rank, show_graph_scores=method != "baseline")


def page_graph_retrieval(data: Mapping[str, Any]) -> None:
    hero(
        "Graph retrieval artifacts",
        "Inspect saved graph-mode rankings, document evidence, runtime analysis, and controller decisions.",
        "Artifact audit",
    )
    if not data["graph_topology_available"]:
        st.warning(
            "No serialized graph node/edge topology was saved by these experiment runs. "
            "This page shows actual saved retrieved-document records, graph scores, analyses, "
            "and decisions; it does not draw or reconstruct relationships."
        )

    query = query_selector(data, "graph_query_select")
    method = st.selectbox(
        "Graph method",
        ["graph", "adaptive_fair"],
        format_func=lambda value: METHOD_LABELS[value],
        key="graph_method_select",
    )
    row = get_query_record(data, query["id"], method)
    output = row.get("output", {})
    initial = output.get("initial_retrieval_result", [])
    final_docs = saved_documents(row)
    decisions = output.get("graph_rag_decisions")

    c1, c2, c3 = st.columns(3)
    c1.metric("Initial saved documents", len(initial))
    c2.metric("Final saved documents", len(final_docs))
    c3.metric("Additional retrieval operations", output.get("additional_retrieval_operations", 0))

    if isinstance(decisions, list):
        st.markdown("#### Saved controller / ranking decisions")
        st.json(decisions, expanded=True)
    else:
        st.info("No per-query controller decision record was saved for this run.")

    analysis = output.get("graph_fairness_analysis")
    if isinstance(analysis, Mapping):
        st.markdown("#### Saved runtime fairness analysis")
        st.caption(
            "This is the runtime analyzer’s lexical relevance proxy and demographic metadata. "
            "It is not the offline qrel judgment."
        )
        a, b, c = st.columns(3)
        a.metric("Runtime proxy disparity", fmt(analysis.get("group_disparity")))
        b.metric("Runtime relevance-rate gap", fmt(analysis.get("relevance_rate_gap")))
        c.metric("Groups missing in proxy analysis", len(analysis.get("missing_groups", [])))
        group_rows = []
        for group, values in analysis.get("groups", {}).items():
            group_rows.append(
                {
                    "Saved group": group,
                    "Documents": values.get("documents"),
                    "Proxy-relevant documents": values.get("relevant_documents"),
                    "Proxy relevance rate": values.get("relevance_rate"),
                    "Role": analysis.get("group_roles", {}).get(group, "unassigned"),
                }
            )
        if group_rows:
            st.dataframe(pd.DataFrame(group_rows), hide_index=True, use_container_width=True)

    st.markdown("#### Saved document records")
    st.caption(
        "Document records below are real top-k results saved by the run. The original graph object, "
        "entity nodes, and relationship edges were not serialized."
    )
    score_rows = []
    for rank, doc in enumerate(final_docs, 1):
        score_rows.append(
            {
                "Rank": rank,
                "Document ID": str(doc.get("id", "N/A")),
                "BM25 score": doc.get("score"),
                "Lexical relevance proxy": doc.get("graph_relevance_score"),
                "Graph rank score": doc.get("graph_rank_score"),
                "Saved group metadata": doc.get("gender", doc.get("group", "N/A")),
            }
        )
    if score_rows:
        score_frame = pd.DataFrame(score_rows)
        st.dataframe(score_frame, hide_index=True, use_container_width=True)
        chart_columns = [
            name for name in ("Lexical relevance proxy", "Graph rank score")
            if score_frame[name].notna().any()
        ]
        if chart_columns:
            st.bar_chart(score_frame.set_index("Document ID")[chart_columns])
        for rank, doc in enumerate(final_docs, 1):
            render_document(doc, rank, show_graph_scores=True)


def page_fairness(data: Mapping[str, Any]) -> None:
    hero(
        "Fairness analysis",
        "Separate runtime proxy signals from offline annotation-based evaluation.",
        "Offline evaluation summary",
    )
    st.info(
        "This dashboard reads aggregate offline qrel results already saved after each run. "
        "It does not load qrels into retrieval or generation."
    )
    methods = data["comparison"]["methods"]
    disparities = pd.DataFrame(
        [
            {
                "Method": METHOD_LABELS[method],
                "Relevant-evidence disparity (offline qrels)": methods[method].get(
                    "relevant_evidence_group_disparity"
                ),
                "Answer priority-ratio gap": methods[method]
                .get("answer_group_position_rates", {})
                .get("absolute_gap"),
            }
            for method in METHODS
        ]
    )
    st.markdown("#### Measured disparity")
    st.dataframe(disparities, hide_index=True, use_container_width=True)
    st.caption(
        "Relevant-evidence disparity is the max-minus-min share of qrel-relevant retrieved documents "
        "by group. Answer priority-ratio gap is a separate derived answer metric."
    )

    representation = []
    for method in METHODS:
        group_summary = methods[method].get("relevant_irrelevant_by_group", {})
        for group, counts in group_summary.items():
            representation.append(
                {
                    "Method": METHOD_LABELS[method],
                    "Group": group,
                    "Relevant": counts.get("relevant", 0),
                    "Irrelevant": counts.get("irrelevant", 0),
                }
            )
    st.markdown("#### Retrieved group representation · offline qrels")
    if representation:
        frame = pd.DataFrame(representation)
        st.dataframe(frame, hide_index=True, use_container_width=True)
        chart_rows = []
        for method in METHODS:
            by_group = methods[method].get("relevant_irrelevant_by_group", {})
            chart_rows.append(
                {
                    "Method": METHOD_LABELS[method],
                    "Male relevant": by_group.get("male", {}).get("relevant", 0),
                    "Female relevant": by_group.get("female", {}).get("relevant", 0),
                    "Male irrelevant": by_group.get("male", {}).get("irrelevant", 0),
                    "Female irrelevant": by_group.get("female", {}).get("irrelevant", 0),
                }
            )
        st.bar_chart(pd.DataFrame(chart_rows).set_index("Method"))

    reasons = {
        methods[method].get("equalized_odds", {}).get("reason", "Not reported")
        for method in METHODS
    }
    st.markdown("#### Equalized Odds")
    st.metric("Status", "N/A")
    for reason in reasons:
        st.caption(reason)

    query = query_selector(data, "fairness_query_select")
    method = st.selectbox(
        "Runtime analysis method",
        ["graph", "adaptive_fair"],
        format_func=lambda value: METHOD_LABELS[value],
        key="fairness_method_select",
    )
    row = get_query_record(data, query["id"], method)
    output = row.get("output", {})
    st.markdown("#### Saved decision trace for the selected question")
    decisions = output.get("graph_rag_decisions")
    if decisions is not None:
        st.json(decisions, expanded=True)
    else:
        st.info("No decision trace was saved.")
    st.caption(
        "Per-question graph fairness analysis uses the saved runtime lexical proxy. "
        "It is distinct from the offline qrel-based group counts above."
    )
    for key, label in (
        ("initial_graph_fairness_analysis", "Initial runtime proxy analysis"),
        ("graph_fairness_analysis", "Final runtime proxy analysis"),
    ):
        value = output.get(key)
        if value is not None:
            with st.expander(label):
                st.json(value, expanded=False)


def page_results(data: Mapping[str, Any]) -> None:
    hero(
        "Results comparison",
        "Measured pilot metrics from the saved baseline, graph, and adaptive runs.",
        "10-question pilot · saved metrics",
    )
    methods = data["comparison"]["methods"]
    frame = pd.DataFrame(
        [
            {
                "Method": METHOD_LABELS[method],
                "EM": methods[method].get("exact_match"),
                "ROUGE-1": methods[method].get("rouge_1"),
                "Answer group gap": methods[method]
                .get("answer_group_position_rates", {})
                .get("absolute_gap"),
                "Relevant-evidence disparity": methods[method].get(
                    "relevant_evidence_group_disparity"
                ),
                "Qrel MRR@5": methods[method].get("qrel_mrr_at_5_mean"),
                "Equalized Odds": methods[method].get("equalized_odds", {}).get("status", "N/A"),
                "Retrieved docs": methods[method].get("retrieved_documents"),
                "Extra searches": methods[method].get("additional_retrieval_operations"),
                "Extra docs": methods[method].get("additional_documents"),
                "Latency/query (s)": methods[method].get("average_end_to_end_seconds_per_question"),
            }
            for method in METHODS
        ]
    )
    st.dataframe(frame, hide_index=True, use_container_width=True)

    st.markdown("#### Answer quality")
    st.bar_chart(frame.set_index("Method")[["EM", "ROUGE-1"]])
    left, right = st.columns(2)
    with left:
        st.markdown("#### Retrieval")
        st.bar_chart(frame.set_index("Method")[["Qrel MRR@5", "Relevant-evidence disparity"]])
    with right:
        st.markdown("#### Cost")
        st.caption("End-to-end latency per query (seconds)")
        st.bar_chart(frame.set_index("Method")[["Latency/query (s)"]])
        st.caption("Adaptive retrieval overhead (counts)")
        st.bar_chart(frame.set_index("Method")[["Extra searches", "Extra docs"]])

    st.markdown("#### Unavailable or unestablished measures")
    st.markdown(
        "- **Equalized Odds:** N/A; annotations do not provide both outcome classes within each group.\n"
        "- **Geographic fairness:** not evaluated in this gender pilot.\n"
        "- **Statistical significance:** not assessed for 10 questions.\n"
        "- **Full-corpus retrieval:** not measured; runs use restricted judged pools."
    )


def page_limitations(data: Mapping[str, Any]) -> None:
    hero(
        "Limitations and scope",
        "Interpret these results as a small system demonstration, not evidence of general fairness improvement.",
        "Read before presenting",
    )
    limitations = [
        ("10 questions", "One saved question from each of ten selected topic blocks; too small for significance claims."),
        ("Restricted candidate pools", "Topic-specific pools contain only released judged documents (150–188 unique documents per topic), not the complete corpus."),
        ("No full-corpus reproduction", "The original Pyserini index and Wikipedia-scale corpus were not downloaded or evaluated."),
        ("Fairness improvement not demonstrated", "Offline qrel-based relevant-evidence disparity is 1.0 for all three methods."),
        ("Adaptive false positives", "The adaptive run added two female-group documents judged irrelevant offline and had two fewer relevant male documents in final top-k results."),
        ("Equalized Odds unavailable", "The released split lacks both relevance classes for each group, so group-conditional TPR/FPR cannot be estimated."),
        ("Graph topology not persisted", "Per-run document rankings, scores, group analyses, and decisions are saved; node/edge graph objects are not."),
        ("Gender only", "No geographic fairness result is included in this dashboard."),
    ]
    for title, description in limitations:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)
    st.subheader("Future work")
    st.write(
        "Evaluate a larger, representative question set with a documented full corpus/index, "
        "persist graph nodes and edges for audit, and use annotations with sufficient outcome "
        "coverage to estimate group-conditional error rates."
    )


def main() -> None:
    try:
        data = load_dashboard_data(PROJECT_ROOT)
    except DashboardDataError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.markdown("## ◈ Adaptive Fair Graph RAG")
    st.sidebar.caption("Saved-artifact research demo")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Query demonstration",
            "Graph retrieval",
            "Fairness analysis",
            "Results comparison",
            "Limitations",
        ],
    )
    st.sidebar.divider()
    st.sidebar.metric("Saved questions", data["question_count"])
    st.sidebar.caption("Read-only · no model calls · no qrels passed to inference")
    st.sidebar.caption(f"Source: {data['report_path'].relative_to(PROJECT_ROOT)}")

    pages = {
        "Overview": page_overview,
        "Query demonstration": page_query_demo,
        "Graph retrieval": page_graph_retrieval,
        "Fairness analysis": page_fairness,
        "Results comparison": page_results,
        "Limitations": page_limitations,
    }
    pages[page](data)
    st.markdown("---")
    st.caption("All numbers and answers are loaded from saved run artifacts.")


if __name__ == "__main__":
    main()

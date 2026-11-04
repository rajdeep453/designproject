# Completed Windows pilot

- Dataset: 10 real questions, one per selected topic, from restricted annotation-linked judged pools.
- Retrieval: topic-scoped in-memory BM25, top 5; no full Wikipedia corpus or index was downloaded.
- Generator: local Qwen/Qwen2.5-0.5B-Instruct, revision 7ae557604adf67be50417f59c2c2f167def9a775, FP16 on RTX 4050.
- Results: comparison_metrics.csv and comparison_metrics.json.
- Per-run configs, predictions, retrieval records, and evaluator outputs are in the three trek_2022_fairness_*_{baseline,graph,adaptive}_male_female_pilot10 directories under the parent pilot output directory.
- Offline qrels were joined after each run by topic and document ID. The controller never received the relevance labels.
- The first baseline attempt used a save note without male_female; generation and EM/ROUGE completed, but the original fairness/MRR evaluators could not select the annotation columns. That incomplete attempt is retained in its own output directory and excluded from the comparison. The completed baseline rerun uses baseline_male_female_pilot10.
- These are restricted-pool pilot results, not full-corpus or statistically significant findings.

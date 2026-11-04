$ErrorActionPreference = "Stop"
$py = ".\.venv-afgr\Scripts\python.exe"

& $py -m graph_rag.prepare_pilot
& $py scripts/validate_qwen_gpu.py --max-new-tokens 64
& $py run_exp.py --method_name naive --config_file graph_rag/pilot10_gender.yaml --pilot_mode --comparison_metrics --rag_mode baseline --split pilot10_gender --dataset_name trek_2022_fairness --retrieval_method bm25_small --retrieval_topk 5 --gpu_id 0 --save_note baseline_male_female_pilot10
& $py run_exp.py --method_name naive --config_file graph_rag/pilot10_gender.yaml --pilot_mode --rag_mode graph --split pilot10_gender --dataset_name trek_2022_fairness --retrieval_method bm25_small --retrieval_topk 5 --gpu_id 0 --save_note graph_male_female_pilot10
& $py run_exp.py --method_name naive --config_file graph_rag/pilot10_gender.yaml --pilot_mode --rag_mode adaptive_fair --split pilot10_gender --dataset_name trek_2022_fairness --retrieval_method bm25_small --retrieval_topk 5 --gpu_id 0 --save_note adaptive_male_female_pilot10
& $py scripts/evaluate_pilot_experiments.py
& $py -m unittest discover -s tests -v

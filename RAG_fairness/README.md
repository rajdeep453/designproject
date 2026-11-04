# **Official Git Repository for COLING 2025 Paper**

**Paper Title**  
[**Does RAG Introduce Unfairness in LLMs? Evaluating Fairness in Retrieval-Augmented Generation Systems**](https://aclanthology.org/2025.coling-main.669/)

This repository contains the **source code** and **data** for our paper accepted at COLING 2025. It investigates whether retrieval methods and knowledge sources introduce biases or disparities in Large Language Models (LLMs) when using Retrieval-Augmented Generation (RAG) systems.

---

## **Table of Contents**
1. [Overview](#overview)  
2. [Installation and Environment](#installation-and-environment)  
3. [FlashRAG Modifications](#flashrag-modifications)  
4. [Data Preparation](#data-preparation)  
5. [Index Building](#index-building)  
6. [Evaluation](#evaluation)  
7. [Contact](#contact)

---

## **1. Overview** <a name="overview"></a>

In this project, we explore how the retrieval component in a RAG pipeline can lead to unfair or biased outcomes when combined with LLMs. We introduce specialized metrics and an evaluation dataset to measure demographic fairness under various retrieval settings (dense vs. sparse, different embedding models, etc.).

**Highlights**:

- **RAG Fairness Evaluation**: Custom metrics and code for measuring biases in RAG systems.
- **Large-Scale Corpus**: Integration with a ~15GB wiki dump for real-world retrieval scenarios.
- **Flexible Implementation**: Built atop [FlashRAG](https://github.com/codecodebear/FlashRAG-SCU) for modular retrieval and generation.

---

## **2. Installation and Environment** <a name="installation-and-environment"></a>

We provide a Conda environment file for convenience. **Ensure you have Conda** (e.g., [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)).

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate my_rag_env
```

Note: Adjust the Conda environment name if it's specified differently in environment.yml.


## **3. FlashRAG Modification**<a name="flashrag-modifications"></a>

We introduced custom evaluation metrics and configuration options in FlashRAG to evaluate fairness in RAG systems. These changes may conflict with an existing FlashRAG installation.

Patched Files:
   * `flashrag/config/config.py`: add evaluation mode.
   * `flashrag/evaluator/metrics.py`: add runtime evalution code.
   * `flashrag/prompt/base_prompt.py`: debug for 'instruct' model.

Search for comments like:

```
#-----------------# For RAG Fairness Only #-----------------#

#-----------------------------------------------------------#
```

and apply or overwrite these files in your local FlashRAG setup. Reinstall FlashRAG or adjust your PYTHONPATH to use our modified version.

More details of the new evaluation metrics described in the comments `flashrag/evaluator/metrics.py` of each function.



## **4. Data Preparation**<a name="data-preparation"></a>

1. Preprocessing, the code will automatically download the dataset by (`ir_datasets.load("trec-fair/2022/train")`) and preprocess it. The dataset is stored in the `data/` directory. **We also upload our corpus (expecially for building index) on [Huggingface](https://huggingface.co/datasets/SCU-IR/RAG_fairness). You can use it directly.**

    * Run `preprocess_trec_wiki.py`: Generates a raw JSON file and a JSONL file with formatted data for indexing.
    * Example usage:

      ```
      python preprocess_trec_wiki.py
      ```
      
    * Adjust the script or arguments as needed for your dataset.

2. Generate fairness Evaluation Data, we also prepared the data within the current `/data` folder. **You can download the processed data directly to use (`trec_2022_fairness/`, `trec_gender_2022.tar.gz` and `trec_geo_full_2022.tar.gz`).**

    * Run `prepare_rag_fairness_data.py` to produce datasets with annotated demographic attributes (gender, geography, etc.).
    * Outputs stored in `data/trek_2022_fairness`.

## **5. Index Building** <a name="index-building"></a>

We use [FlashRAG’s index-building guide](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/building-index.md). Ensure your corpus is in JSONL format:

```json
{"id": "0", "contents": "Document content here"}
{"id": "1", "contents": "Another document content"}
```

For dense retrieval methods, especially the popular embedding models, we use faiss to build index. Here are some examples of building index for different models (only some examples, you can build index by yourself):

* Based on `e5-base-v2` embedding model:

```
CUDA_VISIBLE_DEVICES=7 python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path /path_to_your_model_file/e5-base-v2 \
    --corpus_path /path_to_your_data_file/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_formatted.jsonl \
    --save_dir /path_to_your_data_file/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_index \
    --use_fp16 \
    --max_length 256 \
    --batch_size 1024 \
    --pooling_method mean \
    --faiss_type Flat 
```

* Based on `e5-large-v2` embedding model:

```
CUDA_VISIBLE_DEVICES=7 python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path /path_to_your_model_file/e5-large-v2 \
    --corpus_path /path_to_your_data_file/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_formatted.jsonl \
    --save_dir /path_to_your_data_file/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_index_e5_large \
    --use_fp16 \
    --max_length 256 \
    --batch_size 1024 \
    --pooling_method mean \
    --faiss_type Flat 
```

* Based on `bm25` retrieval method:

```
python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path /path_to_your_data_file/trek_fair_2022_train_id_title_doc.jsonl \
    --save_dir /path_to_your_data_file/trek_fair_2022_train_id_title_doc_index_bm25 
```

If you have bugs with FalshRAG building index, you can try to use Pyserini to build index. Here is an example of building index using Pyserini:

```
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /ssd/public_datasets/wiki/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_index_bm25/temp \
  --index /ssd/public_datasets/wiki/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_index_bm25/bm25 \
  --generator DefaultLuceneDocumentGenerator\
  --threads 1 
```

## **6. Evaluating** <a name="evaluation"></a>

To evaluate the model, you can use the following command:

For example:

```
python run_exp.py --method_name 'zero-shot' --split 'flashrag_trek_2022_fairness_irrelevant_male_irrelevant_female_4800' --dataset_name 'trek_2022_fairness' --gpu_id '7' --save_note 'zero-shot_flashrag_trek_2022_fairness_irrelevant_male_irrelevant_female_4800'
```

* The script references my_config.yaml for model paths, hyperparameters, etc.
* Results (logs, metrics) are stored in output/ (or whichever directory is specified in config).
* Basicly, we add four new evalutions metrics (corresponding to the `metrics.py` file in FlashRAG):
  - `priorityfairness` is used to evaluate the fairness of the model for differnt groups, according the formular in the paper Appendix A.1.
  - `optionem` is used to the model generation ability, which the ratio of answers is select from the options (no shown in paper).
  - `mrr` is used to the evaluate the retrieval performance based on the ground truth documents.
  - `judgeeval` is used to the evaluate the judger based model's performance, check the em and fairness for each ground under with judger (retrieve news docs) and without judger (no retrieve news docs).

## **7. Contact** <a name="contact"></a>

If you have any questions, issues, or suggestions, please reach out via:

Contact Email: xwu5@scu.edu

Thank you for your interest in our work! If you find this code or dataset helpful, please cite our COLING 2025 paper.

## Adaptive Fair Graph RAG prototype

For the local read-only demonstration dashboard and PowerShell launch command, see [dashboard/README.md](dashboard/README.md).

The `graph_rag/` package adds a lightweight, in-memory evidence graph while keeping the original `naive` and `zero-shot` paths available. Graph nodes represent documents, proper-name entity candidates, and co-mention relationship records. A co-mention relation is a navigation hint; it is not a claim that a causal or semantic relation was extracted.

Select a mode with `--rag_mode` when invoking the existing `naive` command:

```bash
# Original RAG baseline (same default behavior as before). Add the comparison
# metrics flag for apples-to-apples evaluation with graph modes.
python run_exp.py --method_name naive --rag_mode baseline --comparison_metrics --split SPLIT --dataset_name trek_2022_fairness --gpu_id 0 --save_note baseline_SPLIT

# Graph-ranked evidence
python run_exp.py --method_name naive --rag_mode graph --split SPLIT --dataset_name trek_2022_fairness --gpu_id 0 --save_note graph_SPLIT

# Adaptive fair graph retrieval
python run_exp.py --method_name naive --rag_mode adaptive_fair --split SPLIT --dataset_name trek_2022_fairness --gpu_id 0 --save_note adaptive_SPLIT
```

`baseline` is the default mode. Graph runs reuse the configured retriever, prompt, generator, and evaluator. Each graph run saves `graph_rag_metrics.json` in its own output directory, next to FlashRAG's normal metrics and `intermediate_data.json`. A baseline run with `--comparison_metrics` also writes `baseline_comparison_metrics.json` with elapsed time. Use distinct `--save_note` values so baseline, graph, and adaptive artifacts remain separate. Graph runs enable EM, F1, accuracy, precision, recall, ROUGE-1, `priorityfairness`, and `mrr` in the existing evaluator; the flag enables the same metric set for baseline comparison without changing its retrieval or generation path.

The graph configuration in `my_config.yaml` sets `relevance_threshold`, `disparity_threshold`, `max_rounds`, `max_additional_docs`, `max_expansion_queries`, and group annotations. Optional `document_group_map` is a JSON object mapping corpus document IDs to explicit group labels, for example `{"doc-123": "women", "doc-456": "men"}`. Document records can instead carry `demographic_group`, `group`, `gender`, `geography`, `geographic_location`, or `demographics` metadata. Set `protected_groups`, `non_protected_groups`, and `expected_groups` to labels that exist in the supplied annotation source.

Demographic membership is never used as a relevance signal or appended to retrieval queries. Relevance uses explicit document fields (`is_relevant`, `relevance_label`, or `relevance_score`) when available; otherwise token overlap with the question is used as a limited proxy. The controller retrieves only when expected groups are missing from relevant evidence or group relevance rates exceed the configured gap. It targets question-grounded entity expansions and filters out candidates without relevance support. It does not require equal document counts between groups.

The report contains evaluator utility/retrieval metrics, group evidence disparity, additional retrieval operations, retrieved document count, and elapsed time. Equalized odds (TPR/FPR gaps for document relevance decisions) is reported only when explicit relevance labels and group labels are available. The prepared fairness question files encode answer options and gold answers; they do not by themselves annotate every corpus document with a demographic group. Do not use answer gold labels as runtime evidence labels. Without a valid document annotation map, group-aware analysis is marked unavailable and the adaptive fairness trigger cannot make a group-specific claim.

### Small smoke test

The synthetic smoke setup is `graph_rag/smoke_config.json`; it uses a fake retriever and generator and requires no model, index, GPU, or corpus. From the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests exercise graph construction, relevance versus demographic membership, group disparity behavior, bounded adaptive retrieval, equalized odds with explicit labels, and an end-to-end pipeline-adapter run. This is not a substitute for running the system on the released corpus and models.

### Reproducing a research comparison

Use a compatible Linux Conda environment (the supplied environment is named `flashrag` and pins CUDA/GPU packages), retrieve required Git LFS data, and configure model/corpus/index paths. Provide document group metadata from a documented annotation source. Run the three commands above with the same split, sample, retriever, generator, and seed, changing only `--rag_mode` and `--save_note`. Compare `metric_score.txt` and `graph_rag_metrics.json`. Report elapsed times from equivalent runs and disclose hardware.

### Windows pilot environment and real-data RAG comparison

The verified environment is Python 3.10.20 x64 in `.venv-afgr/`; the prior temporary environment was left intact. Its NumPy native module had previously failed to load with “Access is denied.” The active project-local environment imports NumPy 1.26.4 and Pandas 2.2.2, and the Microsoft VC runtime is present. The exact Windows package pins are in `requirements-windows-afgr.txt` and `requirements-windows-afgr-lock.txt`.

From the repository root in PowerShell, create and install the environment as follows:

```powershell
$base = "$env:TEMP\afgr-python\cpython-3.10.20-windows-x86_64-none\python.exe"
& $base -m venv .venv-afgr
$py = ".\.venv-afgr\Scripts\python.exe"
& $py -m pip install --no-cache-dir "torch==2.6.0+cu126" --extra-index-url https://download.pytorch.org/whl/cu126
& $py -m pip install --no-cache-dir -r requirements-windows-afgr.txt
```

The original Linux `environment.yml` includes Linux-specific and full-corpus tooling. The Windows pilot does not install FAISS, Pyserini, vLLM, or Sentence Transformers: the annotation-linked BM25 backend needs none of them. The original Pyserini implementation remains in place. No full Wikipedia corpus or index was downloaded.

Qwen2.5-0.5B-Instruct was downloaded from Hugging Face at revision `7ae557604adf67be50417f59c2c2f167def9a775`. Its standard `model.safetensors` file is 988 MB according to [the official model file](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/model.safetensors); the local SHA-256 was verified. No quantization is used. CUDA was verified with a real FP16 matrix operation and model generation on the RTX 4050.

To fetch the same pinned snapshot if it is not already present:

```powershell
$py = ".\.venv-afgr\Scripts\python.exe"
& $py -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775', local_dir='models/Qwen2.5-0.5B-Instruct', allow_patterns=['config.json','generation_config.json','merges.txt','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json'])"
```

The 10-question gender pilot selects the first question from each of ten topic blocks. Each topic's restricted candidate pool comes from the released 100 relevant-male/irrelevant-female annotation pairs in that block (150–188 unique judged documents per topic). The document IDs and group labels were matched against the released golden-document records. The qrels stay outside the runtime corpus and are loaded only by offline scoring after retrieval/generation.

Run the three complete pilot modes and offline comparison with the same config and data:

```powershell
$py = ".\.venv-afgr\Scripts\python.exe"
& $py -m graph_rag.prepare_pilot
& $py scripts/validate_qwen_gpu.py --max-new-tokens 64
& $py run_exp.py --method_name naive --config_file graph_rag/pilot10_gender.yaml --pilot_mode --comparison_metrics --rag_mode baseline --split pilot10_gender --dataset_name trek_2022_fairness --retrieval_method bm25_small --retrieval_topk 5 --gpu_id 0 --save_note baseline_male_female_pilot10
& $py run_exp.py --method_name naive --config_file graph_rag/pilot10_gender.yaml --pilot_mode --rag_mode graph --split pilot10_gender --dataset_name trek_2022_fairness --retrieval_method bm25_small --retrieval_topk 5 --gpu_id 0 --save_note graph_male_female_pilot10
& $py run_exp.py --method_name naive --config_file graph_rag/pilot10_gender.yaml --pilot_mode --rag_mode adaptive_fair --split pilot10_gender --dataset_name trek_2022_fairness --retrieval_method bm25_small --retrieval_topk 5 --gpu_id 0 --save_note adaptive_male_female_pilot10
& $py scripts/evaluate_pilot_experiments.py
& $py -m unittest discover -s tests -v
```

The three runs completed on the same 10 questions using top 5. The table reports answer group gap as the absolute difference between the original evaluator's male/female priority ratios; relevant-evidence disparity is calculated separately from released document qrels.

| Method | EM | ROUGE-1 | Answer group gap | Relevant-evidence disparity | Qrel MRR@5 | Equalized Odds | Retrieved docs | Extra searches / docs | Mean end-to-end latency/query |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Baseline RAG | 0.200 | 0.200 | 0.100 | 1.000 | 0.675 | N/A | 50 | 0 / 0 | 0.198 s |
| Graph RAG | 0.200 | 0.200 | 0.200 | 1.000 | 0.833 | N/A | 50 | 0 / 0 | 0.197 s |
| Adaptive Fair Graph RAG | 0.200 | 0.200 | 0.100 | 1.000 | 0.833 | N/A | 50 | 14 / 2 | 0.205 s |

The original baseline pipeline ran with the restricted in-memory `bm25_small` backend; it did not use the original Pyserini index. The adaptive controller added two female-group documents, both judged irrelevant by the released qrels, while two fewer relevant male documents remained in its final top 5 lists. Thus the qrel-based relevant-evidence disparity stayed at 1.0. The result demonstrates the limits of a lexical relevance proxy; it does not show a fairness improvement. Equalized Odds is N/A because this released split lacks both relevance classes for each demographic group. Geographic fairness was not evaluated. These 10 questions and restricted pools are a small pilot, not full-corpus or significance-tested results.

The exact GPU/model check is saved in `output/environment_validation/qwen_gpu_generation.json`; environment details are in `output/environment_validation/environment_report.json`. Per-run configs, predictions, evaluator metrics, and retrieval outputs remain in separate run directories under `output/real_experiments/pilot10_gender/`. The combined CSV/JSON and reproduction PowerShell script are under `full_rag_comparison/`. A threshold-0.20 sensitivity run is separately saved under `sensitivity_threshold_0_20/`; it is post-hoc because its threshold was selected after inspecting the primary qrels and must not be treated as confirmatory.

```
@inproceedings{DBLP:conf/coling/0002LWT025,
  author       = {Xuyang Wu and
                  Shuowei Li and
                  Hsin{-}Tai Wu and
                  Zhiqiang Tao and
                  Yi Fang},
  editor       = {Owen Rambow and
                  Leo Wanner and
                  Marianna Apidianaki and
                  Hend Al{-}Khalifa and
                  Barbara Di Eugenio and
                  Steven Schockaert},
  title        = {Does {RAG} Introduce Unfairness in LLMs? Evaluating Fairness in Retrieval-Augmented
                  Generation Systems},
  booktitle    = {Proceedings of the 31st International Conference on Computational
                  Linguistics, {COLING} 2025, Abu Dhabi, UAE, January 19-24, 2025},
  pages        = {10021--10036},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://aclanthology.org/2025.coling-main.669/},
  timestamp    = {Tue, 28 Jan 2025 16:22:22 +0100},
  biburl       = {https://dblp.org/rec/conf/coling/0002LWT025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


# Reranking Models

This folder contains the implementation for various reranking models used in the Dataset Search with Examples (DSE) task. The following models are included:

- **Stella**
- **SFR-Embedding-Mistral**
- **BGE-reranker**
- **LLM Reranking**

## Prepare Data Files

Before running the reranking models, ensure the following files are properly prepared:

**rerank_{k}.json**

This file should contain a JSON object formatted as:

```json
{
    "case_id_1": ["dataset_id_1", "dataset_id_2", "dataset_id_3", ...],
    "case_id_2": ["dataset_id_4", "dataset_id_5", "dataset_id_6", ...],
    ...
}
```

**Data Folders: fold_{i}**

For each fold (e.g., `fold_1`, `fold_2`, ...), prepare the following JSON files:
- `train.json`
- `valid.json`
- `test.json`

Each file should contain a JSON object formatted like:

```json
{
    "case_id_1": {
        "candidate_dataset_id_1": rel_score,
        "candidate_dataset_id_2": rel_score,
        ...
    },
    "case_id_2": {
        "candidate_dataset_id_3": rel_score,
        "candidate_dataset_id_4": rel_score,
        ...
    },
    ...
}
```

**query_info.json**

This file contains information about each case. The format should be:

```json
{
    "case_id_1": {"content": "case_text_or_info"},
    "case_id_2": {"content": "case_text_or_info"},
    ...
}
```

**dataset_info.json**

This file contains information about each dataset. The format should be:

```json
{
    "dataset_id_1": {"content": "dataset_text_or_info"},
    "dataset_id_2": {"content": "dataset_text_or_info"},
    ...
}
```

## Stella and SFR Reranking Models

We implement reranking using Stella and SFR-Embedding-Mistral models.

- **Stella** model: [stella_en_1.5B_v5](https://huggingface.co/dunzhang/stella_en_1.5B_v5) 
- **SFR** model: [SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral)
  
For details on using and running these models, refer to the code in [./dense_rerank.py](./dense_rerank.py).


## BGE-reranker

We use [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) for BGE reranking. To use it, follow these steps:

1. Install FlagEmbedding with pip:

```
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install  .[finetune]
```

2. Fine-tuning and usage details can be found in [./bge_reranker.py](./bge_reranker.py).


## LLM Reranking

For **LLM**-based reranking, the zero-shot and one-shot prompts are defined in [./LLM/prompt.py](./LLM/prompt.py).

The implementation details of [RankLLM](https://github.com/castorini/rank_llm) refer to [./LLM/rankllm.py](./LLM/rankllm.py) and [./LLM/rank_DSE_template.yaml](./rank_DSE_template.yaml).

The iterative grouping method used for reranking is based on the work of [Zhang et al.](https://doi.org/10.1145/3626772.3657966). For implementation details, refer to [./LLM/iterative_group_rerank.py](./LLM/iterative_group_rerank.py).

## GNN (HINormer & HHGT)

The `GNN` directory is organised as follows:

- data/: the original data of the graph. Please download the data from [Zenodo](https://zenodo.org/records/17828423) and extract it into the directory.
- result/: contains results of baselines.
- code/
    - run.py: reranking by HINormer.
    - run_hhgt.py: reranking by HHGT.
    - model.py: implementation of HINormer(ref: https://github.com/Ffffffffire/HINormer).
    - hhgt_model.py: implementation of HINormer(ref: https://github.com/qiuyu111/HHGT).
    - mag_settings_cla.yaml: hyper parameters setting of HHGT
    - utils/: contains tool functions.

### Requirements

- Python==3.9.0
- Pytorch==1.12.0
- Networkx==2.8.4
- numpy==1.22.3
- dgl==0.9.0
- scikit-learn==1.1.1
- scipy==1.7.3

### Running experiments

We train our model using NVIDIA GeForce RTX 4090 GPU with CUDA 12.2.

For evaluation of HINormer with the annotator split:
```
bash HINormer_union.sh
```

For evaluation of HINormer with the five-fold split:
```
bash HINormer_fold.sh
```

For evaluation of HHGT with the annotator split:
```
bash HHGT_union.sh
```

For evaluation of HHGT with the five-fold split:
```
bash HHGT_fold.sh
```

Converting GNN results into results that can be evaluated:
```bash
python Code/Reranking/GNN/GNN_to_origin.py \
  --mapping Code/Reranking/GNN/GNN_to_origin.json \
  --run path/to/GNN_output.json \
  --result path/to/your_result.json
```

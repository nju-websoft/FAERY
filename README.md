# DSEBench

**DSEBench** is a test collection designed to support the evaluation of **Dataset Search with Examples (DSE)**, a task that generalizes two established paradigms: keyword-based dataset search and similarity-based dataset discovery. Given a textual query $q$ and a set of target datasets $D_t$ known to be relevant, the goal of DSE is to retrieve a ranked list $D_c$ of candidate datasets that are both relevant to $q$ and similar to the datasets in $D_t$.

As an extension, **Explainable DSE** further requires identifying, for each result dataset $d \in D_c$, a subset of metadata or content fields that explain its relevance to $q$ and similarity to $D_t$.

For further details, please refer to the accompanying paper.

---

## Data Download

The full test collection (Datasets, Queries, Cases, Splits, Relevance Judgments) and Evaluation Scripts are hosted on [Zenodo](https://zenodo.org/records/17805089).

Please download the data from Zenodo and extract it into the `Data/` directory in the root of this repository.

---

## Baselines

We provide comprehensive baseline results for Retrieval, Reranking, and Explanation tasks. All result files are stored in the `Baselines/` directory. 

The complete evaluation results are available in [./Baselines/evaluation_results.md](./Baselines/evaluation_results.md). For detailed experimental setups and analysis, please refer to the corresponding section in our paper.

### Retrieval Baselines

We evaluated a wide range of retrieval models, categorized as follows:

* **Sparse Retrieval:**
    * **BM25**, **TF-IDF**
* **Dense Retrieval:**
    * *Unsupervised:* **BGE** (`bge-large-en-v1.5`), **GTE** (`gte-large`)
    * *Supervised:* **DPR**, **ColBERTv2**, **coCondenser**
* **Relevance Feedback:**
    * **Rocchio** (adapted for DSE)

The result files (e.g., `./Baselines/Retrieval/BM25_results.json`) are stored in JSON format.

* **Structure:** `{case_id: {candidate_dataset_id: retrieval_score, ...}, ...}`
* **Meaning:** High scores indicate higher relevance.

**Example Content:**
```json
{
  "1": {
    "002ece58-9603-43f1-8e2e-54e3d9649e84": 1684.3712938069227,
    "99e3b6a2-d097-463f-b6e1-3caceff300c9": 1493.7291680589358,
    ...
  },
  "2": { ... }
}
```

We provide an evaluation script `evaluate_dse.py` (located in `./Code/Evaluation/`) that uses `pytrec_eval` to calculate metrics like MAP, NDCG and Recall.

```bash
python Code/Evaluation/evaluate_dse.py \
  --qrels Data/human_annotated_judgments.json \
  --run Baselines/Retrieval/BM25_results.json
```

### Reranking Baselines

We evaluated reranking models:

  * **Text-based Models:**
      * **Stella** (`stella_en_1.5B_v5`)
      * **SFR** (`SFR-Embedding-Mistral`)
      * **BGE-reranker** (`bge-reranker-v2-minicpm-layerwise`)
  * **Structure-based Models:**
      * **HINormer**
      * **HHGT**
  * **LLM** (Evaluated in Zero-shot, One-shot, RankLLM, and Multi-layer settings)

The file format and the evaluation script are same as retrieval.

### Explanation Baselines (Explainable DSE)

We evaluated post-hoc explanation methods to identify *why* a dataset is retrieved (i.e., identifying indicator fields for query relevance and target similarity).

  * **Explainers:**
      * **Feature Ablation**
      * **LIME**
      * **SHAP**
      * **LLM**

The result files (e.g., `./Baselines/Explanation/SHAP/BM25_result.json`) contain binary masks indicating selected fields.

  * **Structure:** `{case_id: {dataset_id: {"query": [binary_list], "dataset": [binary_list]}, ...}, ...}`
  * **Fields Order:** `['title', 'description', 'tags', 'author', 'summary']`
  * **Meaning:** 
    * `"query"` means explanation of query relevance; `"dataset"` means explanation of target similarity. 
    * `1` indicates the field explains the relevance/similarity; `0` means it does not.

**Example Content:**

```json
{
  "1": {
    "6aec7dbf-87d1-467e-b181-8328cbca79ba": {
        "query":   [1, 1, 1, 0, 1],  // Title & Description & Tags & Summary explain Query Relevance
        "dataset": [1, 1, 1, 0, 1]   // Title & Description & Tags & Summary explain Target Similarity
    }
  }
}
```

We provide an evaluation script `evaluate_explanation.py` (located in `./Code/Evaluation/`) to calculate the F1-score of the generated explanations against human annotations.

```bash
python Code/Evaluation/evaluate_explanation.py \
  --qrels Data/human_annotated_judgments.json \
  --run Baselines/Explanation/SHAP/BM25_result.json
```

---

## Source Codes

All implementation source code is available in the `./Code` directory. For detailed implementation specifics and hyperparameter settings, please refer to `implementation_details.md`.

### Dependencies

To run the code, ensure you have the following dependencies installed:

- Python 3.9
- rank-bm25
- pytrec_eval
- scikit-learn
- sentence-transformers
- faiss-gpu
- ragatouille
- tevatron
- torch
- shap
- lime
- zhipuai
- FlagEmbedding
- Networkx
- dgl
- scipy

### Retrieval Models

Detailed documentation and code examples for retrieval models are provided in the [./Code/Retrieval/README.md](./Code/Retrieval/README.md).

The retrieval models include:
- **Sparse Retrieval Models:**
  - BM25
  - TF-IDF
- **Dense Retrieval Models:**
  - **Unsupervised Dense Retrieval Models:**
    - BGE (bge-large-en-v1.5)
    - GTE (gte-large)
  - **Supervised Dense Retrieval Models:** 
    - coCondenser
    - ColBERTv2
    - DPR

The relevance feedback methods include:
- Rocchio-P
- Rocchio-PN

### Reranking Models

Documentation and code examples for reranking models are provided in the [./Code/Reranking/README.md](./Code/Reranking/README.md).

The reranking models include:
- Stella
- SFR-Embedding-Mistral
- BGE-reranker
- LLM
- HINormer
- HHGT

### Explanation Methods

Documentation and code examples for explanation methods are provided in the [./Code/Explanation/README.md](./Code/Explanation/README.md).

The explanation methods include:
- Feature Ablation
- LIME
- SHAP
- LLM

### LLM Prompts

All prompts are located in [./Code/llm_prompts.py](./Code/llm_prompts.py).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.


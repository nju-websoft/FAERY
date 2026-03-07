# Implementation Details & Hyperparameters

This document provides the exhaustive implementation details and hyperparameter settings for the baselines evaluated on DSEBench, supplementing the descriptions provided in the paper.

## 1. Retrieval Models

* **Input Length:** For all BERT-based models, the maximum length of the input sequence was set to 512 tokens.
* **Hyperparameter Search:** During fine-tuning, we performed a grid search to select the optimal learning rate from `{1e-5, 5e-6}` and batch size from `{8, 16}`.
* **Selected Hyperparameters (Five-Fold Split):**
  * DPR: Learning rate `1e-5`, Batch size `8`
  * ColBERTv2: Learning rate `1e-5`, Batch size `16`
  * coCondenser: Learning rate `1e-5`, Batch size `16`
* **Selected Hyperparameters (Annotator Split):**
  * DPR: Learning rate `1e-5`, Batch size `16`
  * ColBERTv2: Learning rate `1e-5`, Batch size `8`
  * coCondenser: Learning rate `1e-5`, Batch size `8`
* **Relevance Feedback (RF) Methods:** The weights used to combine the vector representations of the original query, positive feedback, and negative feedback to refine the query vector were set to `1.0`, `0.75`, and `0.15`, respectively.

## 2. Reranking Models

* **BGE-reranker:**
  * Five-Fold Split: Learning rate `1e-5`, Batch size `16`
  * Annotator Split: Learning rate `1e-5`, Batch size `8`
* **Structure-based GNN Models (HINormer & HHGT):**
  * Optimization: Trained with a learning rate of `1e-4`. 
  * Batch Size: `64` (Five-fold split) / `256` (Annotator split).
  * HINormer Architecture: `num-gnns = 1`, `num-layers = 2`.
  * HHGT Architecture: `L_hop = 2`, `L_type = 2`, `hops = 1`.
* **LLM Multi-layer Configuration:** The multi-layer approach repeatedly (20 times) divides the datasets into 5 groups, reranks each group, and then determines the final ranking based on the frequency of a dataset appearing in the top half of its respective reranked group.

## 3. Explanation Methods

* **Feature Ablation:** We calculated the ratio of the resulting ranking score (after excluding a field) to the original score. If the ratio dropped below a threshold of `0.95`, it was outputted as an indicator field. If no field met this threshold, the field with the largest score drop was selected as the sole indicator field.
* **LIME:** 
  * Proxy Model: SVM.
  * Perturbation: Given that a dataset has 5 fields, we set the number of perturbed samples to `50`. 
  * Decision Rule: Fields with a predicted negative score from the SVM were outputted as indicator fields.
* **SHAP:** 
  * Algorithm: Partition Explainer. 
  * Decision Rule: Fields with a positive Shapley value were outputted as indicator fields.
* **LLM:** Few-shot prompts include `3` verified examples sampled from training cases.
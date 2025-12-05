import json
import argparse
from sklearn.metrics import f1_score
import numpy as np

FIELDS = ['title', 'description', 'tags', 'author', 'summary']

def load_qrels_explanations(qrels_path):
    """
    Loads ground truth explanations.
    Returns: {case_id: {dataset_id: {'query': [0/1...], 'dataset': [0/1...]}}}
    """
    with open(qrels_path, 'r') as f:
        data = json.load(f)
    
    gt = {}
    for item in data:
        qid = str(item['case_id'])
        doc_id = item['candidate_dataset_id']
        
        if qid not in gt:
            gt[qid] = {}
            
        gt[qid][doc_id] = {
            'query': item['field_query_rel'],
            'dataset': item['field_target_sim']
        }
    return gt

def calculate_f1(gt_list, pred_list):
    return f1_score(gt_list, pred_list, average='macro', zero_division=0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Explainable DSE Results")
    parser.add_argument('--qrels', required=True, help="Path to ground truth file")
    parser.add_argument('--run', required=True, help="Path to system output file")
    args = parser.parse_args()

    gt_data = load_qrels_explanations(args.qrels)
    
    with open(args.run, 'r') as f:
        run_data = json.load(f)

    query_f1s = []
    target_f1s = []

    # Iterate over cases and datasets present in both GT and Run
    for case_id, docs in run_data.items():
        case_id = str(case_id)
        if case_id not in gt_data:
            continue
            
        for doc_id, preds in docs.items():
            if doc_id not in gt_data[case_id]:
                continue
            
            gt_instance = gt_data[case_id][doc_id]
            
            # Evaluate Query Relevance Explanation
            if 'query' in preds and 'query' in gt_instance:
                q_f1 = f1_score(gt_instance['query'], preds['query'], zero_division=0)
                query_f1s.append(q_f1)

            # Evaluate Target Similarity Explanation
            if 'dataset' in preds and 'dataset' in gt_instance:
                t_f1 = f1_score(gt_instance['dataset'], preds['dataset'], zero_division=0)
                target_f1s.append(t_f1)

    print("-" * 30)
    print("Explainable DSE Evaluation Results (Macro F1)")
    print("-" * 30)
    print(f"Query Relevance Explanation F1: {np.mean(query_f1s):.4f}")
    print(f"Target Similarity Explanation F1: {np.mean(target_f1s):.4f}")

if __name__ == "__main__":
    main()

import json
import sys
import pytrec_eval
import argparse

def load_qrels(qrels_path):
    """
    Loads ground truth from the JSON file.
    Calculates relevance score = query_rel * target_sim.
    """
    with open(qrels_path, 'r') as f:
        data = json.load(f)
    
    qrels = {}
    for item in data:
        # Construct a unique query ID based on the case_id
        qid = str(item['case_id'])
        doc_id = item['candidate_dataset_id']
        
        # Calculate relevance as per DSE task definition: product of query_rel and target_sim
        # Ranges: query_rel {0,1,2}, target_sim {0,1,2} -> score {0,1,2,4}
        rel_score = item['query_rel'] * item['target_sim']
        
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = rel_score
        
    return qrels

def load_run(run_path):
    """
    Loads system output from JSON file.
    Format: {case_id: {dataset_id: score}}
    """
    with open(run_path, 'r') as f:
        run = json.load(f)
    # Ensure keys are strings
    return {str(k): v for k, v in run.items()}

def main():
    parser = argparse.ArgumentParser(description="Evaluate DSE Retrieval/Reranking Results")
    parser.add_argument('--qrels', required=True, help="Path to ground truth file (human_annotated_judgments.json)")
    parser.add_argument('--run', required=True, help="Path to system output file")
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    run = load_run(args.run)

    # Define metrics
    metrics = {'map_cut_5', 'ndcg_cut_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'recall_10'}
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run)

    # Calculate averages
    aggregated = {m: 0.0 for m in metrics}
    for qid, measures in results.items():
        for m, val in measures.items():
            aggregated[m] += val
            
    print("-" * 30)
    print("DSE Evaluation Results")
    print("-" * 30)
    if len(results) > 0:
        for m in sorted(metrics):
            print(f"{m}: {aggregated[m] / len(results):.4f}")
    else:
        print("No overlapping queries found between qrels and run.")

if __name__ == "__main__":
    main()
import json


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def min_max_normalize(scores):
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {doc_id: 0.5 for doc_id in scores}
    return {doc_id: (score - min_score) / (max_score - min_score) for doc_id, score in scores.items()}


def interpolate_results(metadata_content, bm25):
    interpolated_results = {}

    for query_id in metadata_content:
        if query_id not in bm25:
            continue

        # Normalize metadata_content scores
        metadata_scores = metadata_content[query_id]
        normalized_metadata = min_max_normalize(metadata_scores)
        # print(metadata_scores)
        # print(normalized_metadata)
        # print("=====================")
        # Normalize BM25 scores
        bm25_scores = bm25[query_id]
        normalized_bm25 = min_max_normalize(bm25_scores)
        # print(bm25_scores, )
        # print(normalized_bm25)
        # print("*******************")
        # Interpolate (average) the normalized scores
        interpolated_scores = {}
        for doc_id in metadata_scores:
            if doc_id in bm25_scores:
                interpolated_score = (normalized_metadata[doc_id] + normalized_bm25[doc_id]) / 2
                interpolated_scores[doc_id] = interpolated_score

        # Sort by interpolated score in descending order
        sorted_scores = sorted(interpolated_scores.items(), key=lambda x: x[1], reverse=True)
        interpolated_results[query_id] = {doc_id: score for doc_id, score in sorted_scores}

    return interpolated_results


# Load the input files
ids = [0, 1]
# ids = [2,4,12,14,15]
# ids = [1,3,6,10,12]
# ids =  [2, 5, 9, 13, 14, ]
method = 'HHGT'
dataset = 'google'
retrieval = 'rerank_union'
fold = 'Annotators_split'
for i in range(len(ids)):
    # metadata_content = load_json(f'../DKG/result/ntcir_metadata_content/run_ntcir_metadata_content_tfidf_{i}.json')
    # metadata_content = load_json(f'result/HINormer/run_{dataset}_{retrieval}_1_1_80_1e-05_10_1_{ids[i]}.json')
    metadata_content = load_json(f'result/{method}/run_{dataset}_{fold}_{retrieval}_1_2_2_20_256_{ids[i]}.json')
    bm25 = load_json(f'../data/{dataset}/{retrieval}.json')

    # Perform interpolation
    interpolated_results = interpolate_results(metadata_content, bm25)

    # Save the results to a new JSON file
    new_dataset = dataset.replace('_stella', '')
    output_file = f'../DKG/result/{new_dataset}_normalize/interpolated_{method}_{fold}_{retrieval}_{i}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(interpolated_results, f, indent=2, ensure_ascii=False)

    print(f"Interpolated results saved to {output_file}")

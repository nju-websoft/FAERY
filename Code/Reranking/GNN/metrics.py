import pytrec_eval
import json
import torch
import numpy as np
import pandas as pd
from torch import nn

def mrr(truth, pred, N):
    """
    MRR
    """
    result = {}
    for k in truth.keys():
        qrels_list, run_list = [], []
        qrels_pos = 0.0
        result[k] = {'mrr': 0.0}
        candidates = truth[k]
        for dataset in candidates:
            qrels_list.append([dataset, candidates[dataset]])
        candidates = pred[k]
        for dataset in candidates:
            run_list.append([dataset, candidates[dataset]])
        qrels_list.sort(key=lambda x: x[1], reverse=True)
        run_list.sort(key=lambda x: x[1], reverse=True)
        if len(run_list) == 0:
            continue
        rr = 0.0
        run_keys = []
        for i in range(N):
            run_keys.append(run_list[i][0])
        for i in range(N):
            if run_keys[i] in truth[k] and truth[k][run_keys[i]] > 0:
                rr += 1 / (i + 1)
                break

        result[k]['mrr'] = rr
    return result


# #
# with open('../data/ntcir/test.json') as f:
#     qrels = json.load(f)

metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10',]
# evaluator = pytrec_eval.RelevanceEvaluator(
#     qrels, {'map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10', })
# # # evaluator = pytrec_eval.RelevanceEvaluator(
# # #     qrels, {'map_cut_5', 'ndcg_cut_5'})
# # # help(pytrec_eval.RelevanceEvaluator)
# repeat = [2,4,12,14,15]
# repeat = [0,2,3,5,1]
# repeat = [1,12,3,10,6]
# repeat = [8,1,2,14,5]
repeat = [0]
# repeat = [0,1,2,3,4]
# repeat = [1,3,5,7,12]
# repeat = [i for i in range(5)]
result2 = {}

for m in metrics:
    result2[m] = torch.zeros(len(repeat))

for i in range(len(repeat)):
    with open('../data/google/Annotators_split/test.json') as f:
        qrels = json.load(f)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {'map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10', })
    # filename = f'../data/ntcir/rm3_50_5.json'
    # filename = f'../DKG/result/ntcir_normalize/interpolated_rm3_40_5_{i}.json'
    # filename = f'result/HINormer/run_ntcir_ablation_theme_rm3_40_5_1_1_80_1e-05_10_1_{repeat[i]}.json'
    # filename = f'../data/google/rerank_union.json'
    filename = f'../DKG/result/google_normalize/interpolated_HHGT_Annotators_split_rerank_union_1.json'
    # filename = f'result/HINormer/run_ntcir_metadata_content_stella_fsdm_1_1_80_1e-05_10_1_{repeat[i]}.json'
    # filename = f'result/HINormer/run_ntcir_tfidf_3_5_40_0.0001_20_128_{i}.json'
    # filename = f'result/HINormer/run_ntcir_metadata_tfidf_1_1_40_1e-05_10_2_{repeat[i]}.json'
    print(filename)
    with open(filename, 'r') as f:
        run = json.load(f)
    result = evaluator.evaluate(run)
    # print(result)
    result_hr = mrr(qrels, run, 5)
    results = {}
    # print(result)
    for metric in metrics:
        # print(x)
        results[metric] = sum([x[metric] for x in result.values()]) / len(result)
    # print(results)
    for m in metrics:
        result2[m][i] = results[m]
for metric in metrics:
    print(f"{metric}: {result2[metric].mean().item():.4f}  std: {result2[metric].std().item():.4f}", )

print("========================================")

# T Test
import scipy.stats as stats
repeat1 = repeat
metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
# repeat = [0,1,12,8,19]
repeat = [8,1,2,14,5]
result1 = {}
for m in metrics:
    result1[m] = torch.zeros(len(repeat))
result2 = {}
for m in metrics:
    result2[m] = torch.zeros(len(repeat))


def calculate_result(test_file, filename):
    result1 = {}
    for m in metrics:
        result1[m] = torch.zeros(repeat)
    with open(test_file) as f:
        qrels = json.load(f)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {'map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10', })
    with open(filename, 'r') as f:
        run = json.load(f)
    new_run = {}
    for k in run.keys():
        candidates = run[k]
        new_run[k] = {}
        for dataset in candidates:
            if dataset == 'DATASET_00000000':
                dataset_node = '0'
            else:
                dataset_node = dataset.lstrip('DATASET_0')
            new_run[k][dataset_node] = candidates[dataset]
    result = evaluator.evaluate(new_run)
    results = {}
    for metric in metrics:
        # print(x)
        results[metric] = sum([x[metric] for x in result.values()]) / len(result)
    # print(results)
    for m in metrics:
        if '10' in m:
            print(f"{m}: {results[m]:.4f}")
    # for m in metrics:
    #     result1[m][number] = results[m]
    return results


bge_result = results
print(bge_result)
for i in range(len(repeat)):
    # filename = f'result/run_FAERY_3_1_10_5e-05_128_qc_{i+4}.json'
    # filename = f"result/HINormer/run_ntcir_metadata_content_stella_bm25_1_1_80_1e-05_10_1_{repeat[i]}.json"
    # filename = '../data/ntcir/rm3_40_5.json'
    filename = f'result/HINormer/run_ntcir_rm3_40_5_1_1_80_1e-05_10_1_{repeat[i]}.json'
    # filename = f'../DKG/result/ntcir_metadata_content/run_ntcir_metadata_content_tfidf_{i}.json'
    results = calculate_result('../data/ntcir/test.json', filename)
    # with open(filename, 'r') as f:
    #     results = json.load(f)
    for m in metrics:
        result1[m][i] = results[m]
    # filename = f'../hinormer_modified/result/simpleHGN/result_Datafinder_4_0.0005_512_qc_{i}.json'
    # filename = f'result/run_DSEBench_3_1_10_5e-05_128_10_5_tq_{i}.json'
    # filename = f"result/run_Datafinder_3_3_10_0.0001_128_qc_{i}.json"
    filename = f'result/HINormer/run_ntcir_metadata_content_stella_rm3_40_5_1_1_80_1e-05_10_1_{repeat1[i]}.json'
    # filename = '../data/ntcir/bm25.json'
    # filename = f'../DKG/result/ntcir_ablation_publish_normalize/interpolated_tfidf_{i}.json'
    results = calculate_result('../data/ntcir/test.json', filename)
    # with open(filename, 'r') as f:
    #     results = json.load(f)
    #     # print(filename)
    #     # print(results.keys())
    for m in metrics:
        result2[m][i] = results[m]
for m in metrics:
    print(m)
    print(result1[m], result2[m])
    print(f'{result1[m].mean().item():.4f}', f'{result2[m].mean().item():.4f}')
    t_stat, p_value = stats.ttest_ind(result1[m], result2[m])

    print(f"t值: {t_stat}")
    print(f"P值: {p_value}")
    print("=="*10)


# repeat = [0]
# result2 = {}
# metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10',]
#
# for m in metrics:
#     result2[m] = torch.zeros(len(repeat))
#
# for i in range(len(repeat)):
#     qrels = {}
#     for j in range(5):
#         with open(f'../data/google/5-Fold_split/fold_{j}/test.json') as f:
#             qrels.update(json.load(f))
#     evaluator = pytrec_eval.RelevanceEvaluator(
#         qrels, {'map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10', })
#     # filename = f'../data/ntcir/rm3_50_5.json'
#     # filename = f'../DKG/result/ntcir_normalize/interpolated_rm3_40_5_{i}.json'
#     # filename = f'result/HINormer/run_ntcir_ablation_theme_rm3_40_5_1_1_80_1e-05_10_1_{repeat[i]}.json'
#     # filename = f'../data/google/rerank_union.json'
#     run_files = [
#         # '../data/google/rerank_union.json',
#         # 'result/HINormer/run_google_5-Fold_split-fold_0_rerank_union_2_1_10_0.0001_20_64_0.json',
#         # 'result/HINormer/run_google_5-Fold_split-fold_1_rerank_union_2_1_10_0.0001_20_64_0.json',
#         # 'result/HINormer/run_google_5-Fold_split-fold_2_rerank_union_2_1_10_0.0001_20_64_0.json',
#         # 'result/HINormer/run_google_5-Fold_split-fold_3_rerank_union_2_1_10_0.0001_20_64_0.json',
#         # 'result/HINormer/run_google_5-Fold_split-fold_4_rerank_union_2_1_10_0.0001_20_64_0.json'
#         # f'../DKG/result/google_normalize/interpolated_HINormer_5-Fold_split-fold_{j}_rerank_union_0.json' for j in range(5)
#         '../DKG/result/google_normalize/interpolated_HHGT_5-Fold_split-fold_0_rerank_union_1.json',
#         '../DKG/result/google_normalize/interpolated_HHGT_5-Fold_split-fold_1_rerank_union_0.json',
#         '../DKG/result/google_normalize/interpolated_HHGT_5-Fold_split-fold_2_rerank_union_1.json',
#         '../DKG/result/google_normalize/interpolated_HHGT_5-Fold_split-fold_3_rerank_union_1.json',
#         '../DKG/result/google_normalize/interpolated_HHGT_5-Fold_split-fold_4_rerank_union_0.json',
#     ]
#     run = {}
#     for run_file in run_files:
#         with open(run_file, 'r') as f:
#             run.update(json.load(f))
#     result = evaluator.evaluate(run)
#     # print(result)
#     results = {}
#     # print(result)
#     for metric in metrics:
#         # print(x)
#         results[metric] = sum([x[metric] for x in result.values()]) / len(result)
#     # print(results)
#     for m in metrics:
#         result2[m][i] = results[m]
# for metric in metrics:
#     print(f"{metric}: {result2[metric].mean().item():.4f}  std: {result2[metric].std().item():.4f}", )
#
# print("========================================")

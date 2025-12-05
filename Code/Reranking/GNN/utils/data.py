import pickle
import sys
import random
import torch
import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp


def load_data(prefix='ntcir', args=None):
    from data_loader import data_loader
    dl = data_loader('../data/' + prefix, args)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    train_data = []
    test_data = []
    val_data = []
    random.seed(2025)
    train_keys = list(dl.labels_train['data'].keys())
    random.shuffle(train_keys)
    for query_id in train_keys:
        candidate_lists = dl.labels_train['data'][query_id]
        for k, v in candidate_lists.items():
            train_data.append([query_id,
                               dl.labels_train['query_info'][query_id]['query_node_ids'],
                               eval(k),
                               v])
    print(train_data[0])
    print(train_data[1])
    # random.shuffle(train_data)
    for query_id, candidate_lists in dl.labels_val['data'].items():
        for k, v in candidate_lists.items():
            val_data.append([query_id,
                             dl.labels_val['query_info'][query_id]['query_node_ids'],
                             eval(k),
                             v])

    for query_id, candidate_lists in dl.labels_test['data'].items():
        for k, v in candidate_lists.items():
            test_data.append([query_id,
                              dl.labels_test['query_info'][query_id]['query_node_ids'],
                              eval(k),
                              v])
    # print('L2_2: ', len(dl.labels_test['data']['L2_2']))
    train_val_test = {'train': train_data, 'val': val_data, 'test': test_data}
    return features, \
        adjM, \
        train_val_test, \
        dl


def batch_data(data, batch_size, method='hinormer'):
    batches = []
    # assert len(data) > 0

    # assert len(data[0]) == 5
    if method == 'hhgt':
        batch = {'pair_id': [], 'query': [], 'dataset': [], 'dataset_id': [], 'labels': []}
    else:
        batch = {'pair_id': [], 'query': [], 'dataset': [], 'labels': []}

    for i in range(len(data)):
        batch['pair_id'].append(data[i][0])
        batch['query'].append(data[i][1])
        batch['dataset'].append(data[i][2])
        if method == 'hhgt':
            batch['dataset_id'].append(data[i][3])
            batch['labels'].append(data[i][4])
        else:
            batch['labels'].append(data[i][3])

        if (i + 1) % batch_size == 0:

            batches.append(batch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if method == 'hhgt':
                batch = {'pair_id': [], 'query': [], 'dataset': [], 'dataset_id': [], 'labels': []}
            else:
                batch = {'pair_id': [], 'query': [], 'dataset': [], 'labels': []}

    if len(batches) * batch_size < len(data):
        batches.append(batch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # print(batches[-1]['pair_id'])
    print(len(data), batch_size, len(batches))
    return batches


if __name__ == '__main__':
    load_data()

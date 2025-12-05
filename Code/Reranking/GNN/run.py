import argparse
import json
import os
import pickle
import random
import sys
import time
from tqdm import tqdm
import dgl
import numpy as np
import torch
import torch.nn.functional as F

from model import HINormer
from utils.data import load_data, batch_data
from utils.pytorchtools import EarlyStopping

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.append('utils/')

run_mode = 'train'


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
#
# # 设置随机数种子
# setup_seed(2025)

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def run_model_ntcir(args):
    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')
    if not os.path.exists('result/HINormer'):
        os.makedirs('result/HINormer', exist_ok=True)

    feats_type = args.feats_type
    features_list, adjM, train_val_test, dl = load_data(args.dataset, args)

    device = torch.device('cuda:' + str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    print(device)
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    # print("feature list", features_list)
    node_cnt = [features.shape[0] for features in features_list]
    # print(node_cnt)
    sum_node = 0
    for x in node_cnt:
        sum_node += x
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    train_data = train_val_test['train']
    val_data = train_val_test['val']
    test_data = train_val_test['test']
    g = dgl.DGLGraph(adjM)
    g = dgl.add_self_loop(g)
    all_nodes = np.arange(sum_node)
    node_seq = torch.zeros(sum_node, args.len_seq).long()

    n = 0
    empty_idx = []
    if os.path.exists(os.path.join(f'../data/{args.dataset}', f'seqs_{args.len_seq}.pickle')):
        node_seq = pickle.load(open(os.path.join(f'../data/{args.dataset}', f'seqs_{args.len_seq}.pickle'), 'rb'))
    else:
        with tqdm(total=len(all_nodes)) as pbar:
            for x in all_nodes:  # sample

                cnt = 0
                scnt = 0
                node_seq[n, cnt] = x
                cnt += 1
                start = node_seq[n, scnt].item()
                while (cnt < args.len_seq):
                    if start == -1:
                        node_seq[n, cnt] = -1
                        cnt += 1
                        if cnt == args.len_seq:
                            break
                        scnt += 1
                        start = node_seq[n, scnt].item()
                        continue
                    sample_list = g.successors(start).numpy().tolist()

                    if len(sample_list) == 0:
                        empty_idx.append(start)
                        node_seq[n, cnt] = -1
                        cnt += 1
                        if cnt == args.len_seq:
                            break
                        scnt += 1
                        start = node_seq[n, scnt].item()
                        continue
                    nsampled = max(len(sample_list), 1)
                    sampled_list = random.sample(sample_list, nsampled)
                    for i in range(nsampled):
                        node_seq[n, cnt] = sampled_list[i]
                        cnt += 1
                        if cnt == args.len_seq:
                            break
                    scnt += 1
                    start = node_seq[n, scnt].item()
                n += 1
                pbar.update(1)
        with open(os.path.join(f'../data/{args.dataset}', f'seqs_{args.len_seq}.pickle'), 'wb') as f:
            pickle.dump(node_seq, f)
    node_type = [i for i, z in zip(range(len(node_cnt)), node_cnt) for x in range(z)]
    print(node_type.count(0))
    print(node_type.count(1))
    g = g.to(device)
    for seq in node_seq:
        assert len(seq) == args.len_seq
    seqs = [[], [], []]
    for i, data_list in enumerate([train_data, val_data, test_data]):
        for d in data_list:
            query_seq = node_seq[d[1]][
                torch.squeeze(torch.nonzero(torch.count_nonzero(node_seq[d[1]] + 1, dim=1) == args.len_seq))]
            # contexts_seq = node_seq[d[2]][
            #     torch.squeeze(torch.nonzero(torch.count_nonzero(node_seq[d[2]] + 1, dim=1) == args.len_seq))]
            dataset_seq = node_seq[d[2]]
            if -1 in dataset_seq or len(query_seq) == 0:
                # print(d[0], dataset_seq[0])
                continue
            seqs[i].append([d[0], query_seq, dataset_seq, d[3]])

    train_batches = batch_data(seqs[0], args.batch_size)
    val_batches = batch_data(seqs[1], 512)
    test_batches = batch_data(seqs[2], 512)

    # num_classes = dl.labels_train['num_classes']
    num_classes = 1
    type_emb = torch.eye(len(node_cnt)).to(device)
    node_type = torch.tensor(node_type).to(device)
    print(type_emb)
    print(type_emb.shape)
    print(node_type)
    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    test_results = {}
    for m in metrics:
        test_results[m] = torch.zeros(args.repeat)
    for i in range(args.repeat):

        net = HINormer(g, num_classes, in_dims, args.hidden_dim, args.num_layers, args.num_gnns, args.num_heads,
                       args.dropout,
                       temper=args.temperature, num_type=len(node_cnt), beta=args.beta)

        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                       save_path='checkpoint/HINormer_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(
                                           args.dataset,
                                           args.fold.replace('/', '-'),
                                           args.mode,
                                           args.num_layers,
                                           args.num_gnns,
                                           args.len_seq,
                                           args.lr,
                                           args.top_k,
                                           args.batch_size, i))
        steps = 0
        early_stop = False
        for epoch in range(args.epoch):
            if run_mode == 'test':
                continue
            t_start = time.time()
            # training
            net.train()
            for batch in train_batches:
                logits = net(features_list, batch['query'], batch['dataset'], type_emb, node_type, args.l2norm)
                batch['labels'] = torch.tensor(batch['labels'], dtype=torch.float).to(device)
                # print(batch['labels'].shape)
                # train_loss = F.nll_loss(logp, batch['labels'])
                # print(logits)
                # train_loss = torch.nn.CrossEntropyLoss()(logits, batch['labels'])
                train_loss = F.binary_cross_entropy_with_logits(logits, batch['labels'])
                # logp = F.log_softmax(logits, 1)
                # print(logp, batch['labels'])

                # train_loss = F.nll_loss(logp, batch['labels'])
                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                t_end = time.time()

                # print training info
                print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
                    epoch, train_loss.item(), t_end - t_start))

                t_start = time.time()
                steps += 1
            # validation

            net.eval()
            print(f"*****************Eval {i}*****************")
            print(args)
            with torch.no_grad():
                preds = []
                labels = []
                qrels, run = {}, {}
                # with open(f'../data/{args.dataset}/test.json', 'r') as f:
                #     qrels = json.load(f)
                val_loss = 0
                for val_batch in val_batches:
                    logits = net(features_list, val_batch['query'], val_batch['dataset'], type_emb, node_type,
                                 args.l2norm)
                    # print(logits.shape)
                    val_batch['labels'] = torch.tensor(val_batch['labels'], dtype=torch.float).to(device)
                    val_loss += F.binary_cross_entropy_with_logits(logits, val_batch['labels'])
                    # val_loss += torch.nn.CrossEntropyLoss()(logits, val_batch['labels']).item()
                    # logp = F.log_softmax(logits, 1)
                    # val_loss = F.nll_loss(logp, val_batch['labels'])
                    # logits = logits.cpu().numpy().argmax(axis=1)
                    # print(logits)
                    # onehot = np.eye(num_classes, dtype=np.int32)
                    # pred = onehot[pred]
                    # print(val_batch['labels'][:20])
                    for id_ in range(len(val_batch['labels'])):
                        if val_batch['pair_id'][id_] not in qrels.keys():
                            qrels[val_batch['pair_id'][id_]] = {}
                        if val_batch['pair_id'][id_] not in run.keys():
                            run[val_batch['pair_id'][id_]] = {}
                        label_score = val_batch['labels'][id_].cpu().numpy().tolist()
                        dataset_id = val_batch['dataset'][id_].cpu().numpy().tolist()[0]
                        # pred_score = logits[id_].tolist()
                        pred_score = logits[id_].cpu().numpy().tolist()
                        qrels[val_batch['pair_id'][id_]][str(dataset_id)] = int(label_score)
                        run[val_batch['pair_id'][id_]][str(dataset_id)] = pred_score
                    # print(run)
                eval_result = dl.evaluate_valid(qrels, run, metrics)
                print(eval_result)

            scheduler.step(val_loss)
            print(scheduler.get_last_lr())
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss, t_end - t_start))
            # early stopping
            early_stopping(val_loss, eval_result, net)
            early_stop = early_stopping.early_stop
            if early_stop:
                print('Early stopping!')
                break
            # if early_stop:
            #     print('Early stopping!')
            #     break

        # testing with evaluate_results_nc
        # args.mode = 'bm25'
        net.load_state_dict(torch.load(
            'checkpoint/HINormer_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(args.dataset, args.fold.replace('/', '-'),
                                                                          args.mode, args.num_layers,
                                                                          args.num_gnns,
                                                                          args.len_seq, args.lr, args.top_k,
                                                                          args.batch_size, i)))
        net.eval()
        with torch.no_grad():
            qrels, run = {}, {}
            with open(f'../data/{args.dataset}/{args.fold}/test.json', 'r') as f:
                qrels = json.load(f)
            for batch in test_batches:
                logits = net(features_list, batch['query'], batch['dataset'], type_emb, node_type, args.l2norm)
                test_logits = logits.cpu().numpy()
                # test_logits = logits.cpu().numpy().argmax(axis=1)
                test_logits = logits.cpu().numpy()
                # test_logits = torch.nn.Sigmoid()(logits).cpu().numpy()
                for id_ in range(len(batch['labels'])):
                    if batch['pair_id'][id_] not in run.keys():
                        run[batch['pair_id'][id_]] = {}
                    dataset_id = batch['dataset'][id_].cpu().numpy().tolist()[0]
                    pred_score = test_logits[id_].tolist()
                    # pred_score = test_logits[id_].tolist()
                    run[batch['pair_id'][id_]][str(dataset_id)] = pred_score
            result = dl.evaluate_valid(qrels, run, metrics)
            print(f"Repeat: {i}")
            for metric in metrics:
                test_results[metric][i] = result[metric]
                print(f'{metric}: {result[metric]:.4f}', end='\t')
            print('\n')
            with open('result/HINormer/result_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(args.dataset,
                                                                                         args.fold.replace('/', '-'),
                                                                                         args.mode,
                                                                                         args.num_layers,
                                                                                         args.num_gnns,
                                                                                         args.len_seq, args.lr,
                                                                                         args.top_k,
                                                                                         args.batch_size, i),
                      'w') as f:
                json.dump(result, f)
            with open('result/HINormer/run_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(args.dataset,
                                                                                         args.fold.replace('/', '-'),
                                                                                         args.mode,
                                                                                         args.num_layers,
                                                                                         args.num_gnns,
                                                                                         args.len_seq, args.lr,
                                                                                         args.top_k,
                                                                                         args.batch_size, i),
                      'w') as f:
                json.dump(run, f)
    for metric in metrics:
        print(f"{metric}: {test_results[metric].mean().item():.4f}, std: {test_results[metric].std().item():.4f}", )


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='HINormer')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2' +
                         '4 - only term features (id vec for others);' +
                         '5 - only term features (zero vec for others).')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=256,
                    help='Dimension of the node hidden state. Default is 32.')
    ap.add_argument('--dataset', type=str, default='ntcir', help='DBLP, IMDB, Freebase, AMiner, DBLP-HGB, IMDB-HGB')
    ap.add_argument('--num-heads', type=int, default=2,
                    help='Number of the attention heads. Default is 2.')
    ap.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=20, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2, help='The number of layers of HINormer layer')
    ap.add_argument('--num-gnns', type=int, default=4,
                    help='The number of layers of both structural and heterogeneous encoder')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--retrieve-num', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--val-batch-size', type=int, default=256)
    ap.add_argument('--eval-steps', type=int, default=500)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=0)
    ap.add_argument('--len-seq', type=int, default=50, help='The length of node sequence.')
    ap.add_argument('--l2norm', type=bool, default=True, help='Use l2 norm for prediction')
    ap.add_argument('--mode', type=str, default="bm25", help='retrieval method: bm25/tfidf')
    ap.add_argument('--fold', type=str, default="", help='fold')
    ap.add_argument('--temperature', type=float, default=1.0, help='Temperature of attention score')
    ap.add_argument('--beta', type=float, default=1.0, help='Weight of heterogeneity-level attention score')

    args = ap.parse_args()
    print(args)
    run_model_ntcir(args)

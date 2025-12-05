import numpy as np
import torch
import scipy.sparse as sp
from collections import defaultdict
import dgl
from tqdm import tqdm
import torch.nn as nn


class KTRingNeighborhood:
    def __init__(self, graph, node_types, num_node_types, max_k=3):
        """
        graph: DGLGraph
        node_types: 每个节点的类型标签，形状为 [num_nodes] (numpy array or torch tensor)
        num_node_types: 节点类型总数
        max_k: 最大环数
        """
        self.graph = graph
        self.node_types = node_types if isinstance(node_types, np.ndarray) else node_types.numpy()
        self.num_node_types = num_node_types
        self.max_k = max_k

    def get_k_ring_neighborhood(self, target_node):
        """获取目标节点的k-ring邻域，返回结构完整的字典"""
        k_rings = {}

        # 初始化 0-ring: 所有类型默认为空，仅目标节点类型有值
        ring_0 = {t: [] for t in range(self.num_node_types)}
        target_type = self.node_types[target_node]
        ring_0[target_type] = [int(target_node)]  # 确保是 Python int
        k_rings[0] = ring_0

        # BFS for k >= 1
        visited = set([target_node])
        current_ring = set([target_node])

        for k in range(1, self.max_k + 1):
            next_ring = set()

            for node in current_ring:
                neighbors = self.graph.successors(node).numpy()
                for neighbor in neighbors:
                    if neighbor not in visited:
                        next_ring.add(int(neighbor))
                        visited.add(neighbor)

            # 初始化当前 k-ring：所有类型为空列表
            ring_k = {t: [] for t in range(self.num_node_types)}

            # 填充存在的节点
            for node in next_ring:
                node_type = self.node_types[node]
                if 0 <= node_type < self.num_node_types:
                    ring_k[node_type].append(node)
                else:
                    raise ValueError(f"Node {node} has invalid type {node_type}")

            k_rings[k] = ring_k
            current_ring = next_ring

        return k_rings

    def batch_get_k_t_rings(self, target_nodes):
        """批量获取多个目标节点的(k,t)-ring邻域"""
        batch_k_t_rings = []
        with tqdm(total=len(target_nodes), desc="Building k-t rings") as pbar:
            for node in target_nodes:
                k_rings = self.get_k_ring_neighborhood(int(node))
                batch_k_t_rings.append(k_rings)
                pbar.update(1)
        return batch_k_t_rings


class Ring2Token:
    def __init__(self, hidden_dim, num_node_types, max_k=3):
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        self.max_k = max_k

    def __call__(self, batch_k_t_rings, node_features):
        return self.forward(batch_k_t_rings, node_features)

    def forward(self, batch_k_t_rings, node_features):
        batch_tokens = []
        device = node_features.device
        with tqdm(total=len(batch_k_t_rings), desc="Building token2vec") as pbar:
            for i, k_t_rings in enumerate(batch_k_t_rings):
                # if i>20:
                #     break
                node_tokens = []
                # print(k_t_rings)
                # 遍历所有 k ∈ [0, max_k]
                for k in range(self.max_k + 1):
                    ring_tokens = []

                    # 现在 k_t_rings[k] 一定存在且包含所有类型
                    for t in range(self.num_node_types):
                        nodes = k_t_rings[k][t]  # 总是存在，可能为空列表
                        if len(nodes) > 0:
                            features = node_features[nodes]  # [L, hidden_dim]
                            token = features.mean(dim=0)  # [hidden_dim]
                        else:
                            token = torch.zeros(self.hidden_dim, device=device)
                        ring_tokens.append(token)

                    ring_token_seq = torch.stack(ring_tokens, dim=0)  # [num_types, hidden_dim]
                    node_tokens.append(ring_token_seq)

                node_tokens = torch.stack(node_tokens, dim=0)  # [max_k+1, num_types, hidden_dim]
                # print(node_tokens)
                batch_tokens.append(node_tokens)
                pbar.update(1)

        return torch.stack(batch_tokens, dim=0)  # [batch_size, max_k+1, num_types, hidden_dim]
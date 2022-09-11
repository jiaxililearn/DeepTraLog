import os
import torch
import numpy as np
from scipy import sparse as sp
from torch_geometric.data import Data, Batch
from ppr import PPR


class Subgraph:
    # Class for subgraph extraction

    def __init__(self, gid, data, path, maxsize=50, n_order=10):
        self.gid = gid
        self.x, self.edge_index, (_, self.edge_type), self.node_types = data

        self.path = path
        self.edge_num = self.edge_index[0].size(0)
        self.node_num = self.x.size(0)
        self.maxsize = maxsize

        self.ppr = PPR(gid, self.x, self.edge_index, n_order=n_order)

        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = {}

    def process_adj_list(self):
        for i in range(self.node_num):
            self.adj_list[i] = set()
        for i in range(self.edge_num):
            u, v = self.edge_index[0][i].item(), self.edge_index[1][i].item()
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)

    def adjust_edge(self, idx):
        # Generate edges for subgraphs
        dic = {}
        for i in range(len(idx)):
            dic[idx[i]] = i

        new_index = [[], []]
        nodes = set(idx)
        edge_types = []

        for i in idx:
            edge = list(self.adj_list[i] & nodes)

            print(edge)
            # resolve edge types
            if len(edge) > 0:
                cond = (self.edge_index[0] == i) & (
                    sum(self.edge_index[1] == k for k in edge).bool()
                )
                edge_types += self.edge_type[cond].tolist()

            edge = [dic[_] for _ in edge]
            # edge = [_ for _ in edge if _ > i]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
            
            print(len(new_index[0]), len(edge_types))
        return torch.LongTensor(new_index), torch.LongTensor(edge_types)

    def adjust_x(self, idx):
        # Generate node features for subgraphs
        return self.x[idx]

    def build(self):
        # Extract subgraphs for all nodes
        if (
            os.path.isfile(self.path + f"_subgraph{self.gid}.pt")
            and os.stat(self.path + f"_subgraph{self.gid}.pt").st_size != 0
        ):
            print("Exists subgraph file")
            self.subgraph = torch.load(self.path + f"_subgraph{self.gid}.pt")
            return

        self.neighbor = self.ppr.search_all(self.node_num, self.path)
        self.process_adj_list()
        for i in range(self.node_num):
            nodes = self.neighbor[i][: self.maxsize]
            x = self.adjust_x(nodes)
            edge, edge_types = self.adjust_edge(nodes)
            self.subgraph[i] = Data(x, edge, edge_types)
        torch.save(self.subgraph, self.path + f"_subgraph{self.gid}.pt")

    def search(self, node_list):
        # Extract subgraphs for nodes in the list
        batch = []
        index = []
        size = 0
        for node in node_list:
            batch.append(self.subgraph[node])
            index.append(size)
            size += self.subgraph[node].x.size(0)
        index = torch.tensor(index)
        batch = Batch().from_data_list(batch)
        return batch, index

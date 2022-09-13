import os
import torch
import numpy as np
from scipy import sparse as sp
from torch_geometric.data import Data, Batch
from ppr import PPR


class Subgraph:
    # Class for subgraph extraction

    def __init__(
        self, gid, data=None, path=None, maxsize=100, n_order=10, subgraph_path=None
    ):
        self.gid = gid
        self.x, self.edge_index, (_, self.edge_type), self.node_types = data

        self.path = path
        self.subgraph_path = path
        if subgraph_path:
            self.subgraph_path = subgraph_path

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

        tmp_edge_list = []
        row, col = self.edge_index
        for i in idx:
            edge = list(self.adj_list[i] & nodes)
            
            # TODO: resolve duplicate edge in both direction
            # resolve edge types after sampling
            if len(edge) > 0:
                cond = (
                    (row == i)
                    & (sum(col == k for k in edge).bool())
                ) | (
                    (col == i)
                    & (sum(row == k for k in edge).bool())
                )
                edge_types += self.edge_type[cond].tolist()

                new_row = [dic[_] for _ in row[cond].tolist()]
                new_col = [dic[_] for _ in col[cond].tolist()]
            
                # edge = [dic[_] for _ in edge]
                # edge = [_ for _ in edge if _ > i]
                new_index[0] += new_row
                new_index[1] += new_col

            print(f"new_index length: {len(new_index[0])}")
            print(f"edge_types length: {len(edge_types)}")
            print(f'i: {i}')
            print(new_index)
            print(edge_types)
            if len(new_index[0]) != len(edge_types):
                raise Exception('Not Matched')

        return torch.LongTensor(new_index), torch.LongTensor(edge_types)

    def adjust_x(self, idx):
        # Generate node features for subgraphs
        node_types_ = []
        for node in idx:
            for ntype, node_list in enumerate(self.node_types):
                if node in node_list:
                    node_types_.append(ntype)

        return self.x[idx], torch.LongTensor(node_types_)

    def build(self):
        # Extract subgraphs for all nodes
        if (
            os.path.isfile(self.subgraph_path + f"_subgraph{self.gid}.pt")
            and os.stat(self.subgraph_path + f"_subgraph{self.gid}.pt").st_size != 0
        ):
            # print("Exists subgraph file")
            self.subgraph = torch.load(self.subgraph_path + f"_subgraph{self.gid}.pt")
            return

        self.neighbor = self.ppr.search_all(self.node_num, self.path)
        self.process_adj_list()
        for i in range(self.node_num):
            nodes = self.neighbor[i][: self.maxsize]
            x, node_types = self.adjust_x(nodes)
            edge, edge_types = self.adjust_edge(nodes)
            self.subgraph[i] = Data(x, edge, edge_types, pos=node_types)
        torch.save(self.subgraph, self.subgraph_path + f"_subgraph{self.gid}.pt")

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

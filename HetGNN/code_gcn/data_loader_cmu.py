import os
import json
from re import L
from zipfile import ZipFile
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class CMUGraphDataset(Dataset):
    def __init__(self, data_root_path=None, transform=None, **kwargs):
        """
        node_feature_csv: path to the node feature csv file
        het_neigh_root: path to the het neighbour list root dir
        """
        super(CMUGraphDataset, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.transform = transform

        self.n_graphs = 600
        self.max_num_edge_embeddings = 12
        self.incoming_node_embedding_size = 26
        self.topk = 12

        self.data_root_path = data_root_path
        if data_root_path is None:
            self.data_root_path = '../custom_data_simple'

        self.relation_types_f = ["a_a_list.txt",
                                 "a_b_list.txt",
                                 "a_c_list.txt",
                                 "a_d_list.txt",
                                 "a_e_list.txt",
                                 "a_f_list.txt",
                                 "a_g_list.txt",
                                 "a_h_list.txt",
                                 "b_a_list.txt",
                                 "b_b_list.txt",
                                 "b_c_list.txt",
                                 "b_d_list.txt",
                                 "b_e_list.txt",
                                 "b_h_list.txt"]

        self.graph_edge_embedding = np.zeros(
            (len(self.relation_types_f), self.n_graphs, self.max_num_edge_embeddings, self.incoming_node_embedding_size))

        self.node_features = pd.read_csv(f'{self.data_root_path}/incoming_edge_embedding.csv')

        # load up feature matrix based on the relation types
        for relation_id, relation_f in enumerate(self.relation_types_f):
            print(f'Reading Relation Type File: {relation_f}')
            with open(f'{self.data_root_path}/{relation_f}', 'r') as fin:
                cnt = 0
                line = fin.readline()
                current_gid = -1
                current_src_id = -1
                i = -1  # aggregate top k src-neighs in the neigh for a graph
                while line:
                    part_ = line.strip().split(':')
                    gid = int(part_[0])
                    src_id = int(part_[1])
                    neigh_list = [int(i) for i in part_[2].split(',')]

                    # reset counters when a new graph reached
                    if current_gid != gid:
                        i = -1
                        current_src_id = -1
                        current_gid = gid
                    if current_src_id != src_id:
                        i += 1
                        current_src_id = src_id

                    if i >= self.topk:
                        print(f'Skip src-neigh list since limit reached k: {self.topk}')
                        continue 

                    for dst_id in neigh_list:
                        cond = (self.node_features['destination-id'] == dst_id) & (self.node_features['graph-id'] == gid)
                        self.graph_edge_embedding[relation_id][gid][i] += self.node_features[cond].values[:, 2:][0]

                    line = fin.readline()
                    cnt += 1
                    if cnt % 5000 == 0:
                        print(f'Processed {cnt} lines')
        print('done')

    # def read_graph(self, gid):
    #     """
    #     read a graph from disk
    #     """
    #     f_path = f'{self.het_neigh_root}/g{gid}.json'
    #     with open(f_path, 'r') as fin:
    #         g_het = json.load(fin)
    #     return g_het

    def __getitem__(self, gid):
        """
        get graph data on graph id
        return: (node_feature, graph_het_feature)
                node_feature: (n_node, n_feature)
                graph_het_feature: (n_neigh, n_node, topk, n_feature)
        """
        graph_node_feature = torch.from_numpy(
            self.node_feature_df[self.node_feature_df.trace_id == gid].iloc[:, 2:].values).float().to(self.device)

        edge_index = torch.from_numpy(
            self.edge_inedx_df[self.edge_inedx_df.trace_id == gid][['src_id', 'dst_id']].values.reshape(2, -1)
        ).type(torch.LongTensor).to(self.device)

        if self.include_edge_weight:
            edge_weight = torch.from_numpy(
                self.edge_inedx_df[self.edge_inedx_df.trace_id == gid]['weight'].values.reshape(-1,)
            ).float().to(self.device)
        else:
            edge_weight = None

        return graph_node_feature, edge_index, edge_weight, self.node_types[gid]


if __name__ == '__main__':
    dataset = CMUGraphDataset(
    )

    print(dataset[0])

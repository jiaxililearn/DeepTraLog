import json
from re import L
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset

class EventGraphDataset(Dataset):
    def __init__(self, node_feature_csv, het_neigh_root, node_type_csv, num_node_types=8, transform=None):
        """
        node_feature_csv: path to the node feature csv file
        het_neigh_root: path to the het neighbour list root dir
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_node_type = num_node_types
        self.topk = 10

        self.node_type_to_id = {chr(97 + i): i for i in range(num_node_types)}
        self.node_id_to_type = {i: chr(97 + i) for i in range(num_node_types)}

        print('reading node features..')
        self.node_feature_df = pd.read_csv(node_feature_csv).sort_values(['trace_id', 'node_id'])
        self.node_type_df = pd.read_csv(node_type_csv)

        self.het_neigh_root = het_neigh_root

        self.num_features = len(self.node_feature_df.columns) - 2

        # self.node_type_to_id = {chr(97 + i): i for i in range(self.num_node_type)}

        # self.het_neigh_dict = {}
        # print('reading het neighbour lists..')
        # for i in tqdm(range(9)):  # 9 files under the folder
        #     het_file_path = f'{het_neigh_root}/het_neigh_list_{i}.json'
        #     with open(het_file_path, 'r') as fin:
        #         _het_neigh_list = json.load(fin)
        #     self.het_neigh_dict.update(_het_neigh_list)

        self.transform = transform
        print('done')
    
    def read_graph(self, gid):
        """
        read a graph from disk
        """
        f_path = f'{self.het_neigh_root}/g{gid}.json'
        with open(f_path, 'r') as fin:
            g_het = json.load(fin)
        return g_het

    def __getitem__(self, gid):
        """
        get graph data on index
        return: (node_feature, graph_het_feature)
                node_feature: (n_node, n_feature)
                graph_het_feature: (n_neigh, n_node, topk, n_feature)
        """
        graph_node_feature = torch.from_numpy(
            self.node_feature_df[self.node_feature_df.trace_id == gid].iloc[:, 2:].values).float().to(self.device)

        graph_node_types = self.node_type_df[self.node_type_df.trace_id == gid]['node_type'].values
        
        het_neigh_dict = self.read_graph(gid)

        num_node = graph_node_feature.shape[0]
        graph_het_feature = torch.zeros(self.num_node_type, num_node, self.topk, self.num_features).to(self.device)

        for node, neigh_dict in het_neigh_dict.items():
            node_id = int(node[1:])

            for neigh_type, neigh_list in neigh_dict.items():
                neigh_type_id = self.node_type_to_id[neigh_type]
                for i, n in enumerate(neigh_list):
                    nid = int(n[1:])

                    graph_het_feature[neigh_type_id][node_id][i] = graph_node_feature[nid]

        # graph_data = {}
        # print(self.het_neigh_list.keys())
        # for node, neigh_dict in self.het_neigh_list[str(gid)].items():
        #     node_id = int(node[1:])
        #     neigh_tensor = torch.zeros(self.num_node_type, self.topk, self.num_features)

        #     for neigh_type, neigh_list in neigh_dict.items():
        #         for i, neigh in enumerate(neigh_list):
        #             neigh_id = int(neigh[1:])
        #             neigh_tensor[self.node_type_to_id[neigh_type]][i] = torch.from_numpy(self.node_feature_df[
        #                 (self.node_feature_df['trace_id'] == gid) & (self.node_feature_df['node_id'] == neigh_id)
        #             ].iloc[0, 2:].values)
        #     graph_data[node] = neigh_tensor
        return graph_node_feature, graph_het_feature, graph_node_types

if __name__ == '__main__':
    dataset = EventGraphDataset(
        '../ProcessedData_clean/node_feature_norm.csv',
        '../ProcessedData_clean/graph_het_neigh_list'
    )

    print(dataset[0])

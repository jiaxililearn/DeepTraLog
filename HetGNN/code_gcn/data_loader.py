import json
from re import L
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset

class EventGraphDataset(Dataset):
    def __init__(self, node_feature_csv, het_neigh_root, transform=None):
        """
        node_feature_csv: path to the node feature csv file
        het_neigh_root: path to the het neighbour list root dir
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print('reading node features..')
        self.node_feature_df = pd.read_csv(node_feature_csv).sort_values(['trace_id', 'node_id'])

        self.num_features = len(self.node_feature_df.columns) - 2
        self.num_node_type = 8
        self.topk = 10
        # self.node_type_to_id = {chr(97 + i): i for i in range(self.num_node_type)}

        self.het_neigh_dict = {}
        print('reading het neighbour lists..')
        for i in tqdm(range(9)):  # 9 files under the folder
            het_file_path = f'{het_neigh_root}/het_neigh_list_{i}.json'
            with open(het_file_path, 'r') as fin:
                _het_neigh_list = json.load(fin)
            self.het_neigh_dict.update(_het_neigh_list)

        self.transform = transform
        print('done')

    def __getitem__(self, gid):
        """
        get graph data on index
        return: (node_feature, het_neigh_list)
        """
        graph_node_feature = torch.from_numpy(
            self.node_feature_df[self.node_feature_df.trace_id == gid].iloc[:, 2:].values).float().to(self.device)
        het_neigh_dict = self.het_neigh_dict[str(gid)]
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
        return graph_node_feature, het_neigh_dict

if __name__ == '__main__':
    dataset = EventGraphDataset(
        '../ProcessedData_clean/node_feature_norm.csv',
        '../ProcessedData_clean/het_neigh_list'
    )

    print(dataset[0])

import os
import json
from re import L
from zipfile import ZipFile
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset


class HetGCNEventGraphDataset(Dataset):
    def __init__(
        self,
        node_feature_csv,
        edge_index_csv=None,
        node_type_txt=None,
        transform=None,
        ignore_weight=False,
        include_edge_type=False,
        ppr_zip_root_dir=None,
        **kwargs,
    ):
        """
        node_feature_csv: path to the node feature csv file
        het_neigh_root: path to the het neighbour list root dir
        """
        super(HetGCNEventGraphDataset, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.include_edge_weight = False
        self.include_edge_type = include_edge_type

        print("reading node features..")
        self.node_feature_df = pd.read_csv(node_feature_csv).sort_values(
            ["trace_id", "node_id"]
        )

        print("reading edge index..")
        self.edge_inedx_df = pd.read_csv(edge_index_csv)

        if ignore_weight:
            self.include_edge_weight = False
            print("Ignore Edge Weights.")
        else:
            if "weight" in self.edge_inedx_df.columns:
                self.include_edge_weight = True
                print("Found Edge Weights.")

        print("read node types ..")
        self.node_types = []

        with open(node_type_txt, "r") as fin:
            for line in fin.readlines():
                _node_types = json.loads(line)
                self.node_types.append(_node_types)
        print(f"node types txt: {len(self.node_types)}")

        if ppr_zip_root_dir:
            print("unzipping ppr subgraphs ..")
            with ZipFile(f"{ppr_zip_root_dir}/ppr_subgraphs.zip", "r") as zip_obj:
                zip_obj.extractall(path=ppr_zip_root_dir)

        # self.node_type_to_id = {chr(97 + i): i for i in range(self.num_node_type)}

        # self.het_neigh_dict = {}
        # print('reading het neighbour lists..')
        # for i in tqdm(range(9)):  # 9 files under the folder
        #     het_file_path = f'{het_neigh_root}/het_neigh_list_{i}.json'
        #     with open(het_file_path, 'r') as fin:
        #         _het_neigh_list = json.load(fin)
        #     self.het_neigh_dict.update(_het_neigh_list)

        self.transform = transform
        print("done")

    def read_graph(self, gid):
        """
        read a graph from disk
        """
        f_path = f"{self.het_neigh_root}/g{gid}.json"
        with open(f_path, "r") as fin:
            g_het = json.load(fin)
        return g_het

    def __getitem__(self, gid):
        """
        get graph data on graph id
        return: (node_feature, graph_het_feature)
                node_feature: (n_node, n_feature)
                graph_het_feature: (n_neigh, n_node, topk, n_feature)
        """
        graph_node_feature = (
            torch.from_numpy(
                self.node_feature_df[self.node_feature_df.trace_id == gid]
                .iloc[:, 2:]
                .values
            )
            .float()
            .to(self.device)
        )

        edge_index = (
            torch.from_numpy(
                self.edge_inedx_df[self.edge_inedx_df.trace_id == gid][
                    ["src_id", "dst_id"]
                ].values.reshape(2, -1)
            )
            .type(torch.LongTensor)
            .to(self.device)
        )

        if self.include_edge_weight:
            edge_weight = (
                torch.from_numpy(
                    self.edge_inedx_df[self.edge_inedx_df.trace_id == gid][
                        "weight"
                    ].values.reshape(
                        -1,
                    )
                )
                .float()
                .to(self.device)
            )
        else:
            edge_weight = None

        if self.include_edge_type:
            edge_type = (
                torch.from_numpy(
                    self.edge_inedx_df[self.edge_inedx_df.trace_id == gid][
                        "edge_type"
                    ].values.reshape(
                        -1,
                    )
                )
                .float()
                .to(self.device)
            )
            return (
                graph_node_feature,
                edge_index,
                (edge_weight, edge_type),
                self.node_types[gid],
            )

        else:
            return graph_node_feature, edge_index, (edge_weight, None), self.node_types[gid]


if __name__ == "__main__":
    dataset = HetGCNEventGraphDataset(
        "../ProcessedData_clean/node_feature_norm.csv",
        "../ProcessedData_clean/graph_het_neigh_list",
    )

    print(dataset[0])

import torch
from cytoolz import curry
from torch_geometric.data import Data, Batch
from data_loader import HetGCNEventGraphDataset
from subgraph import Subgraph
from ppr import PPR
import multiprocessing as mp


class SubgraphGenerator(object):
    def __init__(self):
        self.data_root_dir = "../ProcessedData_HetGCN"
        self.ppr_path = "../ProcessedData_HetGCN/ppr_neighbours/ppr"

        self.dataset = HetGCNEventGraphDataset(
            node_feature_csv=f"{self.data_root_dir}/node_feature_norm.csv",
            edge_index_csv=f"{self.data_root_dir}/edge_index.csv",
            node_type_txt=f"{self.data_root_dir}/node_types.txt",
            ignore_weight=True,
            include_edge_type=True,
        )

    def process_ppr(self, gid):
        x, edge_index, (_, edge_type), node_types = self.dataset[gid]
        ppr = PPR(gid, x, edge_index, n_order=10)
        ppr.search_all(x.shape[0], self.ppr_path)
    
    def run(self):
        with mp.Pool() as pool:
            list(
                pool.imap_unordered(self.process_ppr, list(range(1000)), chunksize=500)
            )

if __name__ == '__main__':
    sub = SubgraphGenerator()
    sub.run()
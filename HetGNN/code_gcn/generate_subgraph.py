import torch
from tqdm import tqdm
from cytoolz import curry
from torch_geometric.data import Data, Batch
from data_loader import HetGCNEventGraphDataset
from subgraph import Subgraph
from ppr import PPR

# import multiprocessing as mp
from threading import Thread


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

    def process_ppr(self, gids):
        for gid in tqdm(gids):
            x, edge_index, (_, edge_type), node_types = self.dataset[gid]
            ppr = PPR(gid, x, edge_index, n_order=10)
            ppr.search_all(x.shape[0], self.ppr_path)

    def run(self):
        chunk_size = 500
        num_nodes = 1000  # 132485
        threads = [
            Thread(target=self.process_ppr, args=(list(range(i, i + chunk_size))))
            for i in range(0, num_nodes, chunk_size)
        ]

        # start the threads
        for thread in threads:
            thread.start()

        # wait for the threads to complete
        for thread in threads:
            thread.join()


if __name__ == "__main__":
    sub = SubgraphGenerator()
    sub.run()

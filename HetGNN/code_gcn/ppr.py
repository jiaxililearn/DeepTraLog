import os
import torch
from sklearn.preprocessing import normalize, StandardScaler
import numpy as np
import multiprocessing as mp
from scipy import sparse as sp
from cytoolz import curry


class PPR:
    # Node-wise personalized pagerank
    def __init__(self, gid, x, edge_index, maxsize=200, n_order=2, alpha=0.85):
        self.gid = gid
        self.n_order = n_order
        self.maxsize = maxsize
        edge_num = edge_index.shape[1]
        node_num = x.shape[0]

        x = x.cpu().numpy()
        edge_index = edge_index.cpu().numpy()

        sp_adj = sp.csc_matrix(
            (np.ones(edge_num), (edge_index[0], edge_index[1])),
            shape=[node_num, node_num],
        )
        self.adj_mat = sp_adj
        self.P = normalize(self.adj_mat, norm="l1", axis=0)
        self.d = np.array(self.adj_mat.sum(1)).squeeze()

    def search(self, seed, alpha=0.85):
        x = sp.csc_matrix(
            (np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1]
        )
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)
        
        print(f'a: {x.data}')


        idx = scores.argsort()[::-1][: self.maxsize]
        neighbor = np.array(x.indices[idx])

        seed_idx = np.where(neighbor == seed)[0]
        if seed_idx.size == 0:
            neighbor = np.append(np.array([seed]), neighbor)
        else:
            seed_idx = seed_idx[0]
            neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]

        assert np.where(neighbor == seed)[0].size == 1
        assert np.where(neighbor == seed)[0][0] == 0

        return neighbor

    @curry
    def process(self, path, seed):
        ppr_path = os.path.join(path, "ppr{}".format(seed))
        if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
            # print("Processing node {}.".format(seed))
            neighbor = self.search(seed)
            torch.save(neighbor, ppr_path)
        else:
            print("File of node {} exists.".format(seed))

    def search_all(self, node_num, path):
        neighbor = {}
        if (
            os.path.isfile(path + f"_neighbor{self.gid}.pt")
            and os.stat(path + f"_neighbor{self.gid}.pt").st_size != 0
        ):
            print("Exists neighbor file")
            neighbor = torch.load(path + f"_neighbor{self.gid}.pt")
        else:
            # print("Extracting subgraphs")
            os.system("mkdir {}".format(path))
            with mp.Pool() as pool:
                list(
                    pool.imap_unordered(
                        self.process(path), list(range(node_num)), chunksize=1000
                    )
                )

            # print("Finish Extracting")
            for i in range(node_num):
                neighbor[i] = torch.load(os.path.join(path, "ppr{}".format(i)))
            torch.save(neighbor, path + f"_neighbor{self.gid}.pt")
            os.system("rm -r {}".format(path))
            # print("Finish Writing")
        return neighbor

# import six.moves.cPickle as pickle
import numpy as np
import pandas as pd
import os
import string
import json
import re
import random
import math
from collections import Counter, defaultdict
from itertools import *
from sklearn.preprocessing import normalize
from scipy import sparse
import pickle


class input_data(object):
    def __init__(self, args):
        self.args = args

        self.n_nodes = 863
        self.max_num_edge_embeddings = 150
        self.n_graphs = 132485

        # Creating neighbour embedding based on above edge embeddings
        list_train = {}
        # relation_f = ['a_a_list.txt', 'a_b_list.txt', 'a_c_list.txt',
        #               'a_d_list.txt', 'a_e_list.txt', 'a_f_list.txt',
        #               'a_g_list.txt', 'a_h_list.txt', 'b_a_list.txt',
        #               'b_b_list.txt', 'b_c_list.txt', 'b_d_list.txt',
        #               'b_e_list.txt', 'b_h_list.txt']
        relation_f = [
            '0_0_list.txt',
            '0_1_list.txt',
            '0_2_list.txt',
            '0_3_list.txt',
            '0_4_list.txt',
            '0_5_list.txt',
            '0_6_list.txt',
            '0_7_list.txt',
            '1_0_list.txt',
            '1_1_list.txt',
            '1_2_list.txt',
            '1_3_list.txt',
            '1_4_list.txt',
            '1_5_list.txt',
            '1_6_list.txt',
            '1_7_list.txt',
            '2_0_list.txt',
            '2_1_list.txt',
            '2_2_list.txt',
            '2_3_list.txt',
            '2_4_list.txt',
            '2_5_list.txt',
            '2_6_list.txt',
            '2_7_list.txt',
            '3_0_list.txt',
            '3_1_list.txt',
            '3_2_list.txt',
            '3_3_list.txt',
            '3_4_list.txt',
            '3_5_list.txt',
            '3_6_list.txt',
            '3_7_list.txt',
            '4_0_list.txt',
            '4_1_list.txt',
            '4_2_list.txt',
            '4_3_list.txt',
            '4_4_list.txt',
            '4_5_list.txt',
            '4_6_list.txt',
            '4_7_list.txt',
            '5_0_list.txt',
            '5_1_list.txt',
            '5_2_list.txt',
            '5_3_list.txt',
            '5_4_list.txt',
            '5_5_list.txt',
            '5_6_list.txt',
            '5_7_list.txt',
            '6_0_list.txt',
            '6_1_list.txt',
            '6_2_list.txt',
            '6_3_list.txt',
            '6_4_list.txt',
            '6_5_list.txt',
            '6_6_list.txt',
            '6_7_list.txt',
            '7_2_list.txt',
            '7_4_list.txt',
            '7_5_list.txt'
        ]

        for f_name in relation_f:
            print(f"Reading relation files {f_name}")
            with open(os.path.join(self.args.data_path, f_name), "r") as fin:
                src_type = f_name.split("_")[0]
                dst_type = f_name.split("_")[1]

                relation_type = f"{src_type}_{dst_type}"
                if relation_type not in list_train.keys():
                    list_train[relation_type] = {}

                for i, line in enumerate(fin):

                    if (i + 1) % 5000 == 0:
                        print(f"\tProcessed {i} lines")

                    line_part = line.strip().split(":")

                    gid = int(line_part[0])
                    src_node_id = int(line_part[1])

                    neigh_list = line_part[2].split(",")

                    if gid not in list_train[relation_type].keys():
                        list_train[relation_type][gid] = defaultdict(list)

                    for neigh in neigh_list:
                        list_train[relation_type][gid][src_node_id].append(int(neigh))

        self.list_train = list_train
        # self.a_a_list_train = list_train["a_a"]
        # self.a_b_list_train = list_train["a_b"]
        # self.a_c_list_train = list_train["a_c"]
        # self.a_d_list_train = list_train["a_d"]
        # self.a_e_list_train = list_train["a_e"]
        # self.a_f_list_train = list_train["a_f"]
        # self.a_g_list_train = list_train["a_g"]
        # self.a_h_list_train = list_train["a_h"]

        # self.b_a_list_train = list_train["b_a"]
        # self.b_b_list_train = list_train["b_b"]
        # self.b_c_list_train = list_train["b_c"]
        # self.b_d_list_train = list_train["b_d"]
        # self.b_e_list_train = list_train["b_e"]
        # self.b_h_list_train = list_train["b_h"]

        # node-edge embedding
        node_edge_embedding_filename = os.path.join(
            self.args.data_path, "node_embedding.csv" #"incoming_edge_embedding.csv"
        )

        print(f"Reading Node Edge Embedding file {node_edge_embedding_filename}")
        node_edge_embeddings = pd.read_csv(node_edge_embedding_filename).to_numpy()
        # with open(node_edge_embedding_filename, 'rb') as fin:
        #     node_edge_embeddings = np.loadtxt(fin, delimiter=",", skiprows=1)

        graph_incoming_node_embedding = {}
        self.incoming_node_embedding_size = node_edge_embeddings.shape[1] - 2

        # Getting edge embedding for every node in every graph
        for row in node_edge_embeddings:
            gid = int(row[0])
            dst_id = int(row[1])

            if gid not in graph_incoming_node_embedding.keys():
                graph_incoming_node_embedding[gid] = np.zeros(
                    (self.n_nodes, self.incoming_node_embedding_size), dtype="float32"
                )
            graph_incoming_node_embedding[gid][dst_id] += row[2:]

        self.incoming_edge_embeddings = graph_incoming_node_embedding
        # print(graph_incoming_node_embedding[517248])

        # print(self.incoming_edge_embeddings[517248])

        # Create neighbour edge embeddings
        print("Creating Neighbour Edge Embeddings")
        # print(self.a_a_list_train)

        self.feature_list = []
        for k, v in list_train.items():
            print(f'Adding {k} to feature list')
            # self.feature_list.append(self.compute_edge_embeddings(v))

            # write each large sized feature list
            with open(f'{args.data_path}/feature_list/feature_list_{k}.pkl', 'wb') as fout:
                pickle.dump(self.compute_edge_embeddings(v), fout, protocol=4)
            
        # Getting the list of graph ids in the training set
        self.train_graph_id_list = list(self.incoming_edge_embeddings.keys())

    # compute edge embedding for every graph for every source node
    def compute_edge_embeddings(self, list_train_, topk=10):
        """
        compute the raw edge embedding feature
        """
        # fix the number of features encodings in each graph 
        # encoding_attr_size = topk + 1
        graph_encoding_list = []
        graph_ids = []
        # graph_edge_encoding = np.zeros(
        #     (
        #         self.n_graphs,
        #         self.max_num_edge_embeddings,
        #         self.incoming_node_embedding_size * encoding_attr_size,
        #     )
        # )
        for gid, neigh_dict in list_train_.items():
            # num_src_node = len(neigh_dict.keys())
            # graph_encoding_dict[gid] = np.zeros(
            #     (num_src_node, self.incoming_node_embedding_size * encoding_attr_size))

            for i, (src_id, neigh_list) in enumerate(neigh_dict.items()):
                # if i >= self.max_num_edge_embeddings:
                #     break
                vector_list = [self.incoming_edge_embeddings[gid][src_id]]
                for n in range(topk):
                    try:
                        if n >= len(neigh_list):
                            vector_list.append([0] * self.incoming_node_embedding_size)
                        else:
                            vector_list.append(self.incoming_edge_embeddings[gid][neigh_list[n]])
                    except Exception as e:
                        print(f'Error: {e}')
                        # print('Maybe Not Enough Neighbour .. Append 0s')
                graph_encoding_list.append(np.concatenate(vector_list, axis=None))
                graph_ids.append(gid)

        # Normalisation each feature
        # print(list_train_[0])
        # print(self.incoming_edge_embeddings[0])
        # print(graph_edge_encoding[0])
        print("Normalise processed node attributes ..")
        graph_encoding_list = np.array(graph_encoding_list)
        print(f'Encoding Shape: {graph_encoding_list.shape}')
        graph_encoding_list = sparse.csr_matrix(
            normalize(
                graph_encoding_list,
                axis=0)
        )
        # print(graph_edge_encoding[0])
        # print(graph_edge_encoding[0].shape)
        # print(graph_edge_encoding[0].sum())
        # print('After')
        # print(normalised_graph_encodings[0])
        # print(normalised_graph_encodings[0].shape)
        # print(normalised_graph_encodings[0].sum())
        return graph_encoding_list, graph_ids

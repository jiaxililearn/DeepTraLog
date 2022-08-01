# %%
# %%
from email.policy import default
import pandas as pd
import plotly.express as px
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import json


# %%

# %%
graph_data_path_root = '../GraphData'
output_root_dir = '../HetGNN/ProcessedData_clean'

# %%
with open(f'../HetGNN/ProcessedData_clean/trace_id_to_idx.json') as fin:
    trace_id_idx = json.loads(fin.read())
trace_id_idx


# %%
num_node_type = 8
node_type_maps = []
node_type_maps.append({i:chr(97+i) for i in range(num_node_type)})
node_type_maps.append({chr(97+i):i for i in range(num_node_type)})
node_type_maps

# %%
# %%
import numpy as np
import random
import json

# %%
def generate_random_walks(gid, graph, n_walks=5, max_walk_size=100):
    """
    generate random walks from the graph
    """
    # neighbour_node_size_limit = graph_neighbour_type_distribution(graph)
    # neighbour_node_size = defaultdict(int)

    graph_walks = {}

    for src_node in graph.keys():
        if len(graph[src_node]) < 1:
            continue
     
        walks = []
        for n_w in range(n_walks):
            # restart the walk
            current_node = src_node
            current_walk = []
            walk_size = 0

            while walk_size < max_walk_size:
                # print(walk_size)
                if current_node in graph.keys():
                    neigh_node = random.choice(list(graph[current_node]))
                else:
                    # print("reached the end of the walk, restart new walk.")
                    walks.extend(current_walk)
                    break

                # print((neigh_node, neigh_node_type))
                
                # restart the walk when cyclic
                if neigh_node in current_walk or neigh_node == current_node:
                    walks.extend(current_walk)
                    print('graph cycled, restart')
                    break

                current_walk.append(neigh_node)
                walk_size += 1
                # neighbour_node_size[neigh_node_type] += 1
                current_node = neigh_node
            walks.extend(current_walk)
        tmp_type_df = defaultdict(list)
        for n in walks:
            tmp_type_df[n[0]].append(n)
        graph_walks[src_node] = tmp_type_df
    
    top_graph_walks = top_frequent_neighbours(graph_walks)
    return top_graph_walks

# %%
def top_frequent_neighbours(graph_walks, top_n=10):
    """
    get top K frequent neighbours by its type
    """
    for node, neigh in graph_walks.items():
        for neigh_type, neigh_list in neigh.items():
            graph_walks[node][neigh_type] = [i for i, _ in Counter(neigh_list).most_common()[:top_n]]
    return graph_walks

# %%
# %%
if __name__ == '__main__':
    for idx in range(0, 9):
        het_neigh_list = {}
        with open(f'{graph_data_path_root}/process{idx}.jsons', 'r') as fin:
            for line in tqdm(fin.readlines()):
                
                graph = defaultdict(set)
                trace = json.loads(line)
                node_info = trace['node_info']
                gid = trace_id_idx[trace['trace_id']]
                for (src_id, dst_id), edge_type in zip(trace['edge_index'], trace['edge_attr']):
                    dst_type = node_type_maps[0][node_info[dst_id][4]]
                    src_type = node_type_maps[0][node_info[src_id][4]]
                    graph[f'{src_type}{src_id}'].add(f'{dst_type}{dst_id}')
                # print(graph)
                top_neigh_list = generate_random_walks(gid, graph)

                het_neigh_list[gid] = top_neigh_list
        with open(f'{output_root_dir}/het_neigh_list/het_neigh_list_{idx}.json', 'w') as fout:
            json.dump(het_neigh_list, fout)
        

# %%


# %%




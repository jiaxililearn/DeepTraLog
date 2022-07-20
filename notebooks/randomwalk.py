# %%
from email.policy import default
import pandas as pd
import plotly.express as px
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import json

# %%
graph_data_path_root = '../GraphData'
output_root_dir = '../HetGNN/ProcessedData_rw_top10'

# %%
with open(f'../HetGNN/ProcessedData/trace_id_idx.json') as fin:
    trace_id_idx = json.loads(fin.read())
# trace_id_idx

# %%
# trace_info = defaultdict(list)

# for idx in range(0, 9):
#     with open(f'{graph_data_path_root}/process{idx}.jsons', 'r') as fin:
#         for line in tqdm(fin.readlines()):
#             trace = json.loads(line)
#             trace_info['trace_id'].append(trace_id_idx[trace['trace_id']])
#             trace_info['trace_bool'].append(trace['trace_bool'])
#             trace_info['error_trace_type'].append(trace['error_trace_type'])
#             trace_info['process_idx'].append(idx)
#     break

# trace_info_df = pd.DataFrame(trace_info)
# trace_info_df

# %%
import numpy as np
import random

# %%
# n_nodes = 1000
# n_neighbours = 20

# %%
# graph = np.zeros((n_nodes, n_neighbours)) -1
# graph

# %%
import json

# %%
def write_relation_list_from_graph(gid, graph, node_type_map):
    """
    write to graph from raw graph edges
    """
    relation_list = {}
    # graph_neighbour_type_distribution(graph)
    # print(f'generating randomwalk for graph {gid}')
    walks_ = random_walk(graph, node_type_map=node_type_map)

    for src, neigh_list in walks_.items():
        src_type = node_type_map[src]
        for neigh in neigh_list:
            neigh_type = node_type_map[neigh]
            
            relation_f = f'{src_type}_{neigh_type}'

            # init dict if not exists
            if relation_f not in relation_list.keys():
                relation_list[relation_f] = defaultdict(list)

            relation_list[relation_f][src].append(neigh)
    write_relation_list(gid, relation_list)


def write_relation_list(gid, relation_list):
    """
    write to relation file
    """
    # print(f'writing relation file for graph {gid}')
    for relation_f, neigh_list in relation_list.items():
        with open(f'{output_root_dir}/{relation_f}_list.txt', 'a') as fout:
            for src, neighbours in neigh_list.items():
                neigh_str = ','.join([str(x) for x in neighbours])
                write_line = f'{gid}:{src}:{neigh_str}'
                fout.write(f'{write_line}\n')
    # print(f'Saved graph {gid}')


def graph_neighbour_type_distribution(graph, scale=100):
    """
    get the distribution of different neighbour types
    """
    neighbour_type_count = defaultdict(int)
    for src, dst in graph.items():
        for _, dst_type in dst:
            neighbour_type_count[dst_type] += 1
    
    sum_ = sum([v for _, v in neighbour_type_count.items()])

    for k in neighbour_type_count.keys():
        neighbour_type_count[k] = int(100 * neighbour_type_count[k] / sum_)
    
    return neighbour_type_count


def random_walk(graph, node_type_map=None, walk_size=100, walk_type='even'):
    """
    Randomw walk path from a graph
    using every node as starting node
    default size of the walk is 100
    do not allow cyclic walks, restarts on cyclic walks
    """
    n_walks = 5
    top_n = 10

    # neighbour_node_size_limit = graph_neighbour_type_distribution(graph)
    neighbour_node_size = defaultdict(int)

    graph_walks = defaultdict(list)

    for src_node in graph.keys():
        if len(graph[src_node]) < 1:
            continue
     
        walk_size = 0
        walks = []
        current_walk = []

        for n_w in range(n_walks):
            # restart the walk
            current_node = src_node

            while walk_size < 100:
                # print(walk_size)
                if current_node in graph.keys():
                    neigh_node, neigh_node_type = random.choice(list(graph[current_node]))
                else:
                    # print("reached the end of the walk, restart.")
                    walks.extend(current_walk)
                    current_walk = []
                    current_node = src_node
                    continue

                # print((neigh_node, neigh_node_type))
                
                # restart the walk when cyclic
                if neigh_node in current_walk or neigh_node == current_node:
                    walks.extend(current_walk)
                    current_walk = []
                    current_node = src_node
                    print('graph cycled, restart')
                    continue

                current_walk.append(neigh_node)
                walk_size += 1
                neighbour_node_size[neigh_node_type] += 1
                current_node = neigh_node
            walks.extend(current_walk)
            graph_walks[src_node].append(walks)
    
    for k in graph_walks.keys():
        # get top K
        graph_walks[k] = get_top_frequent_node_by_type(graph_walks[k], node_type_map=node_type_map, k=top_n)

    return graph_walks

def get_top_frequent_node_by_type(walks, node_type_map=None, k=10):
    """
    get top K neighbour of the nodes
    """
    node_cnt_dict = {} # defaultdict(int)
    for walk in walks:
        for w in walk:
            node_type = node_type_map[w]

            if node_type not in node_cnt_dict.keys():
                node_cnt_dict[node_type] = defaultdict(int)

            node_cnt_dict[node_type][w] += 1
    neigh_list = []
    for node_type, node_cnt in node_cnt_dict.items():
        sorted_node_cnt = sorted(node_cnt.items(), key=lambda x: x[1], reverse=True)[:10]
        # print(f'neigh type {node_type}: {sorted_node_cnt}')
        neigh_list.extend(sorted_node_cnt)
    return [n for n, _ in neigh_list]


# %%
if __name__ == '__main__':
    for idx in range(0, 9):
        with open(f'{graph_data_path_root}/process{idx}.jsons', 'r') as fin:
            for line in tqdm(fin.readlines()):
                
                graph = defaultdict(set)
                node_type_map = {}
                trace = json.loads(line)
                node_info = trace['node_info']
                gid = trace_id_idx[trace['trace_id']]
                for (src_id, dst_id), edge_type in zip(trace['edge_index'], trace['edge_attr']):
                    dst_type = node_info[dst_id][4]
                    src_type = node_info[src_id][4]
                    graph[src_id].add((dst_id, dst_type))
                    node_type_map[dst_id] = dst_type
                    node_type_map[src_id] = src_type
                
                write_relation_list_from_graph(gid, graph, node_type_map)
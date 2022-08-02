import json
import torch
from tqdm import tqdm

output_root_dir = '../HetGNN/ProcessedData_clean'
graph_output_path = f'{output_root_dir}/graph_het_neigh_list'

for i in range(1,9):
    het_file_path = f'{output_root_dir}/het_neigh_list/het_neigh_list_{i}.json'
    with open(het_file_path, 'r') as fin:
        _het_neigh_list = json.load(fin)

    for gid, het_neigh in tqdm(_het_neigh_list.items()):
        f_output_path = f'{graph_output_path}/g{gid}.json'
        with open(f_output_path, 'w') as fout:
            json.dump(het_neigh, fout)

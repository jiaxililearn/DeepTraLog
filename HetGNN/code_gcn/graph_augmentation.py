import random
import copy
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj, k_hop_subgraph

# RuntimeError: CUDA error: device-side assert triggered
# CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

class GraphAugmentator:
    def __init__(
        self,
        num_node_types=8,
        num_edge_types=1,
        edge_perturbation_method="xor",
        prior_dist=None,
        subgraph_ratio=0.01,
        insertion_iteration=1,
        node_insertion_method="target_to_source",
        swap_edge_pct=0.05,
        swap_node_pct=0.05,
        **kwarg,
    ):
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.edge_perturbation_method = edge_perturbation_method
        self.prior_dist = prior_dist
        
        self.subgraph_ratio = subgraph_ratio
        
        self.insertion_iteration = insertion_iteration
        self.node_insertion_method = node_insertion_method

        self.swap_node_pct = swap_node_pct
        self.swap_edge_pct = swap_edge_pct

        self.func_list = [
            self.create_het_node_insertion,
            self.create_het_edge_perturbation,
            self.create_node_type_swap,
            self.create_edge_type_swap
        ]

    def get_augment_func(self, augmentation_method):
        """
        getting augmentation functions
        """
        augment_func_dict = {
            "node_insertion": self.create_het_node_insertion,
            "edge_perturbation": self.create_het_edge_perturbation,
            "all": self.create_all_augmentations
        }
        return augment_func_dict[augmentation_method]
    
    def create_all_augmentations(self, batch_data):
        """
        create all augmentations at random
        """
        new_batch = []
        for i in range(len(batch_data)):
            g_data = random.choice(batch_data)
            func = random.choice(self.func_list)
            new_batch.extend(func([g_data]))
        return new_batch

    def create_het_edge_perturbation(self, batch_data):
        """
        generate the same number of 'synthetic' abnormal graphs as the input batch
        """
        new_batch = []
        for i in range(len(batch_data)):
            g_data = random.choice(batch_data)
            new_batch.append(
                self.het_edge_perturbation_from_prior(
                    g_data,
                    self.prior_dist,
                    num_node_types=self.num_node_types,
                    method=self.edge_perturbation_method,
                )
            )
        return new_batch

    def het_edge_perturbation_from_prior(
        self, g_data, prior_dist, num_node_types=8, method="xor", size=None
    ):
        """
        create edge perturbation based on prior distributions
        """
        node_feature, edge_index, (edge_weight, edge_type), node_types = g_data
        device = edge_index.device

        if size is None:
            size = node_feature.shape[0]

        total_num_edges = edge_index.shape[1]
        num_edge_types = len(prior_dist.keys())

        new_edge_index = []
        new_edge_type = []

        for etype in range(num_edge_types):
            for src_type in range(num_node_types):
                for dst_type in range(num_node_types):
                    src_dst_type = f"{src_type}_{dst_type}"

                    try:
                        edge_ratio = prior_dist[etype][src_dst_type]
                    except Exception as e:
                        # print(f"could not found key {src_dst_type} in edge type {etype}")
                        continue

                    src_node_list = node_types[src_type]
                    dst_node_list = node_types[dst_type]

                    if len(src_node_list) == 0 or len(dst_node_list) == 0:
                        continue
                    # print(f"src_node_list: {src_node_list}")
                    # print(f"dst_node_list: {dst_node_list}")

                    num_edges = int(edge_ratio * total_num_edges) + 1

                    sampled_edge_index = torch.tensor(
                        [
                            random.choices(src_node_list, k=num_edges),
                            random.choices(dst_node_list, k=num_edges),
                        ]
                    ).to(device)
                    sampled_edge_type = torch.tensor([etype] * num_edges).to(device)

                    new_edge_index.append(sampled_edge_index)
                    new_edge_type.append(sampled_edge_type)

        new_edge_index = torch.cat(new_edge_index, dim=1).long().view(2, -1)
        new_edge_type = (
            torch.cat(new_edge_type)
            .int()
            .view(
                -1,
            )
        )

        # print(f'new_edge_index: {new_edge_index.shape}')
        # print(f'new_edge_type: {new_edge_type.shape}')
        # print(f'num_edge_types: {num_edge_types}')

        generated_adj_matrix = to_dense_adj(
            new_edge_index, edge_attr=new_edge_type + 1, max_num_nodes=size
        ).view(size, -1)

        origin_adj_matrix = to_dense_adj(
            edge_index, edge_attr=edge_type + 1, max_num_nodes=size
        ).view(size, -1)

        mask = torch.logical_xor(origin_adj_matrix, generated_adj_matrix)

        new_adj_matrix = origin_adj_matrix.masked_fill(
            ~mask, 0
        ) + generated_adj_matrix.masked_fill(~mask, 0)

        new_edge_index, new_edge_type = dense_to_sparse(new_adj_matrix)
        new_edge_type -= 1
        return (
            node_feature,
            new_edge_index,
            (None, new_edge_type),
            node_types,
        )  # ignores edge weight for now. TODO: need to add support for CMU data

    # def het_edge_perturbation(
    #     g_data, num_node_types=8, method="xor", perturbation_prob=0.0002
    # ):
    #     """
    #     perturbate a graph with random dropping/adding edges
    #     """
    #     m = torch.distributions.Bernoulli(torch.tensor([perturbation_prob]))

    #     node_feature, edge_index, (edge_weight, edge_type), node_types = g_data
    #     num_nodes = node_feature.size()[0]
    #     new_adj_mat = torch.zeros(num_nodes, num_nodes).to(edge_index.device)

    #     if edge_type is None:
    #         edge_type = torch.zeros(
    #             edge_index.shape[1],
    #         ).to(edge_index.device)

    #     # original dense edge types, use edge_type + 1 to avoid misunderstanding with 0
    #     dense_edge_with_attr = to_dense_adj(edge_index, edge_attr=edge_type + 1).view(
    #         num_nodes, -1
    #     )

    #     random_het_adj_mat = (
    #         m.sample((num_nodes, num_nodes))
    #         .view((num_nodes, num_nodes))
    #         .to(node_feature.device)
    #     )

    #     for src_type in range(num_node_types):
    #         for dst_type in range(num_node_types):
    #             src_node_list = node_types[src_type]
    #             dst_node_list = node_types[dst_type]

    #             src_bool = torch.Tensor(
    #                 [True if i in src_node_list else False for i in range(0, num_nodes)]
    #             ).to(edge_index.device)
    #             dst_bool = torch.Tensor(
    #                 [True if i in dst_node_list else False for i in range(0, num_nodes)]
    #             ).to(edge_index.device)
    #             bool_mat = torch.matmul(
    #                 src_bool.view(-1, 1), dst_bool.view(1, -1)
    #             ).bool()

    #             masked_adj_mat = random_het_adj_mat.masked_fill(~bool_mat, 0)
    #             # print(f'{src_type}:{dst_type} - {masked_adj_mat.sum()}')

    #             # TODO: create edge type for the new edges <<<<<<<<<<<<
    #             het_edge_types = (
    #                 dense_edge_with_attr.masked_fill(~bool_mat, 0)
    #                 .unique()
    #                 .nonzero()
    #                 .flatten()
    #             )
    #             # print(
    #             #     f"het_edge_types: {het_edge_types.shape[0]}, masked_adj_mat: {masked_adj_mat.nonzero().shape[0]}"
    #             # )
    #             if (
    #                 het_edge_types.shape[0] != 0
    #                 and masked_adj_mat.nonzero().shape[0] != 0
    #             ):
    #                 masked_edge_index_, masked_edge_type_ = dense_to_sparse(
    #                     masked_adj_mat
    #                 )
    #                 masked_edge_type_ = het_edge_types.index_select(
    #                     0,
    #                     het_edge_types.float().multinomial(
    #                         masked_edge_type_.shape[0], replacement=True
    #                     ),
    #                 )

    #                 masked_adj_mat = to_dense_adj(
    #                     masked_edge_index_,
    #                     edge_attr=masked_edge_type_,
    #                     max_num_nodes=num_nodes,
    #                 ).view(num_nodes, -1)
    #             # TODO: From here

    #             new_adj_mat += masked_adj_mat

    #     mask = torch.logical_xor(dense_edge_with_attr, new_adj_mat)

    #     new_edge_index, new_edge_type = dense_to_sparse(
    #         dense_edge_with_attr.masked_fill(~mask, 0)
    #         + new_adj_mat.masked_fill(~mask, 0)
    #     )
    #     # print(new_edge_index.shape)

    #     # default no edge weights
    #     new_edge_weight = None

    #     # minus 1 to get back to original edge types
    #     new_edge_type -= 1
    #     return (
    #         node_feature,
    #         new_edge_index,
    #         (new_edge_weight, new_edge_type),
    #         node_types,
    #     )

    def create_het_node_insertion(self, batch_data):
        """
        create node insertion
        TODO: Add node deletion
        """
        new_batch = []
        for i in range(len(batch_data)):
            g_data = random.choice(batch_data)

            for iter_ in range(self.insertion_iteration):
                g_data = self.het_node_insertion(
                    g_data,
                    subgraph_ratio=self.subgraph_ratio,
                    method=self.node_insertion_method,
                )
            new_batch.append(g_data)
        return new_batch

    def het_node_insertion(
        self,
        g_data,
        subgraph_ratio=0.01,
        method="target_to_source",
        size=None,
    ):
        """
        add / removing node from the graphs
        It will also remove/add edges to the graph with the associated node
        """
        node_features, edge_index, (edge_weight, edge_type), node_types = g_data
        device = edge_index.device

        sampled_nodes = random.sample(
            range(node_features.shape[0]),
            int(node_features.shape[0] * subgraph_ratio) + 1,
        )

        try:
            _, sub_edge_index, _, sub_edge_mask = k_hop_subgraph(
                node_idx=sampled_nodes,
                num_hops=1,
                edge_index=edge_index,
                flow=method,
            )
        except Exception as e:
            print(f"sampled_nodes: {sampled_nodes}")
            raise Exception from e

        row, col = edge_index
        retained_edges = torch.stack([row[~sub_edge_mask], col[~sub_edge_mask]]).to(
            edge_index.device
        )
        retained_edge_types = edge_type[~sub_edge_mask]

        new_node_list = []

        # rewiring edge to new node
        last_node_id = node_features.shape[0] - 1
        for ntype, ntype_list in enumerate(node_types):
            if len(ntype_list) == 0:
                # print(f"skip node type {ntype}")
                continue
            _mask = sum(sub_edge_index[1] == i for i in ntype_list).bool()

            # skip if no node matched
            if _mask.sum() == 0:
                # print("skip mask")
                continue

            new_node_id = last_node_id + 1
            new_node_list.append((new_node_id, ntype))
            sub_edge_index[1] = sub_edge_index[1].masked_fill_(_mask, new_node_id)

            last_node_id = new_node_id
        new_edge_index = torch.cat([retained_edges, sub_edge_index], dim=1)

        # add new node to node feature matrix and node type list
        new_node_types = copy.deepcopy(node_types)
        new_node_features = [node_features]
        for new_node_id, new_node_type in new_node_list:
            new_node_feature = node_features[
                sample_one_node_from_list(node_types[new_node_type])
            ].view(1, -1)
            new_node_types[new_node_type].append(new_node_id)

            new_node_features.append(new_node_feature)
        new_node_features = torch.cat(new_node_features, dim=0)

        # add new edge type to the existing
        added_edge_types = edge_type[sub_edge_mask]
        new_edge_types = torch.cat([retained_edge_types, added_edge_types])

        return (
            new_node_features,
            new_edge_index,
            (None, new_edge_types),
            new_node_types,
        )  # default edge weight to None. TODO: update for CMU dataset

    def create_edge_type_swap(self, batch_data):
        """
        generate edge swap
        """
        new_batch = []
        for i in range(len(batch_data)):
            g_data = random.choice(batch_data)
            new_batch.append(self.edge_type_swap(g_data, swap_pct=self.swap_edge_pct))
        return new_batch

    def edge_type_swap(self, g_data, swap_pct=0.05):
        """
        swap edge types
        """
        node_features, edge_index, (edge_weight, edge_type), node_types = g_data
        unique_edge_types = torch.unique(edge_type.view(-1,))

        print(f'edge_type shape: {edge_type.shape}')
        # TODO: handle case when cannot augment the data
        # skip if no enough edge types
        if unique_edge_types.shape[0] < 2:
            return False
        sampled_indices = torch.multinomial(unique_edge_types.float(), 2).long()
        swap_edge_types = torch.index_select(unique_edge_types, 0, sampled_indices)

        # TODO: From here
        src_edge_indices = (edge_type.view(-1,) == swap_edge_types[0]).nonzero()
        dst_edge_indices = (edge_type.view(-1,) == swap_edge_types[1]).nonzero()
        
        num_edge_swap = int(min(
            src_edge_indices.shape[0] * swap_pct + 1,
            dst_edge_indices.shape[0] * swap_pct + 1,
        ))
        print(f'src_edge_indices: {src_edge_indices}')
        print(f'src_edge_indices: {dst_edge_indices}')

        swap_src = torch.multinomial(src_edge_indices.float(), num_edge_swap).long()
        swap_dst = torch.multinomial(dst_edge_indices.float(), num_edge_swap).long()

        new_edge_type = copy.deepcopy(edge_type)

        for a_edge, b_edge in zip(swap_src, swap_dst):
            self.swap_values(new_edge_type, a_edge, b_edge)
        
        print(f'new_edge_type shape: {new_edge_type.shape}')
        return node_features, edge_index, (edge_weight, new_edge_type.view(-1,)), node_types
    
    def swap_values(self, values, src_idx, dst_idx):
        tmp = values[src_idx]
        values[src_idx] = values[dst_idx]
        values[dst_idx] = tmp
        # return values

    def create_node_type_swap(self, batch_data):
        """
        create node swap
        """
        new_batch = []
        for i in range(len(batch_data)):
            g_data = random.choice(batch_data)
            new_batch.append(self.node_type_swap(g_data, swap_pct=self.swap_node_pct))
        return new_batch

    def node_type_swap(self, g_data, swap_pct=0.05):
        """
        Swap node types if valid
        """
        node_features, edge_index, (edge_weight, edge_type), node_types = g_data
        valid_node_types = [
            idx
            for idx, node_type_list in enumerate(node_types)
            if len(node_type_list) > 0
        ]
        if len(valid_node_types) >= 2:
            swap_node_types = random.sample(valid_node_types, 2)
        else:
            return False
        num_node_swap = int(min(
            len(node_types[swap_node_types[0]]) * swap_pct + 1,
            len(node_types[swap_node_types[1]]) * swap_pct + 1,
        ))

        swap_nodes = [
            random.sample(node_types[swap_node_types[0]], num_node_swap),
            random.sample(node_types[swap_node_types[1]], num_node_swap)
        ]

        new_node_types = copy.deepcopy(node_types)

        self.remove_from_list(new_node_types[swap_node_types[0]], swap_nodes[0])
        self.add_to_list(new_node_types[swap_node_types[0]], swap_nodes[1])
        self.remove_from_list(new_node_types[swap_node_types[1]], swap_nodes[1])
        self.add_to_list(new_node_types[swap_node_types[1]], swap_nodes[0])

        return node_features, edge_index, (edge_weight, edge_type), new_node_types

    def remove_from_list(self, src_list, removal):
        """
        remove from list
        """
        for i in removal:
            src_list.remove(i)

    def add_to_list(self, src_list, addition):
        """
        add to list
        """
        src_list.extend(addition)


def sample_one_node_from_list(node_list):
    """
    random sample a het node
    """
    return random.choice(node_list)


# class GraphAugementor:
#     """
#     Method for add/remove nodes/edges from the het graph
#     """

#     def __init__(self):
#         pass

#     def add_new_node(self):
#         """
#         add new node
#         """

#     def remove_node(self):
#         """
#         remove node
#         """

#     def add_new_edge(self):
#         """
#         add new edge
#         """

#     def remove_edge(self):
#         """
#         remove edge
#         """

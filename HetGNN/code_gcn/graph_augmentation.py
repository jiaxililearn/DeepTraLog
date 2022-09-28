import random
import torch
from torch_geometric.utils import (
    dense_to_sparse,
    to_dense_adj,
)


def create_het_edge_perturbation(
    batch_data, num_node_types=8, num_edge_types=1, method="xor"
):
    """
    generate the same number of 'fake' abnormal graphs as the input batch
    """
    new_batch = []
    for i in range(len(batch_data)):
        g_data = random.choice(batch_data)
        new_batch.append(
            het_edge_perturbation(
                g_data,
                num_node_types=num_node_types,
                method=method,
            )
        )
    return new_batch


def het_edge_perturbation(g_data, num_node_types=8, method="xor"):
    """
    perturbate a graph with random dropping/adding edges
    """
    perturbation_prob = 0.0002
    m = torch.distributions.Bernoulli(torch.tensor([perturbation_prob]))

    node_feature, edge_index, (edge_weight, edge_type), node_types = g_data
    num_nodes = node_feature.size()[0]
    new_adj_mat = torch.zeros(num_nodes, num_nodes).to(edge_index.device)

    # original dense edge types, use edge_type + 1 to avoid misunderstanding with 0
    dense_edge_with_attr = to_dense_adj(edge_index, edge_attr=edge_type + 1).view(
        num_nodes, -1
    )

    random_het_adj_mat = (
        m.sample((num_nodes, num_nodes))
        .view((num_nodes, num_nodes))
        .to(node_feature.device)
    )

    for src_type in range(num_node_types):
        for dst_type in range(num_node_types):
            src_node_list = node_types[src_type]
            dst_node_list = node_types[dst_type]

            src_bool = torch.Tensor(
                [True if i in src_node_list else False for i in range(0, num_nodes)]
            ).to(edge_index.device)
            dst_bool = torch.Tensor(
                [True if i in dst_node_list else False for i in range(0, num_nodes)]
            ).to(edge_index.device)
            bool_mat = torch.matmul(src_bool.view(-1, 1), dst_bool.view(1, -1)).bool()

            masked_adj_mat = random_het_adj_mat.masked_fill(~bool_mat, 0)
            # print(f'{src_type}:{dst_type} - {masked_adj_mat.sum()}')

            # TODO: create edge type for the new edges <<<<<<<<<<<<
            het_edge_types = (
                dense_edge_with_attr.masked_fill(~bool_mat, 0)
                .unique()
                .nonzero()
                .flatten()
            )
            # print(
            #     f"het_edge_types: {het_edge_types.shape[0]}, masked_adj_mat: {masked_adj_mat.nonzero().shape[0]}"
            # )
            if het_edge_types.shape[0] != 0 and masked_adj_mat.nonzero().shape[0] != 0:
                masked_edge_index_, masked_edge_type_ = dense_to_sparse(masked_adj_mat)
                masked_edge_type_ = het_edge_types.index_select(
                    0,
                    het_edge_types.float().multinomial(
                        masked_edge_type_.shape[0], replacement=True
                    ),
                )

                masked_adj_mat = to_dense_adj(
                    masked_edge_index_,
                    edge_attr=masked_edge_type_,
                    max_num_nodes=num_nodes,
                ).view(num_nodes, -1)
            # TODO: From here

            new_adj_mat += masked_adj_mat

    mask = torch.logical_xor(dense_edge_with_attr, new_adj_mat)

    new_edge_index, new_edge_type = dense_to_sparse(
        dense_edge_with_attr.masked_fill(~mask, 0) + new_adj_mat.masked_fill(~mask, 0)
    )
    # print(new_edge_index.shape)

    # default no edge weights
    new_edge_weight = None

    # minus 1 to get back to original edge types
    new_edge_type -= 1
    return node_feature, new_edge_index, (new_edge_weight, new_edge_type), node_types


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

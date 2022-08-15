from tqdm import tqdm
import torch
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, degree


class HetGCNConvSum(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types, hidden_channels=16, flow='target_to_source'):
        super(HetGCNConvSum, self).__init__(aggr='add', flow=flow)  # "Add" aggregation.
        self.in_channels = in_channels
        self.num_node_types = num_node_types
        self.hidden_channels = hidden_channels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.k = 12

        fc_node_content_layers = []
        fc_node_content_bias = []
        for _ in range(self.num_node_types):
            fc_node_content_layers.append(torch.nn.Linear(in_channels * self.k, hidden_channels, bias=False))
            fc_node_content_bias.append(Parameter(torch.Tensor(hidden_channels)))

        self.fc_node_content_layers = torch.nn.ModuleList(fc_node_content_layers)
        self.fc_node_content_bias = torch.nn.ParameterList(fc_node_content_bias)

        # self.lin1 = torch.nn.Linear(in_channels, self.hidden_channels, bias=False)
        self.lin2 = torch.nn.Linear(hidden_channels * num_node_types, out_channels, bias=False)
        # self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        # self.bias1 = Parameter(torch.Tensor(self.hidden_channels))
        self.bias2 = Parameter(torch.Tensor(out_channels))

        self.relu = torch.nn.LeakyReLU()
    
        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters from Layers and Parameters
        """
        for lin in self.fc_node_content_layers:
            lin.reset_parameters()
        self.lin2.reset_parameters()
        for bias in self.fc_node_content_bias:
            torch.nn.init.zeros_(bias)
        torch.nn.init.zeros_(self.bias2)
    
    def forward(self, batch_data):
        """
        forward method
        """
        # Step 3: compute Het Edge Index from node-type-based adjacancy matrices
        het_h_embeddings = []
        for ntype in range(self.num_node_types):
            _out = torch.zeros(len(batch_data), self.k * self.in_channels, device=self.device)

            for i, (node_feature, edge_index, edge_weight, node_types) in enumerate(batch_data):
                # self.het_edge_index(edge_index, edge_weight, node_types):
                # print(f'het_edge_index shape: {het_edge_index.shape}')
                # print(f'het_edge_weight shape: {het_edge_weight.shape}')

                _, het_edge_index, het_edge_weight = self.get_het_edge_index(edge_index, edge_weight, node_types, ntype)

                if het_edge_index is None:
                    # TODO: finer way for compute het hidden embedding when no neigh in this neigh type
                    _het_out = torch.zeros(node_feature.shape[0], self.hidden_channels, device=edge_index.device)
                else:
                    _het_out = self.propagate(het_edge_index, x=node_feature, edge_weight=het_edge_weight)

                # Concat the top K source node in graph
                mask = torch.sum(_het_out, 1).bool()

                if _het_out[mask].shape[0] > 0:
                    _out[i, :_het_out[mask].shape[0] * _het_out[mask].shape[1]] = _het_out[mask][:self.k].view(1, -1)
            
            _out = _out.view(len(batch_data), 1, self.k * self.in_channels)
            _out = torch.transpose(_out, 0, 1)

            # print(f'_out: {_out}')
            # _het_out = torch.sum(_het_out, 0)
            het_out = self.fc_node_content_layers[ntype](_out)
            het_out += self.fc_node_content_bias[ntype]
            het_out = self.relu(het_out).view(len(batch_data), self.hidden_channels)

            het_h_embeddings.append(het_out)

        combined_het_embedding = torch.cat(het_h_embeddings, 1).view(len(batch_data), self.hidden_channels * self.num_node_types)
        print(f'combined_het_embedding shape: {combined_het_embedding.shape}')
        # print(f'combined_het_embedding: {combined_het_embedding}')

        out = self.lin2(combined_het_embedding)
        out += self.bias2
        print(f'out shape: {out.shape}')
        return out

    # def forward(self, x, edge_index, node_types=None, edge_weight=None):
    #     """
    #     forward method
    #     """
    #     # x has shape [num_nodes, in_channels]
    #     # edge_index has shape [2, E]

    #     # Step 1: Norm and add self loop
    #     # edge_index, edge_weight = self._norm(edge_index, size=x.size(0), edge_weight=edge_weight)

    #     # node_types = [
    #     #   type0:[0,2,3,5]
    #     #   type1:[6,7,8]
    #     #   type2:[9]
    #     #   type3[10,11]
    #     # ]

    #     # x = self.lin(x)

    #     # Step 3: compute Het Edge Index from node-type-based adjacancy matrices
    #     het_h_embeddings = []
    #     for ntype, het_edge_index, het_edge_weight in self.het_edge_index(edge_index, edge_weight, node_types):
    #         # print(f'het_edge_index shape: {het_edge_index.shape}')
    #         # print(f'het_edge_weight shape: {het_edge_weight.shape}')

    #         if het_edge_index is None:
    #             # TODO: finer way for compute het hidden embedding when no neigh in this neigh type
    #             _het_out = torch.zeros(x.shape[0], self.hidden_channels, device=edge_index.device)
    #         else:
    #             # Step 2: Linearly transform node feature matrix. Neighbour type specific node feature hidden embedding
    #             # Step 3.1: propagate het message
    #             # het_edge_index, het_edge_weight = self._norm(het_edge_index,
    #             #                                              size=x.size(0),
    #             #                                              edge_weight=het_edge_weight,
    #             #                                              flow=self.flow)
    #             _het_out = self.propagate(het_edge_index, x=x, edge_weight=het_edge_weight)
    #         # print(f'Neigh Type {ntype}: {_het_out}')

    #         # print(f'{torch.sum(_het_out, 1)}')
    #         # print(f'{torch.sum(_het_out, 1).shape}')
    #         # Concat the top K source node in graph
    #         mask = torch.sum(_het_out, 1).bool()
    #         _out = torch.zeros(1, self.k * self.in_channels, device=edge_index.device)
    #         if _het_out[mask].shape[0] > 0:
    #             _out[0, :_het_out[mask].shape[0] * _het_out[mask].shape[1]] = _het_out[mask][:self.k].view(1, -1)
    #         _out = _out.view(1, 1, self.k * self.in_channels)
    #         _out = torch.transpose(_out, 0, 1)

    #         # print(f'_out: {_out}')
    #         # _het_out = torch.sum(_het_out, 0)
    #         het_out = self.fc_node_content_layers[ntype](_out)
    #         het_out += self.fc_node_content_bias[ntype]
    #         het_out = het_out.relu().view(1, self.hidden_channels)

    #         # print(f'het_out: {het_out}')

    #         # print(f'het_out shape: {het_out.shape}')

    #         het_h_embeddings.append(het_out)
    #     # print(f'het_h_embeddings shape: {het_h_embeddings.shape}')

    #     combined_het_embedding = torch.cat(het_h_embeddings, 1)
    #     # print(f'combined_het_embedding shape: {combined_het_embedding.shape}')
    #     # print(f'combined_het_embedding: {combined_het_embedding}')

    #     out = self.lin2(combined_het_embedding)
    #     out += self.bias2
    #     return out

    def get_het_edge_index(self, edge_index, edge_weight, node_types, ntype):
        """
        get het edge index by given type
        """
        row, col = edge_index
        if len(node_types[ntype]) == 0:
            return ntype, None, None
        het_mask = sum(col == i for i in node_types[ntype]).bool()
        return ntype, torch.stack([row[het_mask], col[het_mask]]), edge_weight[het_mask]

    def het_edge_index(self, edge_index, edge_weight, node_types):
        """
        return a generator of het neighbour edge indices
        """
        row, col = edge_index
        for ntype, n_list in enumerate(node_types):
            # print(f'col: {col}')
            # print(f'n_list: {n_list}')

            if len(n_list) == 0:
                yield ntype, None, None
                continue
            # TODO: look into the masking shape of the results
            het_mask = sum(col == i for i in n_list).bool()
            # print(f'het mask: {het_mask}')

            yield ntype, torch.stack([row[het_mask], col[het_mask]]), edge_weight[het_mask]

    def _norm(self, edge_index, size, edge_weight=None, flow='source_to_target'):
        assert flow in ["source_to_target", "target_to_source"]

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)

        edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_attr=edge_weight, num_nodes=size)

        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

        row, col = edge_index
        if flow == 'source_to_target':
            deg = scatter_add(edge_weight, col, dim=0, dim_size=size)
        else:
            deg = scatter_add(edge_weight, row, dim=0, dim_size=size)

        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight

    def message(self, x_j, edge_weight):
        # x_j has shape [num_edges, out_channels]
        return edge_weight.view(-1, 1) * x_j

    def update(self, inputs):
        # aggr_out has shape [num_nodes, out_channels]

        # Step 5: Return new node embeddings.
        return inputs

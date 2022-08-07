from tqdm import tqdm
import torch
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, degree

class HetGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(HetGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.num_node_types = 8
        self.hidden_channels = 16

        self.lin1 = torch.nn.Linear(in_channels, self.hidden_channels, bias=False)
        self.lin2 = torch.nn.Linear(self.hidden_channels * self.num_node_types, out_channels, bias=False)
        # self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        self.bias1 = Parameter(torch.Tensor(self.hidden_channels))
        self.bias2 = Parameter(torch.Tensor(out_channels))
    
        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters from Layers and Parameters
        """
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.zeros_(self.bias2)

    def forward(self, x, edge_index, node_types=None):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Norm and add self loop
        edge_index, edge_weight = self._norm(edge_index, size=x.size(0))

        # node_types = [
        #   type0:[0,2,3,5]
        #   type1:[6,7,8]
        #   type2:[9]
        #   type3[10,11]
        # ]

        # Step 2: Linearly transform node feature matrix.

        x = self.lin1(x)

        # Step 3: compute Het Edge Index from node-type-based adjacancy matrices
        het_h_embeddings = []
        for het_edge_index, het_edge_weight in self.het_edge_index(edge_index, edge_weight, node_types):

            if het_edge_index is None:
                het_out = torch.zeros(x.shape[0], self.hidden_channels, device=edge_index.device)
            else:
                # Step 3.1: propagate het message
                het_out = self.propagate(het_edge_index, edge_weight=het_edge_weight, size=(x.size(0), x.size(0)), x=x)

            het_out += self.bias1
            het_out = het_out.tanh()
            het_h_embeddings.append(het_out)

        combined_het_embedding = torch.cat(het_h_embeddings, 1)
        out = self.lin2(combined_het_embedding)
        out += self.bias2
        return out

    def het_edge_index(self, edge_index, edge_weight, node_types):
        """
        return a generator of het neighbour edge indices
        """
        row, col = edge_index
        for ntype, n_list in enumerate(node_types):
            # print(f'col: {col}')
            # print(f'n_list: {n_list}')

            if len(n_list) == 0:
                yield None, None
                continue

            het_mask = sum(col == i for i in n_list).bool()
            # print(f'het mask: {het_mask}')

            yield torch.stack([row[het_mask], col[het_mask]]), edge_weight[het_mask]

    def _norm(self, edge_index, size, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)

        edge_index, tmp_edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, num_nodes=size)

        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

        row, col = edge_index
        deg = scatter_add(edge_weight, col, dim=0, dim_size=size)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight

    def message(self, x_j, edge_weight, size):
        # x_j has shape [num_edges, out_channels]
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [num_nodes, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

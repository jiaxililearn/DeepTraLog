from tqdm import tqdm
import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, degree

class HetGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(HetGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        self.bias = Parameter(torch.Tensor(out_channels))

    def forward(self, x, edge_index):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Norm and add self loop
        edge_index, edge_weight = self._norm(edge_index, size=x.size(0))

        # print(f'is instance of tensor: {isinstance(edge_index, torch.Tensor)}')
        # print(f'dtype_long: {edge_index.dtype == torch.long}')
        # print(f'is_tensor: {torch.is_tensor(edge_index)}')
        # print(f'dim_2: {edge_index.dim() == 2}; {edge_index.dim()}')
        # print(f'size(0): {edge_index.size(0)}')

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        out = self.propagate(edge_index, edge_weight=edge_weight, size=(x.size(0), x.size(0)), x=x)
        out += self.bias
        return out

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

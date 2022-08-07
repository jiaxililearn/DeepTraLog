from tqdm import tqdm
import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class HetGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(HetGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

        self.bias = Parameter(torch.Tensor(out_channels))

    def forward(self, x, edge_index):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_weight = add_self_loops(edge_index, num_nodes=x.size(0))

        # print(f'is instance of tensor: {isinstance(edge_index, torch.Tensor)}')
        # print(f'dtype_long: {edge_index.dtype == torch.long}')
        # print(f'is_tensor: {torch.is_tensor(edge_index)}')
        # print(f'dim_2: {edge_index.dim() == 2}; {edge_index.dim()}')
        # print(f'size(0): {edge_index.size(0)}')

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        out += self.bias
        return out

    def message(self, x_j, edge_index, size):
        # x_j has shape [num_edges, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    # def aggregate(self, inputs, index, ptr=None, dim_size=None):
    #     """
    #     Step 4
    #     """

    def update(self, aggr_out):
        # aggr_out has shape [num_nodes, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

from turtle import forward
import torch
from torch import nn
from torch_geometric.nn import GCNConv

from config import relations

class HetGCN(nn.Module):
    def __init__(self):
        super(HetGCN, self).__init__()
        torch.manual_seed(42)

        self.out_embed_d = 16
        self.embed_d = 7
        self.num_neigh_relations = len(relations)

        # node feature encoder
        # self.conv1 = GCNConv(self.embed_d, self.embed_d)
        self.fc_node = nn.Linear(self.embed_d, self.embed_d)
        # self.conv2 = GCNConv(16, 7)

        # Neighbour Aggregation
        fc_neigh_agg_layers = []
        for i in range(self.num_neigh_relations):
            fc_neigh_agg_layers.append(nn.Linear(self.embed_d, self.out_embed_d))
        self.fc_neigh_agg_layers = nn.ModuleList(fc_neigh_agg_layers)

        # Heterogeneous Aggregation
        self.fc_het_neigh_agg = nn.Linear(self.out_embed_d * self.num_neigh_relations, self.out_embed_d)

        # Others
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_node_feature, x_het_neighbour_list, x_edge_index=None):
        """
        forward propagate based on graph and its node features, edges, and het neighbourhood
        """
        # TODO: Need to normalise on the input node features
        h = self.fc_node(x_node_feature)
        h = h.tanh()

        # TODO: Aggregate node on same neighbourhood type
        aggregated_neighbours = self.aggregate_neighbours(h, x_het_neighbour_list)

        # TODO: aggregate on all Het neighbourhood
        graph_embedding = self.aggregate_het_types(aggregated_neighbours)

        return graph_embedding

    def aggregate_neighbour(self, het_neighbour_list, x_edge_index=None):
        """
        aggregate on one neighbourhood type 
        """
        # TODO
        aggregated_neighbour = None
        return aggregated_neighbour

    def aggregate_neighbours(self, h_embed, het_neighbour_list):
        """
        aggregate on each neighbourhood type
        return a list of aggregated neighbour
        """
        aggregated_neighbours = []
        for relation_idx in range(self.num_neigh_relations):
            # TODO: for each relation type, get the neighbourhood aggregation
            # _neighbour_edge_index = self.get_neigbour_edge(x_edge_index, relation_idx)
            _het_neighbour_list = self.get_het_neighbour_list(het_neighbour_list, relation_idx)
            _aggregated_neighbour = self.aggregate_neighbour(h_embed, _het_neighbour_list)
            aggregated_neighbours.append(_aggregated_neighbour)
        return aggregated_neighbours
    
    def aggregate_het_types(self, aggregated_embeddings):
        """
        aggregate all the het types
        """
        # TODO
        het_graph_embedding = None
        return het_graph_embedding

    def get_neigbour_edge(self, edge_index, relation_idx):
        """
        get the corresponding neighbour edge index of the relation type
        """
        # TODO
        neighbour_edge_index = None
        return neighbour_edge_index

    def get_het_neighbour_list(self, het_neighbour_list, relation_idx):
        """
        get the corresponding neighbour edge index of the relation type
        """
        # TODO
        het_neighbour_list = None
        return het_neighbour_list

    @staticmethod
    def svdd_batch_loss():
        """
        Compute SVDD Loss on batch
        """
        # TODO
        loss = None
        return loss

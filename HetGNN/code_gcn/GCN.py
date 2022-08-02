from tqdm import tqdm
import torch
from torch import nn
# from torch_geometric.nn import GCNConv

from config import relations, node_types

class HetGCN(nn.Module):
    def __init__(self, model_path=None, **kwargs):
        super(HetGCN, self).__init__()
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.svdd_center = None
        self.model_path = model_path

        self.embed_d = kwargs['feature_size']
        self.out_embed_d = kwargs['feature_size']

        # self.num_neigh_relations = len(relations)
        self.num_node_types = len(node_types)
        self.node_type_to_id = {chr(97 + i): i for i in range(self.num_node_types)}
        self.node_id_to_type = {i: chr(97 + i) for i in range(self.num_node_types)}

        # node feature content encoder
        # self.conv1 = GCNConv(self.embed_d, self.embed_d)
        # self.conv2 = GCNConv(16, 7)
        fc_node_content_layers = []
        for i in range(self.num_node_types):
            fc_node_content_layers.append(nn.Linear(self.embed_d, self.out_embed_d))
        self.fc_node_content_layers = nn.ModuleList(fc_node_content_layers)

        # type-based Neighbour Aggregation
        # TODO: Currently Not Used
        # fc_neigh_agg_layers = []
        # for i in range(self.num_node_types):
        #     fc_neigh_agg_layers.append(nn.Linear(self.embed_d, self.out_embed_d))
        # self.fc_neigh_agg_layers = nn.ModuleList(fc_neigh_agg_layers)

        # Heterogeneous Aggregation
        self.fc_het_neigh_agg = nn.Linear(self.out_embed_d * (1 + self.num_node_types), self.out_embed_d)

        # Others
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    
    def init_weights(self):
        """
        init weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x_node_feature, x_het_neighbour_list, x_edge_index=None):
        """
        forward propagate based on graph and its node features, edges, and het neighbourhood
        """
        # TODO: Need to normalise on the input node features
        # h = self.fc_node(x_node_feature)
        # h = h.tanh()

        graph_node_het_embedding = self.node_het_embedding(x_node_feature, x_het_neighbour_list)
        # print(f'Node Het Embedding: {graph_node_het_embedding}')

        graph_embedding = self.graph_node_pooling(graph_node_het_embedding)
        # print(f'Graph Embedding: {graph_embedding}')
        return graph_embedding

    def node_het_embedding(self, h_embed, het_neighbour_list, x_edge_index=None):
        """
        compute graph embedding
        """
        graph_het_node_embedding = torch.zeros(h_embed.shape).to(self.device)

        for node, node_het_neigh in tqdm(het_neighbour_list.items()):
            node_type = self.node_type_to_id[node[0]]
            node_id = int(node[1:])
            het_neigh_embedding = torch.zeros(self.num_node_types + 1, self.out_embed_d).to(self.device)
            # print(node_het_neigh.keys())
            for neigh_type in range(self.num_node_types):
                if self.node_id_to_type[neigh_type] in node_het_neigh.keys():
                    neigh_list = [int(i[1:]) for i in node_het_neigh[self.node_id_to_type[neigh_type]]]
                else:
                    neigh_list = []

                if len(neigh_list) == 0:
                    continue

                neigh_features = h_embed[neigh_list]

                neigh_ = self.encode_node_content(neigh_features, neigh_type)
                neigh_ = self.relu(neigh_)

                neigh_aggregated = self.aggregate_neighbour(neigh_)

                het_neigh_embedding[neigh_type] = neigh_aggregated

            # adding self at the end of the neighbourhood
            het_neigh_embedding[-1] = self.relu(self.encode_node_content(h_embed[node_id], node_type))

            het_node_embedding = self.aggregate_het_neigh_types(het_neigh_embedding)
            het_node_embedding = self.sigmoid(het_node_embedding)

            graph_het_node_embedding[node_id] = het_node_embedding

        return graph_het_node_embedding

    def aggregate_neighbour(self, neigh_embedding):
        """
        Mean of neighbours in same neigh type
        """
        return torch.mean(neigh_embedding, 0)

    def encode_node_content(self, node_feature, node_type):
        """
        Aggregate the content of the node (input node features)
        """
        content_embedding = self.fc_node_content_layers[node_type](node_feature)
        return content_embedding

    def aggregate_het_neigh_types(self, het_neigh_embedding):
        """
        aggregate all the het types
        """
        node_het_embedding = self.fc_het_neigh_agg(
            het_neigh_embedding.view(1, (self.num_node_types + 1) * self.out_embed_d)
        )
        return node_het_embedding

    def graph_node_pooling(self, graph_node_het_embedding):
        """
        average all the node het embedding
        """
        return torch.mean(graph_node_het_embedding, 0)

    # def get_neigbour_edge(self, edge_index, neigh_type):
    #     """
    #     get the corresponding neighbour edge index of the node type
    #     """
    #     neighbour_edge_index = None
    #     return neighbour_edge_index

    # def get_het_neighbour_list(self, het_neighbour_list, neigh_type):
    #     """
    #     get the corresponding neighbour edge index of the node type
    #     """
    #     node_het_neighbour_list = {}
    #     for k, het_neigh in het_neighbour_list.items():
    #         node_het_neighbour_list[k] = het_neigh[neigh_type]

    #     return node_het_neighbour_list

    def set_svdd_center(self, center):
        """
        set svdd center
        """
        self.svdd_center = center

    @staticmethod
    def svdd_batch_loss(model, embed_batch, l2_lambda=0.001):
        """
        Compute SVDD Loss on batch
        """
        # TODO
        out_embed_d = model.out_embed_d
        l2_lambda = l2_lambda

        _batch_out = embed_batch
        _batch_out_resahpe = _batch_out.view(_batch_out.size()[0] * _batch_out.size()[1], out_embed_d)

        if model.model.svdd_center is None:
            with torch.no_grad():
                print('Set initial center ..')
                hypersphere_center = torch.mean(_batch_out_resahpe, 0)
                model.model.set_svdd_center(hypersphere_center)
                torch.save(hypersphere_center, f'{model.model_path}/HetGNN_SVDD_Center.pt')
        else:
            hypersphere_center = model.model.svdd_center

        dist = torch.square(_batch_out_resahpe - hypersphere_center)
        loss_ = torch.mean(torch.sum(dist, 1))

        l2_norm = sum(p.pow(2.0).sum() for p in model.model.parameters())

        loss = loss_ + l2_lambda * l2_norm
        return loss

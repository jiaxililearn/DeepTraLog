# from tqdm import tqdm
import torch
from torch import nn
# from torch_geometric.nn import GCNConv
from config import relations, node_types

class HetGCN_1(nn.Module):
    def __init__(self, model_path=None, feature_size=7, out_embed_s=32, **kwargs):
        super(HetGCN_1, self).__init__()
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.svdd_center = None
        self.model_path = model_path

        self.k = 10

        self.embed_d = feature_size
        self.out_embed_d = out_embed_s

        # self.num_neigh_relations = len(relations)
        self.num_node_types = len(node_types)
        self.node_type_to_id = {chr(97 + i): i for i in range(self.num_node_types)}
        self.node_id_to_type = {i: chr(97 + i) for i in range(self.num_node_types)}

        # node feature content encoder
        # self.conv1 = GCNConv(self.embed_d, self.embed_d)
        # self.conv2 = GCNConv(16, 7)
        fc_node_content_layers = []
        for i in range(self.num_node_types):
            fc_node_content_layers.append(nn.Linear(self.embed_d, self.embed_d))
        self.fc_node_content_layers = nn.ModuleList(fc_node_content_layers)

        # type-based Neighbour Aggregation
        fc_neigh_agg_layers = []
        for i in range(self.num_node_types):
            fc_neigh_agg_layers.append(nn.Linear(self.embed_d * self.k, self.embed_d))
        self.fc_neigh_agg_layers = nn.ModuleList(fc_neigh_agg_layers)

        # Heterogeneous Aggregation
        self.fc_het_neigh_agg = nn.Linear(self.embed_d * (1 + self.num_node_types), self.out_embed_d)

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

    def forward(self, data, x_edge_index=None):
        """
        forward propagate based on graph and its node features, edges, and het neighbourhood
        """
        x_node_feature, x_graph_het_feature, graph_node_types = data
        # TODO: Need to normalise on the input node features
        # h = self.fc_node(x_node_feature)
        # h = h.tanh()

        graph_node_het_embedding = self.node_het_embedding(x_node_feature, x_graph_het_feature, graph_node_types)
        # print(f'Node Het Embedding: {graph_node_het_embedding}')

        graph_embedding = self.graph_node_pooling(graph_node_het_embedding)
        # print(f'Graph Embedding: {graph_embedding}')
        return graph_embedding

    def node_het_embedding(self, h_embed, x_graph_het_feature, graph_node_types, x_edge_index=None):
        """
        compute graph embedding
        """
        # graph_het_node_embedding = torch.zeros(h_embed.shape).to(self.device)
        all_het_neigh_aggregated = []
        for neigh_type in range(self.num_node_types):
            node_het_features = x_graph_het_feature[neigh_type]

            # compute individual neighbour feature encode for every node in graph. size: (num_node, k, num_feature)
            node_het_neigh_embed = self.encode_node_content(node_het_features.view(
                node_het_features.shape[0] * node_het_features.shape[1],
                node_het_features.shape[2]), neigh_type
            ).view(node_het_features.shape[0], node_het_features.shape[1], node_het_features.shape[2])

            # compute aggregation on top K neigh for each node. size: (num_node, num_feature)
            node_het_neigh_aggregated = self.aggregate_neighbour(
                node_het_neigh_embed.view(
                    node_het_neigh_embed.shape[0], node_het_neigh_embed.shape[1] * node_het_neigh_embed.shape[2]),
                neigh_type)
            all_het_neigh_aggregated.append(node_het_neigh_aggregated)

        # adding self to the end of the neigh aggregation
        node_self_embed = torch.zeros(h_embed.shape[0], h_embed.shape[1]).to(self.device)
        for i, (node_feature, node_type) in enumerate(zip(h_embed, graph_node_types)):
            node_self_embed[i] = self.encode_node_content(node_feature, node_type)

        all_het_neigh_aggregated.append(node_self_embed)

        # combine all het neigh embeddings. size: (num_node, num_neigh_type * num_feature)
        concat_het_embedding = torch.cat(all_het_neigh_aggregated, 1)

        graph_node_het_embeddings = self.aggregate_het_neigh_types(concat_het_embedding)  # size: (num_node, num_feature)
        graph_node_het_embeddings = self.sigmoid(graph_node_het_embeddings)

        return graph_node_het_embeddings

    def aggregate_neighbour(self, neigh_embedding, neigh_type):
        """
        Mean of neighbours in same neigh type
        """
        return self.relu(
            self.fc_neigh_agg_layers[neigh_type](neigh_embedding))

    def encode_node_content(self, node_feature, node_type):
        """
        Aggregate the content of the node (input node features)
        """
        content_embedding = self.fc_node_content_layers[node_type](node_feature)
        return self.relu(content_embedding)

    def aggregate_het_neigh_types(self, het_neigh_embedding):
        """
        aggregate all the het types
        """
        node_het_embedding = self.fc_het_neigh_agg(
            het_neigh_embedding
            # het_neigh_embedding.view(1, (self.num_node_types + 1) * self.out_embed_d)
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

    def load_svdd_center(self):
        """
        load existing svdd center
        """
        self.set_svdd_center(torch.load(f'{self.model_path}/HetGNN_SVDD_Center.pt', map_location=self.device))

    def load_checkpoint(self, checkpoint):
        """
        load model checkpoint
        """
        checkpoint_model_path = f'{self.model_path}/HetGNN_{checkpoint}.pt'
        self.load_state_dict(torch.load(checkpoint_model_path, map_location=self.device))

    def predict_score(self, g_data):
        """
        calc dist given graph features
        """
        graph_node_feature, graph_het_feature, graph_node_types = g_data
        with torch.no_grad():
            _out = self(graph_node_feature, graph_het_feature, graph_node_types)
            score = torch.mean(torch.square(_out - self.svdd_center))
        return score

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

        if model.svdd_center is None:
            with torch.no_grad():
                print('Set initial center ..')
                hypersphere_center = torch.mean(_batch_out_resahpe, 0)
                model.set_svdd_center(hypersphere_center)
                torch.save(hypersphere_center, f'{model.model_path}/HetGNN_SVDD_Center.pt')
        else:
            hypersphere_center = model.svdd_center

        dist = torch.square(_batch_out_resahpe - hypersphere_center)
        loss_ = torch.mean(torch.sum(dist, 1))

        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

        loss = loss_ + l2_lambda * l2_norm
        return loss

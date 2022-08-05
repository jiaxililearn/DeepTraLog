# from tqdm import tqdm
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from config import relations, node_types

class HetGCN_2(nn.Module):
    def __init__(self, model_path=None, feature_size=7, out_embed_s=32, **kwargs):
        """
        test model with homegeneoug GCNConv
        """
        super(HetGCN_2, self).__init__()
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.svdd_center = None
        self.model_path = model_path

        self.embed_d = feature_size
        self.out_embed_d = out_embed_s

        self.num_node_types = len(node_types)

        # node feature content encoder
        self.conv1 = GCNConv(self.embed_d, self.out_embed_d)
        # self.conv2 = GCNConv(16, 7)

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

    def forward(self, data):
        """
        forward propagate based on node features and edge index
        """
        x_node_feature, x_edge_index = data
        h = self.conv1(x_node_feature, x_edge_index)
        # h = h.tanh()

        graph_embedding = self.graph_node_pooling(h)
        return graph_embedding

    def graph_node_pooling(self, graph_node_het_embedding):
        """
        average all the node het embedding
        """
        return torch.mean(graph_node_het_embedding, 0)

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
        with torch.no_grad():
            _out = self(g_data)
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

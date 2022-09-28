# from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from HetGCNConv_11 import HetGCNConv_11
from graph_augmentation import create_het_edge_perturbation


class HetGCN_11(nn.Module):
    def __init__(
        self,
        model_path=None,
        dataset=None,
        source_types=None,
        feature_size=7,
        out_embed_s=32,
        random_seed=32,
        num_node_types=7,
        hidden_channels=16,
        num_hidden_conv_layers=1,
        model_sub_version=0,
        num_edge_types=1,
        **kwargs,
    ):
        """
        Het GCN based on MessagePassing
            + segragation of the source neighbour type
            + relational edge type

        Adding Graph Augmentation Methods
            + Edge Perturbation (add/removing edges comply with Het characteristics)
        """
        super().__init__()
        torch.manual_seed(random_seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.svdd_center = None
        self.model_path = model_path
        self.dataset = dataset
        self.source_types = source_types
        self.model_sub_version = model_sub_version

        self.embed_d = feature_size
        self.out_embed_d = out_embed_s

        self.num_node_types = num_node_types

        # node feature content encoder
        if model_sub_version == 0:
            self.het_node_conv = HetGCNConv_11(
                self.embed_d,
                self.out_embed_d,
                self.num_node_types,
                hidden_channels=hidden_channels,
                num_hidden_conv_layers=num_hidden_conv_layers,
                num_src_types=len(source_types),
                num_edge_types=num_edge_types,
            )

        else:
            pass

        print(f"num_hidden_conv_layers: {num_hidden_conv_layers}")

        self.final_logistic = nn.Sequential(
            nn.Linear(self.out_embed_d, 1, bias=True), nn.Sigmoid()
        )

        # Others
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        # loss
        self.loss = torch.nn.BCELoss()

    def init_weights(self):
        """
        init weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, gid_batch, train=True):
        """
        forward propagate based on gid batch
        """
        batch_data = [self.dataset[i] for i in gid_batch]

        if train:
            print("Edge Perturbating for the batch ..")
            # het_edge_perturbation(args)
            synthetic_data = create_het_edge_perturbation(batch_data)

            combined_data = batch_data + synthetic_data
            combined_labels = (
                torch.tensor([0 for _ in batch_data] + [1 for _ in synthetic_data])
                .to(self.device)
                .view(-1, 1)
            )
        else:
            combined_data = batch_data
            combined_labels = None

        # print(f'x_node_feature shape: {x_node_feature.shape}')
        # print(f'x_edge_index shape: {x_edge_index.shape}')
        _out = torch.zeros(len(combined_data), 1, device=self.device)
        for i, g_data in enumerate(combined_data):
            h = self.het_node_conv(g_data, source_types=self.source_types)
            h = self.sigmoid(h)
            h = self.final_logistic(h)
            _out[i] = h
        # print(f'combined_labels: {combined_labels.shape}')
        # print(f'_out: {_out.shape}')
        # print(f'combined_labels: {combined_labels}')
        # TODO: also returns the labels
        return _out, combined_labels

    # def graph_node_pooling(self, graph_node_het_embedding):
    #     """
    #     average all the node het embedding
    #     """
    #     if graph_node_het_embedding.shape[0] == 1:
    #         return graph_node_het_embedding
    #     return torch.mean(graph_node_het_embedding, 0)

    # def set_svdd_center(self, center):
    #     """
    #     set svdd center
    #     """
    #     self.svdd_center = center

    # def load_svdd_center(self):
    #     """
    #     load existing svdd center
    #     """
    #     self.set_svdd_center(
    #         torch.load(
    #             f"{self.model_path}/HetGNN_SVDD_Center.pt", map_location=self.device
    #         )
    #     )

    def load_checkpoint(self, checkpoint):
        """
        load model checkpoint
        """
        checkpoint_model_path = f"{self.model_path}/HetGNN_{checkpoint}.pt"
        self.load_state_dict(
            torch.load(checkpoint_model_path, map_location=self.device)
        )

    def predict_score(self, g_data):
        """
        calc dist given graph features
        """
        with torch.no_grad():
            scores, _ = self(g_data, train=False)
        return scores


# def svdd_batch_loss(model, embed_batch, l2_lambda=0.001, fix_center=True):
#     """
#     Compute SVDD Loss on batch
#     """
#     # TODO
#     out_embed_d = model.out_embed_d

#     _batch_out = embed_batch
#     _batch_out_resahpe = _batch_out.view(
#         _batch_out.size()[0] * _batch_out.size()[1], out_embed_d
#     )

#     if fix_center:
#         if model.svdd_center is None:
#             with torch.no_grad():
#                 print("Set initial center ..")
#                 hypersphere_center = torch.mean(_batch_out_resahpe, 0)
#                 model.set_svdd_center(hypersphere_center)
#                 torch.save(
#                     hypersphere_center, f"{model.model_path}/HetGNN_SVDD_Center.pt"
#                 )
#         else:
#             hypersphere_center = model.svdd_center
#             #  with torch.no_grad():
#             #     hypersphere_center = (model.svdd_center + torch.mean(_batch_out_resahpe, 0)) / 2
#             #     model.set_svdd_center(hypersphere_center)
#     else:
#         with torch.no_grad():
#             print("compute batch center ..")
#             hypersphere_center = torch.mean(_batch_out_resahpe, 0)

#     dist = torch.square(_batch_out_resahpe - hypersphere_center)
#     loss_ = torch.mean(torch.sum(dist, 1))

#     l2_norm = sum(p.pow(2.0).sum() for p in model.parameters()) / 2

#     loss = loss_ + l2_lambda * l2_norm
#     return loss

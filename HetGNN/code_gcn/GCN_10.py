# from tqdm import tqdm
import random
import torch
from torch import nn
from HetGCNConv_10 import HetGCNConv_10
from subgraph import Subgraph


class HetGCN_10(nn.Module):
    def __init__(
        self,
        model_path=None,
        dataset=None,
        ppr_path=None,
        subgraph_path=None,
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

        Adding subgraph sampling to the self-supervised model
        # TODO
        """
        super().__init__()
        torch.manual_seed(random_seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.svdd_center = None
        self.model_path = model_path
        self.dataset = dataset
        self.ppr_path = ppr_path
        self.subgraph_path = subgraph_path
        self.source_types = source_types
        self.model_sub_version = model_sub_version

        self.embed_d = feature_size
        self.out_embed_d = out_embed_s
        self.sample_graph_size = 100

        self.num_node_types = num_node_types

        # node feature content encoder
        if model_sub_version == 0:
            self.het_node_conv = HetGCNConv_10(
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

        # Others
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.marginloss = nn.MarginRankingLoss(0.5)

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

    def forward(self, gid_batch, accumulate_loss=False):
        """
        forward propagate based on gid batch
        """
        if accumulate_loss:
            loss_list = []
        else:
            loss_list = None
        # print(f'x_node_feature shape: {x_node_feature.shape}')
        # print(f'x_edge_index shape: {x_edge_index.shape}')
        _out = torch.zeros(len(gid_batch), self.out_embed_d, device=self.device)
        for i, gid in enumerate(gid_batch):

            # sample subgraph
            subgraph = Subgraph(
                i,
                data=None,
                path=None,
                subgraph_path=self.subgraph_path,
            )
            subgraph.build()
            sample_idx = random.sample(
                range(self.dataset[gid][0].size(0)),
                min(self.sample_graph_size, self.dataset[gid][0].size(0)),
            )

            try:
                batch, index = subgraph.search(sample_idx)
                batch = batch.to(self.device)
            except Exception as e:
                print(f"graph size: {self.dataset[gid][0].size(0)}")
                print(f'batch graph {gid}: {batch}')
                print(f'sample_idx: {sample_idx}')
                raise Exception(e) from e

            g_data = (
                batch.x,
                batch.edge_index,
                (None, batch.edge_attr),
                self.resolve_node_types(batch.pos),
            )

            h_node, h = self.het_node_conv(g_data, source_types=self.source_types)
            h = self.sigmoid(h)
            _out[i] = h

            # TODO: calc loss
            if accumulate_loss:
                loss_ = self.margin_loss(h_node, h)
                loss_list.append(loss_)

        return _out, loss_list

    def resolve_node_types(self, node_types):
        ntypes = [[] for _ in range(self.num_node_types)]
        for nid, nt in enumerate(node_types):
            ntypes[nt].append(nid)
        return ntypes

    # def forward(self, data):
    #     """
    #     forward propagate based on node features and edge index
    #     """
    #     x_node_feature, x_edge_index, x_edge_weight, x_node_types = data

    #     # print(f'x_node_feature shape: {x_node_feature.shape}')
    #     # print(f'x_edge_index shape: {x_edge_index.shape}')
    #     h = self.conv1(x_node_feature, x_edge_index, x_node_types, edge_weight=x_edge_weight)
    #     # h = self.relu(h)
    #     # h = self.conv2(h, x_edge_index, x_node_types, edge_weight=x_edge_weight)
    #     h = h.sigmoid()

    #     graph_embedding = self.graph_node_pooling(h)
    #     return graph_embedding

    def graph_node_pooling(self, graph_node_het_embedding):
        """
        average all the node het embedding
        """
        if graph_node_het_embedding.shape[0] == 1:
            return graph_node_het_embedding
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
        self.set_svdd_center(
            torch.load(
                f"{self.model_path}/HetGNN_SVDD_Center.pt", map_location=self.device
            )
        )

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
            _, score = self(g_data, accumulate_loss=True)
            # score = torch.mean(torch.square(_out - self.svdd_center), 1)  # mean on rows
        return score

    def margin_loss(self, hidden1, summary1):
        """
        calc margin loss
        """
        shuf_index = torch.randperm(summary1.size(1))
        hidden2 = hidden1[:, shuf_index]
        summary2 = summary1[:, shuf_index]

        logits_aa = torch.sigmoid(torch.mean(hidden1 * summary1, dim=-1))
        logits_bb = torch.sigmoid(torch.mean(hidden2 * summary2, dim=-1))
        logits_ab = torch.sigmoid(torch.mean(hidden1 * summary2, dim=-1))
        logits_ba = torch.sigmoid(torch.mean(hidden2 * summary1, dim=-1))

        # print(f'logits_aa shape: {logits_aa.shape}')
        # print(f'hidden2 shape: {hidden2.shape}')
        # print(f'summary2 shape: {summary2.shape}')
        
        print(f'hidden1: {hidden1}')
        print(f'hidden2: {hidden2}')
        print(f'summary1: {summary1}')
        print(f'summary2: {summary2}')
        print(f'logits_aa: {logits_aa}')
        print(f'logits_bb: {logits_bb}')
        print(f'logits_ab: {logits_ab}')
        print(f'logits_ba: {logits_ba}')

        total_loss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        total_loss += self.marginloss(logits_aa, logits_ba, ones)
        total_loss += self.marginloss(logits_bb, logits_ab, ones)

        return total_loss


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

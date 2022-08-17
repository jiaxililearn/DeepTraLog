import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from args import read_args
# import numpy as np
# import string
# import re
# import math
# args = read_args()


class HetAgg(nn.Module):
    def __init__(self, model_path, dataset, svdd_center=None, **kwargs):
        # a_neigh_list_train, b_neigh_list_train,
        #  a_train_id_list, b_train_id_list):

        super(HetAgg, self).__init__()
        embed_d = 26

        self.model_path = model_path

        self.k = 12
        self.num_node_types = 14

        self.dataset = dataset

        self.svdd_center = svdd_center

        self.fc_a_a_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_a_b_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_a_c_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_a_d_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_a_e_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_a_f_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_a_g_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_a_h_agg = nn.Linear(embed_d * self.k, embed_d)

        self.fc_b_a_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_b_b_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_b_c_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_b_d_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_b_e_agg = nn.Linear(embed_d * self.k, embed_d)
        self.fc_b_h_agg = nn.Linear(embed_d * self.k, embed_d)

        self.fc_het_neigh_agg = nn.Linear(embed_d * self.num_node_types, embed_d)

        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.LeakyReLU()
        # self.drop = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(embed_d)
        self.bn2 = nn.BatchNorm1d(embed_d)
        self.embed_d = embed_d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge_content_agg(self, gid_batch, edge_type):

        embed_d = self.embed_d
        if edge_type == 'a_a':
            edge_embed = self.dataset[gid_batch][0]
            fc_agg = self.fc_a_a_agg
        elif edge_type == 'a_b':
            edge_embed = self.dataset[gid_batch][1]
            fc_agg = self.fc_a_b_agg
        elif edge_type == 'a_c':
            edge_embed = self.dataset[gid_batch][2]
            fc_agg = self.fc_a_c_agg
        elif edge_type == 'a_d':
            edge_embed = self.dataset[gid_batch][3]
            fc_agg = self.fc_a_d_agg
        elif edge_type == 'a_e':
            edge_embed = self.dataset[gid_batch][4]
            fc_agg = self.fc_a_e_agg
        elif edge_type == 'a_f':
            edge_embed = self.dataset[gid_batch][5]
            fc_agg = self.fc_a_f_agg
        elif edge_type == 'a_g':
            edge_embed = self.dataset[gid_batch][6]
            fc_agg = self.fc_a_g_agg
        elif edge_type == 'a_h':
            edge_embed = self.dataset[gid_batch][7]
            fc_agg = self.fc_a_h_agg
        elif edge_type == 'b_a':
            edge_embed = self.dataset[gid_batch][8]
            fc_agg = self.fc_b_a_agg
        elif edge_type == 'b_b':
            edge_embed = self.dataset[gid_batch][9]
            fc_agg = self.fc_b_b_agg
        elif edge_type == 'b_c':
            edge_embed = self.dataset[gid_batch][10]
            fc_agg = self.fc_b_c_agg
        elif edge_type == 'b_d':
            edge_embed = self.dataset[gid_batch][11]
            fc_agg = self.fc_b_d_agg
        elif edge_type == 'b_e':
            edge_embed = self.dataset[gid_batch][12]
            fc_agg = self.fc_b_e_agg
        # elif edge_type == 'b_h':
        else:
            edge_embed = self.dataset[gid_batch][13]
            fc_agg = self.fc_b_h_agg

        concate_embed = edge_embed.view(len(gid_batch), 1, embed_d * self.k)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        output = fc_agg(concate_embed)
        # print(output)
        # output_bn = self.bn1(output)
        # return torch.mean(output, 0)
        return self.act(output).view(len(gid_batch), embed_d)

    def node_neigh_agg(self, gid_batch, edge_type):  # type based neighbor aggregation with rnn
        # embed_d = self.embed_d
        neigh_agg = self.edge_content_agg(gid_batch, edge_type)
        return neigh_agg

    # heterogeneous neighbor aggregation
    def node_het_agg(self, gid_batch):

        a_a_agg_batch = self.node_neigh_agg(gid_batch, 'a_a')
        a_b_agg_batch = self.node_neigh_agg(gid_batch, 'a_b')
        a_c_agg_batch = self.node_neigh_agg(gid_batch, 'a_c')
        a_d_agg_batch = self.node_neigh_agg(gid_batch, 'a_d')
        a_e_agg_batch = self.node_neigh_agg(gid_batch, 'a_e')
        a_f_agg_batch = self.node_neigh_agg(gid_batch, 'a_f')
        a_g_agg_batch = self.node_neigh_agg(gid_batch, 'a_g')
        a_h_agg_batch = self.node_neigh_agg(gid_batch, 'a_h')

        b_a_agg_batch = self.node_neigh_agg(gid_batch, 'b_a')
        b_b_agg_batch = self.node_neigh_agg(gid_batch, 'b_b')
        b_c_agg_batch = self.node_neigh_agg(gid_batch, 'b_c')
        b_d_agg_batch = self.node_neigh_agg(gid_batch, 'b_d')
        b_e_agg_batch = self.node_neigh_agg(gid_batch, 'b_e')
        b_h_agg_batch = self.node_neigh_agg(gid_batch, 'b_h')

        agg_batch = torch.cat((a_a_agg_batch, a_b_agg_batch, a_c_agg_batch,
                               a_d_agg_batch, a_e_agg_batch, a_f_agg_batch,
                               a_g_agg_batch, a_h_agg_batch, b_a_agg_batch,
                               b_b_agg_batch, b_c_agg_batch, b_d_agg_batch,
                               b_e_agg_batch, b_h_agg_batch),
                              1).view(len(a_a_agg_batch), self.embed_d * self.num_node_types)

        het_agg_batch = self.sigmoid(
            self.fc_het_neigh_agg(agg_batch)
        )

        # skip attention module
        # atten_w = self.act(
        #     torch.bmm(concat_embed,)
        # )
        return het_agg_batch

    def het_agg(self, gid_batch):
        # aggregate heterogeneous neighbours
        _agg = self.node_het_agg(gid_batch)
        return _agg

    def aggregate_all(self, gid_batch):
        _agg = self.het_agg(gid_batch)
        return _agg

    def forward(self, gid_batch):
        _out = self.aggregate_all(gid_batch)
        return _out

    def set_svdd_center(self, center):
        self.svdd_center = center

    def predict_score(self, batch):
        with torch.no_grad():
            _out = self(batch)
            score = torch.mean(torch.square(_out - self.svdd_center), 1)
        return score

# SVDD Loss
def svdd_batch_loss(model, embed_batch, l2_lambda=0.001, **kwargs):  # nu: {0.1, 0.01}
    l2_lambda = l2_lambda

    _batch_out = embed_batch
    _batch_out_resahpe = _batch_out.view(_batch_out.size()[0] * _batch_out.size()[1], model.embed_d)

    if model.svdd_center is None:
        with torch.no_grad():
            print('Set initial center ..')
            hypersphere_center = torch.mean(_batch_out_resahpe, 0)
            model.set_svdd_center(hypersphere_center)
            torch.save(hypersphere_center, model.model_path + 'HetGNN_SVDD_Center.pt')
        # with open(model.args.model_path + "HetGNN_SVDD_Center.pt", 'w') as fout:
        #     fout.write()
    else:
        hypersphere_center = model.svdd_center

    dist = torch.square(_batch_out_resahpe - hypersphere_center)
    loss_ = torch.mean(torch.sum(dist, 1))

    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

    loss = loss_ + l2_lambda * l2_norm
    return loss


# Original loss imp
# def cross_entropy_loss(c_embed_batch, pos_embed_batch, neg_embed_batch, embed_d):
#     batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]
#
#     c_embed = c_embed_batch.view(batch_size, 1, embed_d)
#     pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
#     neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)
#
#     out_p = torch.bmm(c_embed, pos_embed)
#     out_n = - torch.bmm(c_embed, neg_embed)
#
#     sum_p = F.logsigmoid(out_p)
#     sum_n = F.logsigmoid(out_n)
#     loss_sum = - (sum_p + sum_n)
#
#     # loss_sum = loss_sum.sum() / batch_size
#
#     return loss_sum.mean()

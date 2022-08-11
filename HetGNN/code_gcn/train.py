import os
import time
from collections import defaultdict
import torch
import torch.optim as optim
# from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import pickle

from data_loader_origin import EventGraphDataset
from data_loader import HetGCNEventGraphDataset

import boto3

class Train(object):
    def __init__(self, data_path, model_path, train_iter_n, num_train, batch_s, mini_batch_s, lr,
                 save_model_freq, s3_stage, s3_bucket, s3_prefix, model_version, dataset_id,
                 test_set=True, fix_center=True, num_eval=None, unzip=False,
                 split_data=True, **kwargs):
        super(Train, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Device: {self.device}')

        self.data_root_dir = data_path
        self.model_path = model_path
        self.model_version = model_version
        self.dataset_id = dataset_id
        self.test_set = test_set
        self.split_data = split_data

        self.fix_center = fix_center

        if self.model_version == 1:
            from GCN_1 import HetGCN_1 as HetGCN

            self.dataset = EventGraphDataset(
                f'{self.data_root_dir}/node_feature_norm.csv',
                f'{self.data_root_dir}/graph_het_neigh_list',
                f'{self.data_root_dir}/node_types.csv',
                unzip=unzip
            )
        elif self.model_version == 2:
            from GCN_2 import HetGCN_2 as HetGCN
            self.dataset = EventGraphDataset(
                node_feature_csv=f'{self.data_root_dir}/node_feature_norm.csv',
                edge_index_csv=f'{self.data_root_dir}/edge_index.csv',
                het_types=False,
                unzip=unzip
            )
        elif self.model_version == 3:
            from GCN_3 import HetGCN_3 as HetGCN

            if self.dataset_id == 0:
                self.dataset = HetGCNEventGraphDataset(
                    node_feature_csv=f'{self.data_root_dir}/node_feature_norm.csv',
                    edge_index_csv=f'{self.data_root_dir}/edge_index.csv',
                    node_type_txt=f'{self.data_root_dir}/node_types.txt',
                    include_edge_weight=False
                )
            # elif self.dataset_id == 1:
            #     self.dataset = CMUDataset(
            #         node_feature_path=f'{self.data_root_dir}',
            #         edge_index_csv=f'{self.data_root_dir}/edge_index.csv',
            #         node_type_txt=f'{self.data_root_dir}/node_types.txt'
            #     )
        else:
            from GCN import HetGCN
            self.dataset = EventGraphDataset(
                f'{self.data_root_dir}/node_feature_norm.csv',
                f'{self.data_root_dir}/graph_het_neigh_list',
                f'{self.data_root_dir}/node_types.csv',
                unzip=unzip
            )
        
        self.num_train_benign = num_train
        self.num_eval = num_eval
        
        self.embed_d = kwargs['feature_size']
        self.out_embed_d = kwargs['out_embed_s']

        self.train_iter_n = train_iter_n
        self.lr = lr

        self.batch_s = batch_s
        self.mini_batch_s = mini_batch_s

        self.save_model_freq = save_model_freq
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_stage = s3_stage

        self.model = HetGCN(model_path=self.model_path, **kwargs).to(self.device)

        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=0)
        self.model.init_weights()

    def train(self):
        """
        model training method
        """
        print('model training ...')
        self.model.train()

        epoch_loss_list = []
        eval_list = []

        # random split train eval test data or read from existing files
        if self.split_data:
            benign_gid_list, eval_gid_list, test_gid_list = self.train_eval_test_split(test_set=self.test_set)
        else:
            benign_gid_list, eval_gid_list, test_gid_list = self.read_train_eval_test_sets()

        for iter_i in range(self.train_iter_n):
            self.model.train()
            print('iteration ' + str(iter_i) + ' ...')
            batch_list = benign_gid_list.reshape(int(benign_gid_list.shape[0] / self.batch_s), self.batch_s)
            avg_loss_list = []

            epoch_start_time = time.time()
            with torch.autograd.set_detect_anomaly(True):
                for batch_n, k in tqdm(enumerate(batch_list)):
                    batch_start_time = time.time()

                    _out = torch.zeros(
                        int(self.batch_s / self.mini_batch_s), self.mini_batch_s, self.out_embed_d).to(self.device)

                    mini_batch_list = k.reshape(int(len(k) / self.mini_batch_s), self.mini_batch_s)
                    for mini_n, mini_k in enumerate(mini_batch_list):
                        for i, gid in enumerate(mini_k):
                            print(f'forward graph: {i} -- {gid}')
                            _out[mini_n][i] = self.model(self.dataset[gid])

                    batch_loss = self.model.svdd_batch_loss(self.model, _out, fix_center=self.fix_center)
                    avg_loss_list.append(batch_loss.tolist())
                    # print(f'\t Batch Size: {len(k)}; Mini Batch Size: {mini_batch_list.shape}')
                    # print(f'Model Output: {_out}')
                    self.optim.zero_grad()
                    batch_loss.backward(retain_graph=True)
                    self.optim.step()
                    print(f'\t Batch Loss: {batch_loss}; Batch Time: {time.time()-batch_start_time}s;')

            epoch_loss_list.append(np.mean(avg_loss_list))
            print(f'Epoch Loss: {np.mean(avg_loss_list)}; Epoch Time: {time.time() - epoch_start_time}s;')

            if iter_i % self.save_model_freq == 0:
                # Evaluate the model
                print("Evaluating Model ..")
                roc_auc, ap = self.eval_model(eval_gid_list)
                eval_list.append([roc_auc, ap])

                # Save Model
                torch.save(self.model.state_dict(), f'{self.model_path}/HetGNN_{iter_i}.pt')
                # save current all epoch losses
                with open(f'{self.model_path}/train_loss.txt', 'w') as fout:
                    for lo in epoch_loss_list:
                        fout.write(f'{lo}\n')

                with open(f'{self.model_path}/eval_metrics.txt', 'w') as fout:
                    for roc_auc, ap in eval_list:
                        fout.write(f'{roc_auc} {ap}\n')

                # sync to s3 for intermediate save
                if self.s3_stage:
                    self.sync_model_path_to_s3(s3_bucket=self.s3_bucket, s3_prefix=self.s3_prefix)
            print('iteration ' + str(iter_i) + ' finish.')

    def train_eval_test_split(self, test_set=True):
        """
        splite data into train eval test
        """
        print('Random Split Train/Eval/Test.')
        trace_info_df = pd.read_csv(f'{self.data_root_dir}/trace_info.csv', index_col=None)

        benign_gid_list = trace_info_df[trace_info_df['trace_bool'] == True]['trace_id'].values
        attack_gid_list = trace_info_df[trace_info_df['trace_bool'] == False]['trace_id'].values

        num_train_benign = self.num_train_benign

        # Train/Eval/Test = 0.6/0.2/0.2
        rep_train_benign_gid_list = np.random.choice(benign_gid_list, num_train_benign, replace=False)
        left_benign_gid_list = benign_gid_list[np.in1d(
            benign_gid_list, rep_train_benign_gid_list, invert=True)]
        
        if test_set:
            num_eval_benign = int((benign_gid_list.shape[0] - num_train_benign) / 2)
            num_eval_attack = int(attack_gid_list.shape[0] / 2)
        else:
            num_eval_benign = int((benign_gid_list.shape[0] - num_train_benign))
            num_eval_attack = int(attack_gid_list.shape[0])

        eval_benign_gid_list = np.random.choice(left_benign_gid_list,
                                                num_eval_benign, replace=False)
        test_benign_gid_list = left_benign_gid_list[np.in1d(
            left_benign_gid_list, eval_benign_gid_list, invert=True)]

        eval_attack_gid_list = np.random.choice(attack_gid_list, num_eval_attack, replace=False)
        test_attack_gid_list = attack_gid_list[np.in1d(
            attack_gid_list, eval_attack_gid_list, invert=True)]

        eval_gid_list = np.concatenate([eval_benign_gid_list, eval_attack_gid_list], axis=0)
        test_gid_list = np.concatenate([test_benign_gid_list, test_attack_gid_list], axis=0)

        np.random.shuffle(eval_gid_list)
        np.random.shuffle(test_gid_list)
        np.random.shuffle(rep_train_benign_gid_list)

        print(f'Model Training Data Size: {rep_train_benign_gid_list.shape}')
        print(f'Model Eval Data Size: {eval_gid_list.shape}')
        print(f'Model Test Data Size: {test_gid_list.shape}')

        # write out current train/eval/test gids
        with open(f'{self.model_path}/model_gid_list_train.txt', 'w') as fout:
            for i in rep_train_benign_gid_list:
                fout.write(f'{i} ')
            fout.write('\n')
        with open(f'{self.model_path}/model_gid_list_eval.txt', 'w') as fout:
            for i in eval_gid_list:
                fout.write(f'{i} ')
            fout.write('\n')
        with open(f'{self.model_path}/model_gid_list_test.txt', 'w') as fout:
            for i in test_gid_list:
                fout.write(f'{i} ')
            fout.write('\n')

        return rep_train_benign_gid_list, eval_gid_list, test_gid_list

    def read_train_eval_test_sets(self):
        """
        Read existing train eval test datasets graph ids
        """
        print('Read Existing Split Train/Eval/Test.')
        with open(f'{self.model_path}/model_gid_list_train.txt', 'r') as fin:
            train_list = np.array([int(i) for i in fin.read().strip().split()]).astype(int)
        with open(f'{self.model_path}/model_gid_list_eval.txt', 'r') as fin:
            eval_list = np.array([int(i) for i in fin.read().strip().split()]).astype(int)
        with open(f'{self.model_path}/model_gid_list_test.txt', 'r') as fin:
            test_list = np.array([int(i) for i in fin.read().strip().split()]).astype(int)
        return train_list, eval_list, test_list

    def sync_model_path_to_s3(self, s3_bucket, s3_prefix):
        """
        sync model path to S3 periodically
        """
        client = boto3.client('s3')

        for root, dirs, files in os.walk(self.model_path):
            for filename in files:
                local_path = os.path.join(root, filename)

                relative_path = os.path.relpath(local_path, self.model_path)

                s3_path = os.path.join(s3_prefix, relative_path)

                try:
                    print(f"Uploading {s3_path}...")
                    client.upload_file(local_path, s3_bucket, s3_path)

                except Exception as e:
                    print(f"Failed to upload {local_path} to {s3_path}.\n{e}")

    def eval_model(self, eval_list):
        """
        Eval Model
        """
        if self.num_eval:
            eval_list = eval_list[:self.num_eval]

        self.model.eval()
        trace_info_df = pd.read_csv(f'{self.data_root_dir}/trace_info.csv', index_col=None)
        with torch.no_grad():
            pred_scores = []
            for gid in eval_list:
                _score = self.model.predict_score(self.dataset[gid]).cpu().detach().numpy()
                pred_scores.append(_score)

            label = trace_info_df[trace_info_df['trace_id'].isin(eval_list)]['trace_bool'] \
                .apply(lambda x: 0 if x else 1).values

            fpr, tpr, roc_thresholds = roc_curve(label, pred_scores)
            roc_auc = auc(fpr, tpr)

            precision, recall, pr_thresholds = precision_recall_curve(label, pred_scores)
            ap = auc(recall, precision)

            print(f'\tAUC:{roc_auc}; Avg Precision:{ap};')

        return roc_auc, ap

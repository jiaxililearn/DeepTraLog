import os
import time
from collections import defaultdict
import torch
import torch.optim as optim
import data_generator
import tools
from args import read_args
# from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import pandas as pd
import random
import pickle
from config import relations
import boto3


# torch.set_num_threads(2)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(f'cuda available: {torch.cuda.is_available()}')


class model_class(object):
    def __init__(self, args):
        super(model_class, self).__init__()
        self.args = args
        self.gpu = torch.cuda.is_available()

        if self.gpu:
            print(f'current cuda: {torch.cuda.current_device()}')
            print(f'current cuda: {torch.cuda.current_device()}')
            print(f'cuda device count: {torch.cuda.device_count()}')

        feature_list = []
        feature_index = []
        
        if args.preprocess:
            input_data = data_generator.input_data(args=self.args)

            # # save a tmp pickle file for processed feature list
            # print('Save Processed Feature List')
            # with open(f'{args.data_path}/feature_list.pkl', 'wb') as fout:
            #     pickle.dump(input_data.feature_list, fout, protocol=4)
            # input_data.gen_het_rand_walk()

            self.input_data = input_data

            if self.args.train_test_label == 2:  # generate neighbor set of each node
                # input_data.het_walk_restart()
                print("Wrong Arguments. Exit.")
                exit(0)

            feature_list = input_data.feature_list
            # preprocess only then exit
            exit(0)
        else:
            # read feature lists from previous saved files
            # if sagemaker train env is set
            if args.train:
                self.feature_list_root_dir = f"{args.train}"
            else:
                self.feature_list_root_dir = f"{args.data_path}/feature_list"
            #
            for r in relations:
                f_path = f'{self.feature_list_root_dir}/feature_list_{r}.pt'
                idx_path = f'{self.feature_list_root_dir}/feature_index_{r}.pt'
                
                print(f'Read relation feature list {f_path} ..')
                
                feature_ = torch.load(f_path)
                index_ = torch.load(idx_path)

                graph_index = defaultdict(list)
                for i, gid in enumerate(index_):
                    graph_index[gid].append(i)
        
                
#                 print(feature_)
                print(feature_.size())
#                 print(feature_[[0,2,3]])
                
                feature_list.append(feature_)
                feature_index.append(graph_index)

#         for i, fl in enumerate(feature_list):
#             feature_list[i] = torch.from_numpy(np.array(feature_list[i])).float()

        # avoid converting all data into GPU memory, will picking the necessary data to be cuda
        # if self.gpu:
        #     for i, _ in enumerate(feature_list):
        #         print(i)
        #         feature_list[i] = feature_list[i].cuda()

#         graph_feature_idx_list = [] # deprecated list: this would be all the graph ids

        self.model = tools.HetAgg(args, feature_list,
                                  feature_index)

        if self.gpu:
            self.model = self.model.cuda()
        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay=0)
        self.model.init_weights()

    def model_train(self):
        print('model training ...')
        if self.args.checkpoint != '':
            self.model.load_state_dict(torch.load(self.args.checkpoint))

        self.model.train()
        batch_s = self.args.batch_s
        mini_batch_s = self.args.mini_batch_s
        embed_d = self.args.embed_d
        # output embed size
        out_embed_d = self.args.out_embed_d

        epoch_loss_list = []
        eval_list = []
        benign_gid_list, eval_gid_list, test_gid_list = self.train_eval_test_split()

        for iter_i in range(self.args.train_iter_n):
            self.model.train()
            print('iteration ' + str(iter_i) + ' ...')
            batch_list = benign_gid_list.reshape(int(benign_gid_list.shape[0] / batch_s), batch_s)
            avg_loss_list = []

            epoch_start_time = time.time()
            for batch_n, k in enumerate(batch_list):
                batch_start_time = time.time()

                _out = torch.zeros(int(batch_s / mini_batch_s), mini_batch_s, out_embed_d)
                if self.gpu:
                    _out = _out.cuda()

                mini_batch_list = k.reshape(int(len(k) / mini_batch_s), mini_batch_s)
                for mini_n, mini_k in enumerate(mini_batch_list):
                    _out_temp = self.model(mini_k)
                    _out[mini_n] = _out_temp

                # TODO: perhaps batch norm before fc layer
                batch_loss = tools.svdd_batch_loss(self.model, _out)
                avg_loss_list.append(batch_loss.tolist())
                # print(f'\t Batch Size: {len(k)}; Mini Batch Size: {mini_batch_list.shape}')
                self.optim.zero_grad()
                batch_loss.backward(retain_graph=True)
                self.optim.step()
                print(f'\t Batch Loss: {batch_loss}; Batch Time: {1000 * (time.time()-batch_start_time)}ms')
            epoch_loss_list.append(np.mean(avg_loss_list))
            print(f'Epoch Loss: {np.mean(avg_loss_list)}; Epoch Time: {1000 * (time.time() - epoch_start_time)}ms')

            if iter_i % self.args.save_model_freq == 0:
                # Evaluate the model
                print("Evaluating Model ..")
                roc_auc, ap = self.eval_model(eval_gid_list)
                eval_list.append([roc_auc, ap])
                
                # Save Model
                torch.save(self.model.state_dict(), self.args.model_path +
                           "HetGNN_" + str(iter_i) + ".pt")
                # save current all epoch losses
                with open(f'{self.args.model_path}train_loss.txt', 'w') as fout:
                    for lo in epoch_loss_list:
                        fout.write(f'{lo}\n')

                with open(f'{self.args.model_path}eval_metrics.txt', 'w') as fout:
                    for roc_auc, ap in eval_list:
                        fout.write(f'{roc_auc} {ap}\n')

                # sync to s3 for intermediate save
                self.sync_model_path_to_s3(s3_bucket='prod-tpgt-knowledge-lake-sandpit-v1', s3_prefix='application/anomaly_detection/deeptralog/HetGNN/model_save_top10/')

            print('iteration ' + str(iter_i) + ' finish.')

    def sync_model_path_to_s3(self, s3_bucket, s3_prefix):
        """
        sync model path to S3 periodically
        """
        client = boto3.client('s3')

        for root, dirs, files in os.walk(self.args.model_path):
            for filename in files:
                local_path = os.path.join(root, filename)

                relative_path = os.path.relpath(local_path, self.args.model_path)

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
        self.model.eval()
        trace_info_df = pd.read_csv(f'{self.feature_list_root_dir}/trace_info.csv', index_col=None)
        with torch.no_grad():
            pred_scores = self.model.predict_score(eval_list)
            label = trace_info_df[trace_info_df['trace_id'].isin(eval_list)]['trace_bool'].apply(lambda x: 0 if x else 1).values

            if self.gpu:
                pred_scores = pred_scores.cpu()

            fpr, tpr, roc_thresholds = roc_curve(label, pred_scores.numpy())
            roc_auc = auc(fpr, tpr)

            precision, recall, pr_thresholds = precision_recall_curve(label, pred_scores.numpy())
            ap = auc(recall, precision)

            print(f'\tAUC:{roc_auc}; Avg Precision:{ap};')

        return roc_auc, ap
    # def eval_model(self, eval_list):
    #     """
    #     for streamspot
    #     """
    #     self.model.eval()

    #     with torch.no_grad():
    #         pred_scores = self.model.predict_score(eval_list)
    #         label = np.where((eval_list >= 300) & (eval_list < 400), 1, 0)

    #         fpr, tpr, roc_thresholds = roc_curve(label, pred_scores.numpy())
    #         roc_auc = auc(fpr, tpr)

    #         precision, recall, pr_thresholds = precision_recall_curve(label, pred_scores.numpy())
    #         ap = auc(recall, precision)

    #         print(f'\tAUC:{roc_auc}, Avg Precision:{ap}')

    #     return roc_auc, ap
    

    def train_eval_test_split(self):
        """
        splite data into train eval test
        """
        trace_info_df = pd.read_csv(f'{self.feature_list_root_dir}/trace_info.csv', index_col=None)

        all_gid_list = trace_info_df['trace_id'].values
        benign_gid_list = trace_info_df[trace_info_df['trace_bool']==True]['trace_id'].values
        attack_gid_list = trace_info_df[trace_info_df['trace_bool']==False]['trace_id'].values

        num_train_benign = 65400

        # Train/Eval/Test = 0.6/0.2/0.2
        rep_train_benign_gid_list = np.random.choice(benign_gid_list, num_train_benign, replace=False)
        left_benign_gid_list = benign_gid_list[np.in1d(
            benign_gid_list, rep_train_benign_gid_list, invert=True)]

        eval_benign_gid_list = np.random.choice(left_benign_gid_list, int((benign_gid_list.shape[0] - num_train_benign) / 2), replace=False)
        test_benign_gid_list = left_benign_gid_list[np.in1d(
            left_benign_gid_list, eval_benign_gid_list, invert=True)]

        eval_attack_gid_list = np.random.choice(attack_gid_list, int(attack_gid_list.shape[0]/2), replace=False)
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
        with open(f'{self.args.model_path}/model_gid_list_train.txt', 'w') as fout:
            for i in rep_train_benign_gid_list:
                fout.write(f'{i} ')
            fout.write('\n')
        with open(f'{self.args.model_path}/model_gid_list_eval.txt', 'w') as fout:
            for i in eval_gid_list:
                fout.write(f'{i} ')
            fout.write('\n')
        with open(f'{self.args.model_path}/model_gid_list_test.txt', 'w') as fout:
            for i in test_gid_list:
                fout.write(f'{i} ')
            fout.write('\n')

        return rep_train_benign_gid_list, eval_gid_list, test_gid_list

    # def train_eval_test_split(self):
    #     """
    #     for streamspot
    #     """
    #     all_gid_list = np.array(range(600))
    #     benign_gid_list = all_gid_list[(all_gid_list < 300) | (all_gid_list > 399)]
    #     attack_gid_list = np.array(range(300, 400))

    #     # Train/Eval/Test = 0.6/0.2/0.2
    #     rep_train_benign_gid_list = np.random.choice(benign_gid_list, 360, replace=False)
    #     left_benign_gid_list = benign_gid_list[np.in1d(
    #         benign_gid_list, rep_train_benign_gid_list, invert=True)]

    #     eval_benign_gid_list = np.random.choice(left_benign_gid_list, 70, replace=False)
    #     test_benign_gid_list = left_benign_gid_list[np.in1d(
    #         left_benign_gid_list, eval_benign_gid_list, invert=True)]

    #     eval_attack_gid_list = np.random.choice(attack_gid_list, 50, replace=False)
    #     test_attack_gid_list = attack_gid_list[np.in1d(
    #         attack_gid_list, eval_attack_gid_list, invert=True)]

    #     eval_gid_list = np.concatenate([eval_benign_gid_list, eval_attack_gid_list], axis=0)
    #     test_gid_list = np.concatenate([test_benign_gid_list, test_attack_gid_list], axis=0)

    #     np.random.shuffle(eval_gid_list)
    #     np.random.shuffle(test_gid_list)
    #     np.random.shuffle(rep_train_benign_gid_list)

    #     print(f'Model Training Data Size: {rep_train_benign_gid_list.shape}')
    #     print(f'Model Eval Data Size: {eval_gid_list.shape}')
    #     print(f'Model Test Data Size: {test_gid_list.shape}')

    #     # write out current train/eval/test gids
    #     with open('../data/data_splits/rep_model_train_gid_list.txt', 'w') as fout:
    #         for i in rep_train_benign_gid_list:
    #             fout.write(f'{i} ')
    #         fout.write('\n')
    #     with open('../data/data_splits/clf_eval_gid_list.txt', 'w') as fout:
    #         for i in eval_gid_list:
    #             fout.write(f'{i} ')
    #         fout.write('\n')
    #     with open('../data/data_splits/clf_test_gid_list.txt', 'w') as fout:
    #         for i in test_gid_list:
    #             fout.write(f'{i} ')
    #         fout.write('\n')

    #     return rep_train_benign_gid_list, eval_gid_list, test_gid_list


if __name__ == '__main__':
    args = read_args()
    print("------arguments-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))

    # fix random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # model
    model_object = model_class(args)

    if args.train_test_label == 0:
        model_object.model_train()

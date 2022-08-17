import os
from pydoc import resolve
import click
from train import Train

@click.command()
@click.option('--lr', default=0.0001, help='learning rate')
@click.option('--save_model_freq', default=2, help='data_path')
@click.option('--model_path', default='../model_save_clean', help='model path dir')
@click.option('--data_path', default='../ProcessedData_clean', help='data path dir')
@click.option('--num_train', default=64000, help='number of training graphs')
@click.option('--num_eval', default=None, type=int, help='limit number of eval graphs, None for do not limit')
@click.option('--batch_s', default=2000, help='batch size')
@click.option('--mini_batch_s', default=500, help='mini batch size')
@click.option('--train_iter_n', default=250, help='max train iter')
@click.option('--num_node_types', default=8, help='num of node types in data')
@click.option('--source_types', default=None, type=str, help='consider Source types')
@click.option('--input_type', default='single', type=str, help='the way of feeding model. i.e, single | batch')
@click.option('--hidden_channels', default=16, help='size of hidden channels')
@click.option('--feature_size', default=7, help='input node feature size')
@click.option('--out_embed_s', default=32, help='output feature size')
@click.option('--seed', default=32, help='random seed')
@click.option('--model_version', default=3, help='train with model version')
@click.option('--model_sub_version', default=0, help='train with sub model version')
@click.option('--dataset_id', default=0, help='choose dataset used for training')
@click.option('--fix_center', default=True, type=bool, help='if fix the svdd center on first batch pass')
@click.option('--test_set', default=True, type=bool, help='if create test dataset from input')
@click.option('--ignore_weight', default=False, type=bool, help='if ignore the edge weight')
@click.option('--split_data', default=True, type=bool, help='if random split data on train or read from existings')
@click.option('--sagemaker', default=True, type=bool, help='is it running in SageMaker')
@click.option('--unzip', default=True, type=bool, help='if unzip feature lists first')
@click.option('--s3_stage', default=True, type=bool, help='if stage results to s3')
@click.option('--s3_bucket', default='prod-tpgt-knowledge-lake-sandpit-v1',
              help='S3 bucket to upload intermediate artifacts')
@click.option('--s3_prefix', default='application/anomaly_detection/deeptralog/HetGNN/model_save_clean/',
              help='S3 prefix to upload intermediate artifacts')
def main(**args):
    args = resolve_args(args)
    print(args)
    t = Train(**args)
    t.train()

def resolve_args(args):
    if args['sagemaker']:
        args['model_path'] = os.environ['SM_MODEL_DIR']
        args['data_path'] = os.environ['SM_CHANNEL_TRAIN']
    return args

if __name__ == '__main__':
    main()

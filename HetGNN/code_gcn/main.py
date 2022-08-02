import os
from pydoc import resolve
import click
from train import Train

@click.command()
@click.option('--lr', default=0.0001, help='learning rate')
@click.option('--save_model_freq', default=5, help='data_path')
@click.option('--model_path', default='../model_save_clean', help='model path dir')
@click.option('--data_path', default='../ProcessedData_clean', help='data path dir')
@click.option('--batch_s', default=650, help='batch size')
@click.option('--mini_batch_s', default=65, help='mini batch size')
@click.option('--train_iter_n', default=200, help='max train iter')
@click.option('--num_node_type', default=8, help='num of node types in data')
@click.option('--feature_size', default=7, help='input node feature size')
@click.option('--sagemaker', default=True, help='is it running in SageMaker')
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

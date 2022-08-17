from tqdm import tqdm
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict
import numpy as np

from sklearn.manifold import TSNE

def loss_and_eval_results(model_result_root_dir, model_eval_freq=2):
    train_loss = pd.read_csv(f'{model_result_root_dir}/train_loss.txt', names=['loss'])
    train_loss = train_loss.reset_index().rename(columns={'index': 'epoch'})

    eval_result = pd.read_csv(f'{model_result_root_dir}/eval_metrics.txt', sep=' ', names=['AUC', 'AP'])
    eval_result = eval_result.reset_index().rename(columns={'index': 'epoch'})
    eval_result['epoch'] = eval_result['epoch'] * model_eval_freq

    eval_result['avg_metric'] = (eval_result['AUC'] + eval_result['AP'])/2


    # train_loss, eval_result

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for metric in eval_result.columns[1:]:
        fig.add_trace(
            go.Scatter(x=eval_result['epoch'], y=eval_result[metric],
                       mode='lines',
                       name=metric),
            secondary_y=False
        )

    fig.add_trace(
        go.Scatter(x=train_loss['epoch'], y=train_loss['loss'],
                   mode='lines',
                   name='loss'),
        secondary_y=True,
    )
    fig.show()

    return train_loss, eval_result


def load_trace_info(data_root_dir):
    return pd.read_csv(f'{data_root_dir}/trace_info.csv', index_col=None)


def get_train_eval_test_gids(model_result_root_dir, testset=True):
    fname_train_list = 'model_gid_list_train.txt'
    fname_eval_list = 'model_gid_list_eval.txt'
    fname_test_list = 'model_gid_list_test.txt'

    flist = [fname_train_list, fname_eval_list, fname_test_list]
    if not testset:
        flist = flist[:2]
    gid_list = []
    for fname_lst in flist:
        with open(f'{model_result_root_dir}/{fname_lst}', 'r') as fin:
            _list = [int(i) for i in fin.read().strip().split()]
        gid_list.append(_list)
    return gid_list


def model_output(model, data):
    _out = model(data).cpu().detach().numpy()
    _out = _out.reshape((_out.shape[0], -1))
    _score = model.predict_score(data).cpu().detach().numpy()
    return _out, _score

def get_graph_results(data_root_dir, model, dataset, gid_list, name='train'):
    g_embeddings = []
    g_scores = []
    for gid in tqdm(gid_list):
        _embedding, _score = model_output(model, dataset[gid])
        g_embeddings.append(_embedding)
        g_scores.append(_score)
    g_embeddings = np.array(g_embeddings)

    g_embeddings = g_embeddings.reshape(g_embeddings.shape[0], -1)
    resultdf = pd.DataFrame(g_embeddings, columns=[f'e{i}' for i in range(g_embeddings.shape[1])])
    resultdf['scores'] = g_scores
    resultdf['trace_id'] = gid_list
    resultdf['dataset'] = name
    trace_info_df = load_trace_info(data_root_dir)

    resultdf = resultdf.merge(trace_info_df, on='trace_id', how='inner')
    resultdf = add_tsne_embedding(resultdf, name=name)

    fig = px.scatter(resultdf, x='tsne_x', y='tsne_y', color='error_trace_type', hover_name='trace_id', hover_data=['scores'],
                     title=f'{name} Graph Embeddings - by error trace type')
    fig.show()

    px.histogram(resultdf, x='scores', color='error_trace_type', title=f'{name} Scores').show()

    return resultdf


def add_tsne_embedding(resultdf, name='train', num_feature=7):
    tsne_ = TSNE(n_components=2, init='random')
    embeddings = resultdf[[f'e{i}' for i in range(num_feature)]]
    tsne_embeddings = tsne_.fit_transform(embeddings)

    resultdf['tsne_x'] = tsne_embeddings[:,0]
    resultdf['tsne_y'] = tsne_embeddings[:,1]

    fig = px.scatter(resultdf, x='tsne_x', y='tsne_y', color='trace_bool', hover_name='trace_id', hover_data=['scores'],
                     title=f'{name} Graph Embeddings')
    fig.show()

    return resultdf

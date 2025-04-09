# -- coding: utf-8 --
# author : TangQiang
# time   : 2025/3/15
# email  : tangqiang.0701@gmail.com
# file   : predict.py

import pickle
import dgl
from dgl.nn.pytorch import EdgeWeightNorm
from Bio.SeqIO import parse
from utils.utils import *
from models.cncFinder import cncFinder

def load_feature(info_file, node_file):
    kmers2id, node_features = None, None
    with open(info_file, 'rb') as f:
        kmers2id = pickle.load(f)['kmers2id']
    with open(node_file, 'rb') as f:
        node_features = pickle.load(f)
    return kmers2id, node_features

def seq2graph(idseq, embedding):
    newidseq = []
    old2new = {}
    count = 0
    for oneid in idseq:
        if oneid not in old2new:
            old2new[oneid] = count
            count += 1
        newidseq.append(old2new[oneid])
    counter_uv = Counter(list(zip(newidseq[:-1], newidseq[1:])))
    graph = dgl.graph(list(counter_uv.keys()))
    weight = torch.FloatTensor(list(counter_uv.values()))
    norm = EdgeWeightNorm(norm='both')
    norm_weight = norm(graph, weight)
    graph.edata['weight'] = norm_weight
    node_features = embedding[list(old2new.keys())][:]
    graph.ndata['attr'] = torch.tensor(node_features)
    graph = dgl.reorder_graph(graph)
    return graph

def transform(seq, kmers2id, node_features, kmer):
    idseq = [kmers2id[seq[i: i+kmer]] for i in range(len(seq) - kmer + 1)]
    return seq2graph(idseq, node_features)

def save_weight(file, value):
    with open(file, 'a') as f:
        value = value.detach().cpu().numpy().reshape(1, -1)
        np.savetxt(f, value, fmt='%.4f')

def predict(graph, model):
    h = graph.ndata['attr']
    outputs = model(graph, h)
    y_prob = outputs[:, 1].cpu().detach().numpy()
    y_pred = transfer(y_prob, 0.5)
    return y_prob, y_pred

def get_activation(name):
    def hook(model, input, output):
        save_weight('fc.txt', input[0])
    return hook

def get_model(params):
    model_list = []
    for i in range(1, params.kFold+1):
        model_file = params.model_dir / f'GraphLncPro_{i}.pth'
        model = cncFinder(params)
        model.load_model(directory=model_file)
        model = model.model
        #model.classifier[9].register_forward_hook(get_activation('fc'))
        model.eval()
        model_list.append(model)
    return model_list

def cv_predict(graph, model_list):
    y_prob_list, y_pred_list = [], []
    for model in model_list:
        y_prob, y_pred = predict(graph, model)
        y_prob_list.append(y_prob[0])
        y_pred_list.append(y_pred[0])

    len_model = len(model_list)
    y_prob = round(sum(y_prob_list) / len_model, 4) # mean
    y_pred = 1 if sum(y_pred_list) > (len_model//2) else 0 # vote
    return y_prob, y_pred


def model_predict(infile, outfile, params):
    records = parse(infile, 'fasta')
    ids, seqs = [], []
    for one in records:
        ids.append(str(one.id))
        seqs.append(str(one.seq).upper())

    model_list = get_model(params)

    graph_path = params.graph_dir
    info_file = graph_path / f'{params.kmer}kmer_{params.w2v_dim}w2v_info.pkl'
    node_file = graph_path / f'w2v_{params.kmer}kmer_{params.w2v_dim}w2v_info.pkl'
    kmers2id, node_features = load_feature(info_file, node_file)

    y_prob_list, y_pred_list = [], []
    for seq in seqs:
        graph = transform(seq, kmers2id, node_features, params.kmer)
        graph = graph.to(params.device)
        y_prob, y_pred = cv_predict(graph, model_list)
        y_prob_list.append(y_prob)
        y_pred_list.append(y_pred)

    with open(outfile, 'w') as f:
        for i in range(len(ids)):
            f.write(str(y_prob_list[i])+','+ids[i]+'\n')

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage:\npython predict_user.py input_file output_file")
        exit(0)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    params = config()
    model_predict(infile, outfile, params)
    print('predict ok')
    

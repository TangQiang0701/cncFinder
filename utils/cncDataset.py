# -- coding: utf-8 --
# author : TangQiang
# time   : 2025/3/15
# email  : tangqiang.0701@gmail.com
# file   : LncProDataset.py


import dgl
from dgl.data import DGLDataset, save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
from dgl.nn.pytorch import EdgeWeightNorm
from gensim.models import Word2Vec

import pickle
import itertools
import pandas as pd
from collections import Counter
import networkx as nx
import matplotlib.pylab as plt

from utils.config import *
from utils.nodeFeature import nFeatures

params = config()

class cncDataset(DGLDataset):
    def __init__(self, url=None, raw_dir=None, save_dir=None, force_reload=False, verbose=False):
        super(cncDataset, self).__init__(
            name='lncpro',
            url=url,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose
        )
        logging.info('Executing function fllowing download, process, save, load')
        logging.info('Dataset process end!')

    def process(self):
        logging.info('Executing process function')
        self.kmers = params.kmer
        logging.info('Loading the raw data for {}'.format(self.raw_dir))
        df = pd.read_csv(self.raw_dir)
        rnas = df.rna.str.upper().tolist() #all rna sequences
        labels = df.label.tolist() # rna label
        k_rnas = [[r[j:j+self.kmers] for j in range(len(r)-self.kmers+1)] for r in rnas] # rna to kmers
        logging.info('Mapping variables for kmers and kmers id')
        self.kmers2id, self.id2kmers = self.make_kmer()
        self.k_rans = k_rnas
        self.labels = torch.tensor(labels)
        self.id_seq = np.array([[self.kmers2id[i] for i in r] for r in self.k_rans], dtype=object)

        self.node_features(method='w2v')

        self.graphs = []
        self.new2old = []
        for oneseq in self.id_seq:
            new_id_seq = []
            old2new = {}
            new2old = {}
            count = 0
            for oneid in oneseq:
                if oneid not in old2new:
                    old2new[oneid] = count
                    new2old[count] = oneid
                    count += 1
                new_id_seq.append(old2new[oneid])
            counter_uv = Counter(list(zip(new_id_seq[:-1], new_id_seq[1:])))
            graph = dgl.graph(list(counter_uv.keys()))
            weight = torch.FloatTensor(list(counter_uv.values()))
            norm = EdgeWeightNorm(norm='both')
            norm_weight = norm(graph, weight)
            graph.edata['weight'] = norm_weight
            node_features = self.vector['embedding'][list(old2new.keys())]
            graph.ndata['attr'] = torch.tensor(node_features)
            graph = dgl.reorder_graph(graph)
            self.graphs.append(graph)
            self.new2old.append(new2old)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        graph_file = f'{self.save_dir}/{params.kmer}kmer_{params.w2v_dim}w2v.bin'
        info_file = f'{self.save_dir}/{params.kmer}kmer_{params.w2v_dim}w2v_info.pkl'
        logging.info('save graph in {}'.format(graph_file))
        save_graphs(graph_file, self.graphs, {'labels': self.labels})
        info = {'kmers2id':self.kmers2id}
        logging.info('save info in {}'.format(info_file))
        save_info(info_file, info)

    def load(self):
        graph_file = f'{self.save_dir}/{params.kmer}kmer_{params.w2v_dim}w2v.bin'
        info_file = f'{self.save_dir}/{params.kmer}kmer_{params.w2v_dim}.w2v_info.pkl'
        logging.info('load graph from {}'.format(graph_file))
        self.graphs, label_dict = load_graphs(graph_file)
        self.labels = label_dict['labels']
        logging.info('load info from {}'.format(info_file))
        info = load_info(info_file)
        self.kmers2id = info['kmers2id']

    def has_cache(self):
        graph_file = f'{self.save_dir}/{params.kmer}kmer_{params.w2v_dim}w2v.bin'
        info_file = f'{self.save_dir}/{params.kmer}kmer_{params.w2v_dim}w2v_info.pkl'
        return os.path.exists(graph_file) and os.path.exists(info_file)

    def node_features(self, method='phy'):
        self.vector = {}
        if method == 'w2v':
            info_file = f'{self.save_dir}/w2v_{params.kmer}kmer_{params.w2v_dim}w2v_info.pkl'
        if method == 'phy':
            info_file = f'{self.save_dir}/phy_{params.kmer}kmer_info.pkl'
        if os.path.exists(info_file) and not self._force_reload:
            with open(info_file, 'rb') as f:
                self.vector['embedding'] = pickle.load(f)
            logging.info('load cache from {}'.format(info_file))
            return
        if method == 'w2v':
            doc = [i for i in self.k_rans]
            doc += [self.id2kmers]
            len_kmer = len(self.id2kmers)
            model = Word2Vec(doc, min_count=0, vector_size=params.w2v_dim, sg=0, seed=params.seed)
            word2vec = np.zeros((len_kmer, params.w2v_dim), dtype=np.float32)
            for i in range(len_kmer):
                word2vec[i] = model.wv[self.id2kmers[i]]
            self.vector['embedding'] = word2vec

        if method == 'phy':
            len_kmer = len(self.id2kmers)
            f_div = params.n_fea
            features = np.zeros((len_kmer, f_div), dtype=np.float32)
            for i in range(len_kmer):
                kmer = self.id2kmers[i]
                features[i] = np.asarray(nFeatures(kmer), dtype=np.float32)
            self.vector['embedding'] = features

        with open(info_file, 'wb') as f:
            pickle.dump(self.vector['embedding'], f, protocol=4)

    def show(self, idx):
        graph = self.graphs[idx]
        nodes_new = graph.nodes().tolist()
        new2old = self.new2old[idx]
        nodes = [new2old.get(i) for i in nodes_new]
        node_data = [self.id2kmers[i] for i in nodes]
        plt.figure(figsize=(8,8))
        G = graph.to_networkx(edge_attrs=['weight'])
        pos = nx.spring_layout(G)
        nx.draw(G, pos, edge_color='grey', node_size=500, with_labels=True)
        node_labels = {index: str(data) for index, data in enumerate(node_data)}
        pos_higher = {}

        for k, v in pos.items():
            if (v[1] > 0):
                pos_higher[k] = (v[0] - 0.04, v[1] + 0.04)
            else:
                pos_higher[k] = (v[0] - 0.04, v[1] - 0.04)
        nx.draw_networkx_labels(G, pos_higher, labels=node_labels, font_color="brown", font_size=12)
        edge_labels = nx.get_edge_attributes(G, 'weight')

        edge_labels = {(key[0], key[1]): "w:" + str(edge_labels[key].item()) for key in
                       edge_labels}  # 重新组合数据， 边的标签是dict, {(nodeid1,nodeid2):value,...} 这样的形式
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)  # 将Weights属性，显示在边上
        plt.show()

    def make_kmer(self):
        rna = ['A', 'G', 'C', 'T']
        kmers2id, id2kmers = {}, []
        for id, el in enumerate(itertools.product(rna, repeat=params.kmer)):
            k = list(el)
            key = ''.join(k)
            id2kmers.append(key)
            kmers2id[key] = id
        return kmers2id, id2kmers

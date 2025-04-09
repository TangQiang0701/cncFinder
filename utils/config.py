# -- coding: utf-8 --
# author : TangQiang
# time   : 2025/3/15
# email  : tangqiang.0701@gmail.com
# file   : config.py

import os
import torch
import random
import numpy as np
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')

class config(object):
    def __init__(self):
        super(config, self).__init__()
        self.kmer = 3
        self.w2v_dim = 64
        self.hidden_dim = 64
        self.n_fea = 4 * self.kmer + 5 * (self.kmer - 1) + 240 + 704 + 64
        self.n_heads = 3 
        self.drop = 0.3
        self.seed = 2023
        self.bs = 256
        self.n_epochs = 500
        self.lr = 0.0001
        self.kFold = 5
        self.earlyStop = 10
        self.reload = True
        self.upper_sample = True
        self.base_dir =  Path(__file__).resolve().parent.parent
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.make_dir()
        self.set_loggging()

    def make_dir(self):
        checkpoint = self.base_dir / 'checkpoint'
        self.graph_dir = checkpoint / f'graph/{self.kmer}kmer_{self.w2v_dim}w2v'
        self.model_dir = checkpoint / f'model/{self.kmer}kmer_{self.w2v_dim}w2v_{self.hidden_dim}hidden_{self.n_heads}head_{self.bs}bs_{self.lr}lr_{self.drop}drop'
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not self.reload and os.path.exists(checkpoint):
            import shutil
            shutil.rmtree(checkpoint)

    def set_loggging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            filename=f'{self.model_dir}/cnc.log',
            filemode='w'
        )

def set_seed(seed):
    logging.info('set seed {} for everything'.format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

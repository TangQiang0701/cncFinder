# -- coding: utf-8 --
# author : TangQiang
# time   : 2025/3/15
# email  : tangqiang.0701@gmail.com
# file   : train.py

import datetime

from utils.cncDataset import *
from models.cncFinder import *

start = datetime.datetime.now()
params = config()

train_pos = pd.read_csv('./data/train_pos.csv')
train_neg = pd.read_csv('./data/train_neg.csv')

if params.upper_sample and not os.path.exists('./data/train.csv'):
    train_pos_sampler = upper_sample(train_pos, train_neg.shape[0], seed=1024)
    ndf = pd.concat([train_pos_sampler, train_neg])
    ndf.to_csv('./data/train.csv', index=False)
dataset = cncDataset(raw_dir='./data/train.csv', save_dir=params.graph_dir, force_reload=False)

model = cncFinder(params=params)
model.cv_train(dataset, bs=params.bs, epochs=params.n_epochs, earlyStop=params.earlyStop)

end = datetime.datetime.now()
logging.info(f'Total running time of all train is {(end-start).seconds}s.')



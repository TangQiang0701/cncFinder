# -- coding: utf-8 --
# author : TangQiang
# time   : 2025/3/15
# email  : tangqiang.0701@gmail.com
# file   : cncFinder.py

import glob
import time
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

from .classifier import Classifier
from utils.config import *
from utils.utils import *

class cncFinder(object):
    def __init__(self, params):
        super(cncFinder, self).__init__()
        self.in_dim = params.w2v_dim
        self.hidden_dim = params.hidden_dim
        self.n_heads = params.n_heads
        self.device = params.device
        self.drop = params.drop
        self.lr = params.lr
        self.checkpoint = params.model_dir
        self.l_mean = lambda l: sum(l) / len(l)
        self.build_model()

    def build_model(self):
        logging.info(f'build model with params:\nin_dim: {self.in_dim}\nhidden_dim: {self.hidden_dim}\n'
                     f'n_heads: {self.n_heads}\ndevice: {self.device}\n')
        self.model = Classifier(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            device=self.device,
            drop=self.drop)
        self.criterion = torch.nn.CrossEntropyLoss()
        logging.info(f'model info:\n{self.model}')

    def save_model(self, kFlod):
        torch.save(self.model.eval().state_dict(), os.path.join(self.checkpoint, f'GraphLncPro_{kFlod}.pth'))

    def load_model(self, directory=None):
        if directory is None:
            directory = self.checkpoint + 'GraphLncPro_1.pth'
        if not os.path.exists(directory):
            logging.warning('Checkpoint not found! Starting from scratch.')
            return 0
        logging.info(f'Loading model from {directory}')
        self.model.load_state_dict(torch.load(directory, map_location=self.device))

    def cv_train(self, dataset, bs, epochs, kFlod=5, earlyStop=10, seed=2023):
        splits = StratifiedKFold(n_splits=kFlod, shuffle=True, random_state=seed)
        fold_best = []
        for fold, (train_idx, val_idx) in enumerate(splits.split(dataset[:][0], dataset[:][1])):
            set_seed(fold + 1)
            self.model.reset_parameters()

            best_acc = 0.0
            logging.info(f'begin train with fold {fold + 1}')
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(val_idx)

            train_loader = GraphDataLoader(dataset, batch_size=bs, sampler=train_sampler, num_workers=4, pin_memory=True)
            valid_loader = GraphDataLoader(dataset, batch_size=bs, sampler=valid_sampler, num_workers=4, pin_memory=True)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-07)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

            best_record = {'train_loss': 0, 'test_loss': 0, 'train_acc': 0, 'test_acc': 0, 'train_f': 0,
                           'test_f': 0, 'train_pre': 0, 'test_pre': 0, 'train_rec': 0, 'test_rec': 0,
                           'train_roc': 0, 'test_roc': 0}
            nobetter = 0
            for epoch in range(1, epochs+1):
                # metrics --> auc, sn, sp, acc, mcc
                ys_train, loss_train, metrics_train, time_epoch = self.train_epoch(train_loader, optimizer)
                loss_train = np.mean(loss_train)
                logging.info(f'training: Epoch-{epoch}/{epochs} | loss={loss_train:.4f} | time={time_epoch:.4f} min')
                ys_valid, loss_valid, metrics_valid, time_epoch = self.valid_epoch(valid_loader)
                loss_valid = np.mean(loss_valid)
                logging.info(f'validing: Epoch-{epoch}/{epochs} | loss={loss_valid:.4f} | time={time_epoch:.4f} min')

                if best_acc < metrics_valid[3]:
                    nobetter = 0
                    best_acc = metrics_valid[3]
                    best_record['valid_loss'] = loss_valid
                    best_record['valid_auc'] = metrics_valid[0]
                    best_record['valid_sn'] = metrics_valid[1]
                    best_record['valid_sp'] = metrics_valid[2]
                    best_record['valid_acc'] = metrics_valid[3]
                    best_record['valid_mcc'] = metrics_valid[4]
                    best_record['train_loss'] = loss_train
                    best_record['train_auc'] = metrics_train[0]
                    best_record['train_sn'] = metrics_train[1]
                    best_record['train_sp'] = metrics_train[2]
                    best_record['train_acc'] = metrics_train[3]
                    best_record['train_mcc'] = metrics_train[4]
                    logging.info('Get a better model with acc {0:.4f}'.format(best_acc))
                    self.save_model(kFlod=fold+1)
                else:
                    nobetter += 1
                    if nobetter >= earlyStop:
                        logging.info(f'validing acc has not improved for more '
                                     f'than {earlyStop} steps in epoch {epoch}, stop training')
                        break
            fold_best.append(best_record)
            logging.info(f'cv fold {kFlod} for fold {fold + 1} done')
            logging.info(f'Find best model, valid auc:{best_record["valid_auc"]:.3f},  '
                         f'sn:{best_record["valid_sn"]:.3f},  sp:{best_record["valid_sp"]:.3f}, '
                         f'acc:{best_record["valid_acc"]:.3f}, mcc:{best_record["valid_mcc"]:.3f}')
        logging.info('all folds are done')
        row_first = ['Fold', 'auc', 'sn', 'sp', 'acc', 'mcc']
        logging.info(''.join(f'{item:<12}' for item in row_first))
        metrics = ['valid_auc', 'valid_sn', 'valid_sp', 'valid_acc', 'valid_mcc']
        for idx, fold in enumerate(fold_best):
            logging.info(f'{idx+1:<12}' + ''.join(
                f'{fold[key]:<12.3f}' for key in metrics
            ))
        avg = {}
        for item in metrics:
            avg[item] = 0
            for fold in fold_best:
                avg[item] += fold[item]
            avg[item] /= kFlod
        logging.info(f'%-12s' % 'Average' + ''.join(f'{avg[key]:<12.3f}' for key in metrics))

    def train_epoch(self, data_loader, optimizer):
        self.model.train()
        y_true_list, y_prob_list, loss_list = [], [], []
        train_start = time.time()
        for batch_graph, labels in tqdm(data_loader, mininterval=1, desc='Training Processing', leave=False):
            batch_graph, labels = batch_graph.to(self.device), labels.to(self.device)
            feats = batch_graph.ndata['attr']

            optimizer.zero_grad()
            outputs = self.model(batch_graph, feats)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            y_train = labels.cpu().detach().numpy()
            y_prob = outputs[:, 1].cpu().detach().numpy()
            loss_trian = loss.cpu().detach().numpy()
            y_true_list.extend(y_train)
            y_prob_list.extend(y_prob)
            loss_list.append(loss_trian)
        time_epoch = (time.time() - train_start) / 60
        y_pred_list = transfer(y_prob_list, 0.5)
        ys_train = (y_true_list, y_pred_list, y_prob_list)
        metrics_train = cal_performance(y_true_list, y_pred_list, y_prob_list, logging_=True)
        return ys_train, loss_list, metrics_train, time_epoch

    def valid_epoch(self, data_loader):
        train_start = time.time()
        y_true_list, y_prob_list, loss_list = [], [], []
        with torch.no_grad():
            self.model.eval()
            for batch_graph, labels in tqdm(data_loader, mininterval=1, desc='Validing Processing', leave=False):
                batch_graph, labels = batch_graph.to(self.device), labels.to(self.device)
                feats = batch_graph.ndata['attr']

                outputs = self.model(batch_graph, feats)
                loss = self.criterion(outputs, labels)
                y_train = labels.cpu().detach().numpy()
                y_prob = outputs[:, 1].cpu().detach().numpy()
                loss_trian = loss.cpu().detach().numpy()
                y_true_list.extend(y_train)
                y_prob_list.extend(y_prob)
                loss_list.append(loss_trian)
            time_epoch = (time.time() - train_start) / 60
            y_pred_list = transfer(y_prob_list, 0.5)
            ys_train = (y_true_list, y_pred_list, y_prob_list)
            metrics_train = cal_performance(y_true_list, y_pred_list, y_prob_list, logging_=True)
            return ys_train, loss_list, metrics_train, time_epoch

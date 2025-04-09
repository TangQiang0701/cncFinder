# -- coding: utf-8 --
# author : TangQiang
# time   : 2025/3/15
# email  : tangqiang.0701@gmail.com
# file   : utils.py


from sklearn import metrics
from collections import Counter
from .config import *


def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])

def cal_performance(y_true, y_pred, y_prob, logging_=True):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    acc = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    try:
        sn = tp / (tp + fn)
    except:
        sn = 0
    try:
        sp = tn / (tn + fp)
    except:
        sp = 0
    try:
        auc = metrics.roc_auc_score(y_true, y_prob)
    except:
        auc = 0
    if logging_:
        logging.info("tn={0}, fp={1}, fn={2}, tp={3}".format(tn, fp, fn, tp))
        logging.info("y_pred: 0 = {} | 1 = {}".format(Counter(y_pred)[0], Counter(y_pred)[1]))
        logging.info("y_true: 0 = {} | 1 = {}".format(Counter(y_true)[0], Counter(y_true)[1]))
        logging.info("auc={0:.4f}|sn={1:.4f}|sp={2:.4f}|acc={3:.4f}|mcc={4:.4f}".format(auc, sn, sp, acc, mcc))
    return (auc, sn, sp, acc, mcc)

def cal_apr(y_prob):
    logging.info("the num of all pos test is {0}".format(len(y_prob)))
    def get_recall(score):
        x_list = []
        for thres in score_thres:
            x = len([i for i in score if i > thres])
            x_list.append(x / len(score))

        return x_list
    test = 13
    mouse = 19
    daniorerio = 6
    score_t = y_prob[:test]
    thp1 = score_t[:8]
    hl60 = score_t[:3] + score_t[4:10]
    k562 = score_t[:4] + score_t[6:8] + score_t[9:]
    score_m = y_prob[test:test+mouse]
    score_d = y_prob[test+mouse:]
    logging.info("the num test={0}, mouse={1}, daniorerio={2}".format(len(score_t), len(score_m), len(score_d)))

    score_thres = np.linspace(1, 0, 5000)
    y_t = get_recall(score_t)
    area_t = -1 * np.trapz(x=score_thres, y=y_t)

    y_m = get_recall(score_m)
    area_m = -1 * np.trapz(x=score_thres, y=y_m)

    y_d = get_recall(score_d)
    area_d = -1 * np.trapz(x=score_thres, y=y_d)

    y_thp1 = get_recall(thp1)
    area_thp1 = -1 * np.trapz(x=score_thres, y=y_thp1)

    y_hl60 = get_recall(hl60)
    area_hl60 = -1 * np.trapz(x=score_thres, y=y_hl60)

    y_k562 = get_recall(k562)
    area_k562 = -1 * np.trapz(x=score_thres, y=y_k562)


    logging.info("apr test={0:.4f}|mouse={1:.4f}|daniorerio={2:.4f}|HL60={3:.4f}|K562={4:.4f}|THP1={5:.4f}".format(area_t, area_m, area_d, area_hl60, area_k562, area_thp1))
    return area_t, area_m, area_d, area_hl60, area_k562, area_thp1


def upper_sample(df, num, seed=1024):
    set_seed(seed)
    ndf = df.sample(n=num, replace=True, random_state=seed)
    return ndf

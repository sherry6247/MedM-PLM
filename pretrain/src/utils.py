'''
Author: your name
Date: 2021-09-06 10:42:10
LastEditTime: 2021-09-24 09:52:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /model/lsc_code/preTrain_in_MIMIC-III/code/utils.py
'''
import random
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, jaccard_similarity_score, \
    precision_recall_curve, confusion_matrix, auc
import numpy as np
import os
import logging

from sklearn.metrics.classification import precision_score

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(
                y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(
                y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    auc = roc_auc(y_gt, y_prob)
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_pred)
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def metric_report(y_pred, y_true, therhold=0.5):
    y_prob = y_pred.copy()
    y_pred[y_pred > therhold] = 1
    y_pred[y_pred <= therhold] = 0

    acc_container = {}
    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(
        y_true, y_pred, y_prob)
    acc_container['jaccard'] = ja
    acc_container['f1'] = avg_f1
    acc_container['prauc'] = prauc

    # acc_container['jaccard'] = jaccard_similarity_score(y_true, y_pred)
    # acc_container['f1'] = f1(y_true, y_pred)
    # acc_container['auc'] = roc_auc(y_true, y_prob)
    # acc_container['prauc'] = precision_auc(y_true, y_prob)

    for k, v in acc_container.items():
        logger.info('%-10s : %-10.4f' % (k, v))

    return acc_container

def binary_metric_reporter(y_true, y_pred, therhold=0.5):
    y_prob = y_pred.copy()
    y_pred[y_pred > therhold] = 1
    y_pred[y_pred <= therhold] = 0

    acc_container = {}
    acc_container['jaccard'] = jaccard_similarity_score(y_true, y_pred)
    acc_container['f1'] = f1_score(y_true, y_pred)
    acc_container['auc'] = roc_auc_score(y_true, y_prob)
    acc_container['prauc'] = average_precision_score(y_true, y_prob)

    for k, v in acc_container.items():
        logger.info('%-10s : %-10.4f' % (k, v))

    return acc_container

def print_metrics_binary(y_true, predictions, type=None):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = confusion_matrix(y_true, predictions.argmax(axis=1))
    
    print("{} confusion matrix:".format(type))
    print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = precision_recall_curve(y_true, predictions[:, 1])
    auprc = auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    f1 = f1_score(y_true, predictions.argmax(axis=1))
    jac = jaccard_similarity_score(y_true, predictions.argmax(axis=1))
    print("================================{}================================".format(type))
    print("accuracy = {}".format(acc))
    print("precision class 0 = {}".format(prec0))
    print("precision class 1 = {}".format(prec1))
    print("recall class 0 = {}".format(rec0))
    print("recall class 1 = {}".format(rec1))
    print("AUC of ROC = {}".format(auroc))
    print("AUC of PRC = {}".format(auprc))
    print("min(+P, Se) = {}".format(minpse))
    print("F1 = {}".format(f1))

    return acc, auroc, auprc, f1, jac, cf

def acc_report(y_pred, y_true, therhold=0.5, type='train'):
    y_pred = np.stack([1-y_pred, y_pred], axis=1)
    y_pred_hot = y_pred.argmax(axis=1)
    acc_count = 0
    total_count = len(y_true)

    for pred, true in zip(y_pred_hot, y_true):
        if pred == true:
            acc_count += 1
    acc = acc_count/total_count
    
    acc_container = {}
    acc_container['acc'] = acc
    
    acc, auroc, auprc, f1, jac, cf = print_metrics_binary(
        y_true, y_pred, type)
    acc_container['jaccard'] = jac
    acc_container['auc'] = auroc
    acc_container['f1'] = f1
    acc_container['prauc'] = auprc

    # acc_container['jaccard'] = jaccard_similarity_score(y_true, y_pred)
    # acc_container['f1'] = f1(y_true, y_pred)
    # acc_container['auc'] = roc_auc(y_true, y_prob)
    # acc_container['prauc'] = precision_auc(y_true, y_prob)

    for k, v in acc_container.items():
        logger.info('%-10s : %-10.4f' % (k, v))

    return acc_container, cf

def t2n(x):
    return x.detach().cpu().numpy()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

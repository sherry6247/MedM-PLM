from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import dill
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from tensorboardX import SummaryWriter

from utils import metric_report, t2n, get_n_params, binary_metric_reporter
from fusion_config import FuseBertConfig as BertConfig
from fusion_predictive_models import FusionBert_ML_predict, FusionBert_binary_predict

from transformers import BertTokenizer, BertModel
import torch.nn as nn
import json
import pickle

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

class EHRTokenizer(object):
    """Runs end-to-end tokenization"""

    def __init__(self, data_dir, special_tokens=("[PAD]", "[CLS]", "[MASK]")):

        self.vocab = Voc()

        # special tokens
        self.vocab.add_sentence(special_tokens)

        self.rx_voc = self.add_vocab(os.path.join(data_dir, 'rx-vocab.txt'))
        self.dx_voc = self.add_vocab(os.path.join(data_dir, 'dx-vocab.txt'))

    def add_vocab(self, vocab_file):
        voc = self.vocab
        specific_voc = Voc()
        with open(vocab_file, 'r') as fin:
            for code in fin:
                voc.add_sentence([code.rstrip('\n')])
                specific_voc.add_sentence([code.rstrip('\n')])
        return specific_voc

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens

class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, code_input_ids, code_type_ids, code_visit_length, text_ids, text_type_ids, text_att_mask, y):
        self.code_input_ids = np.array(code_input_ids)
        self.code_type_ids = np.array(code_type_ids)
        self.code_visit_length = np.array(code_visit_length)
        self.text_input_ids = np.array(text_ids)
        self.text_type_ids = np.array(text_type_ids)
        self.text_att_mask = np.array(text_att_mask)
        self.y = np.array(y)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, dx_code=None, rx_code=None, text=None, label_id=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            dx_code: for the diagnosis code of the example
            rx_code: for the ATC code of the example
            text: for the discharge summary text of the example
        """
        self.guid = guid
        self.dx_code = dx_code
        self.rx_code = rx_code
        self.text = text
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_pickle(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # load data
        data_multi = pd.read_pickle(os.path.join(
            input_file, 'data-multi-visit.pkl'))#
        data_single = pd.read_pickle(
            os.path.join(input_file, 'data-single-visit.pkl'))#

        return data_multi, data_single
        # return data_multi[:200], data_single[:50]

class binarylabelProcessor(DataProcessor):
    """Processor for the code and text data ."""
    def get_set_data(self,data):
        data_list = []
        for d in data:
            data_list.extend(d)
        set_data = list(set(data_list))
        set_data.sort(key=data_list.index)
        return set_data
    
    def get_discharge_summary_txt(self, data):
        dis_txt = []
        for d in data:
            cate, txt  = d
            if cate == 'Discharge summary':
                dis_txt.append(txt)
        
        if len(dis_txt) == 0:
            discharge_txt = ' '
        else:
            #  只用最后一个时刻的出院小结
            discharge_txt = dis_txt[-1]#' '.join(dis_txt)
        return discharge_txt    

    def split_dataset(self, data_dir):
        data_dict = {}
        data_multi, data_single = self._read_pickle(data_dir)
        data_multi['Set_NDC'] = data_multi.NDC.map(lambda x: self.get_set_data(x))
        data_multi['discharge_summary'] = data_multi.CATEGORY_TEXT.map(lambda x: self.get_discharge_summary_txt(x))

        data_single['Set_NDC'] = data_single.NDC.map(lambda x: self.get_set_data(x))
        data_single['discharge_summary'] = data_single.CATEGORY_TEXT.map(lambda x: self.get_discharge_summary_txt(x))

        # load trian, eval, test data
        ids_file = [os.path.join(data_dir, 'train-id.txt'),
                    os.path.join(data_dir, 'dev-id.txt'),
                    os.path.join(data_dir, 'test-id.txt')]

        def load_ids(data, file_name):
            """
            :param data: multi-visit data
            :param file_name:
            :return: raw data form
            """
            ids = []
            with open(file_name, 'r') as f:
                for line in f:
                    ids.append(int(line.rstrip('\n')))
            return data[data['SUBJECT_ID'].isin(ids)].reset_index(drop=True)
        data_dict['train_data'] = pd.concat([data_single, load_ids(data_multi, ids_file[0])])
        data_dict['dev_data'] = load_ids(data_multi, ids_file[1])
        data_dict['test_data'] = load_ids(data_multi, ids_file[2])

        return data_dict

    def get_train_examples(self, data_dir, label_type):
        """See base class."""
        logger.info("LOOKING AT {} and {}".format(os.path.join(data_dir, "data-multi-visit.pkl"), os.path.join(data_dir, "data-single-visit.pkl")))
        
        return self._create_examples(
            self.split_dataset(data_dir)['train_data'], "train", label_type)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.split_dataset(data_dir)['dev_data'], "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.split_dataset(data_dir)['test_data'], "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, pd_data, set_type, label_type):
        """Creates examples for the training and dev sets. READMISSION_LABEL or HOSPITAL_EXPIRE_FLAG"""
        examples = []
        for i, row in tqdm(pd_data.iterrows(), desc=set_type, ncols=100):
            guid = "%s-%s" % (set_type, i)
            dx_code = list(row['DIAG_CODE'])
            rx_code = row['Set_NDC']
            text = row['discharge_summary']
            label_id = int(row[label_type])
            examples.append(
                InputExample(guid=guid, dx_code=dx_code, rx_code=rx_code, text=text, label_id=label_id))
        return examples
    
class MultilabelProcessor(DataProcessor):
    """Processor for the code and text data with multi-label targets ."""
    def get_set_data(self,data):
        data_list = []
        for d in data:
            data_list.extend(d)
        set_data = list(set(data_list))
        set_data.sort(key=data_list.index)
        return set_data
    
    def get_discharge_summary_txt(self, data):
        dis_txt = []
        for d in data:
            cate, txt  = d
            if cate == 'Discharge summary':
                dis_txt.append(txt)
        
        if len(dis_txt) == 0:
            discharge_txt = ' '
        else:
            #  只用最后一个时刻的出院小结
            discharge_txt = dis_txt[-1]#' '.join(dis_txt)
        return discharge_txt    

    def split_dataset(self, data_dir):
        data_dict = {}
        data_multi, data_single = self._read_pickle(data_dir)
        data_multi['Set_NDC'] = data_multi.NDC.map(lambda x: self.get_set_data(x))
        data_multi['discharge_summary'] = data_multi.CATEGORY_TEXT.map(lambda x: self.get_discharge_summary_txt(x))

        # load trian, eval, test data
        ids_file = [os.path.join(data_dir, 'train-id.txt'),
                    os.path.join(data_dir, 'dev-id.txt'),
                    os.path.join(data_dir, 'test-id.txt')]

        def load_ids(data, file_name):
            """
            :param data: multi-visit data
            :param file_name:
            :return: raw data form
            """
            ids = []
            with open(file_name, 'r') as f:
                for line in f:
                    ids.append(int(line.rstrip('\n')))
            return data[data['SUBJECT_ID'].isin(ids)].reset_index(drop=True)
        train_data = load_ids(data_multi, ids_file[0])
        data_dict['train_data'] = train_data
        data_dict['dev_data'] = load_ids(data_multi, ids_file[1])
        data_dict['test_data'] = load_ids(data_multi, ids_file[2])

        return data_dict

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} ".format(os.path.join(data_dir, "data-multi-visit.pkl")))
        
        return self._create_examples(
            self.split_dataset(data_dir)['train_data'], "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.split_dataset(data_dir)['dev_data'], "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.split_dataset(data_dir)['test_data'], "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, pd_data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        selected_patient = pd_data[pd_data.SUBJECT_ID.isin([21786,67624,72940])]
        for idx, subject_id in tqdm(enumerate(selected_patient['SUBJECT_ID'].unique()), desc=set_type, ncols=100):
            item_df = pd_data[pd_data.SUBJECT_ID==subject_id]
            guid = "%s-%s" % (set_type, subject_id)
            dx_code_list = []
            rx_code_list = []
            text_list = []
            for _, row in item_df.iterrows():
                dx_code = list(row['DIAG_CODE'])
                rx_code = row['Set_NDC']
                text = row['discharge_summary']
                dx_code_list.append(dx_code)
                rx_code_list.append(rx_code)
                text_list.append(text)
            examples.append(
                InputExample(guid=guid, dx_code=dx_code_list, rx_code=rx_code_list, text=text_list)
            )
        return examples

def convert_examples_to_features(examples, txt_tokenizer, code_tokenizer, seq_len, max_visit_len, predict_task=None, gbert_type_data=False, partion=1.0):

    features = []
    max_len = []
    for (ex_index, example) in tqdm(enumerate(examples[:int(len(examples)*partion)]), desc='convert_examples_to_features', ncols=100):
        def fill_to_max(l, seq):
            t = l.copy()
            while len(t) < seq:
                t.append('[PAD]')
            return t
        
        """extract input and output tokens
        """
        input_tokens_list = []
        code_type_id_list =[]
        txt_input_tokens = []
        y_list = []
        for idx in range(min(len(example.dx_code), max_visit_len)):
            dx_record = example.dx_code[idx]
            rx_record = example.rx_code[idx]
            
            if gbert_type_data:
                code_tokens = ['[CLS]'] + fill_to_max(dx_record, seq_len - 1)
                code_tokens.extend(['[CLS]'] + fill_to_max(rx_record, seq_len - 1))
                code_type_ids = [0] * len(['[CLS]'] + fill_to_max(dx_record, seq_len - 1)) + [1] * len(['[CLS]'] + fill_to_max(rx_record, seq_len - 1))
            else:
                _truncate_seq_pair(rx_record, dx_record, seq_len*2)
                code_a = ['[CLS]'] + rx_record + ['[MASK]'] # 用[MASK]代替[SEP]是因为Gbert预训练的时候没有加入[SEP]这个特殊词
                code_b = dx_record + ['[MASK]']
                code_tokens = code_a + code_b
                code_type_ids = [0] + (len(rx_record) +1) * [0] + [1] * (len(dx_record)+1)
            text_reocrd = example.text[idx]
            input_tokens_list.append(code_tokens)
            txt_input_tokens.append(text_reocrd)
            code_type_id_list.append(code_type_ids)

            if example.label_id is None:
                if idx != 0:
                    if predict_task == 'rx': 
                        y_list.append(example.rx_code[idx])
                    elif predict_task == 'dx':
                        y_list.append(example.dx_code[idx])
                
        """convert tokens to id
        """
        code_input_ids = [code_tokenizer.convert_tokens_to_ids(input_tokens) for input_tokens in input_tokens_list]
        tokens_text = txt_tokenizer(txt_input_tokens, padding="max_length", max_length=512, truncation=True)
        txt_input_ids = tokens_text['input_ids']
        txt_token_type_ids = tokens_text['token_type_ids']
        txt_attention_mask = tokens_text['attention_mask']

        """
        Zero_pad up to sequence length
        """
        if gbert_type_data:
            pass
        else:
            for i, (cii, typeid) in enumerate(zip(code_input_ids, code_type_id_list)):
                padding = [0] * (seq_len*2+2 - len(cii))
                cii += padding
                typeid += padding

        if example.label_id is None:
            y = []
            if predict_task == 'rx':
                voc_size = len(code_tokenizer.rx_voc.word2idx)
                for tokens in y_list:
                    tmp_labels = np.zeros(voc_size)
                    tmp_labels[list(
                        map(lambda x: code_tokenizer.rx_voc.word2idx[x], tokens))] = 1
                    y.append(tmp_labels)
            elif predict_task == 'dx': 
                voc_size = len(code_tokenizer.dx_voc.word2idx)
                for tokens in y_list:
                    tmp_labels = np.zeros(voc_size)
                    tmp_labels[list(
                        map(lambda x: code_tokenizer.dx_voc.word2idx[x], tokens))] = 1
                    y.append(tmp_labels)
                 
                    
        else:
            y = example.label_id        
            
        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("code_tokens: %s" % " ".join(
                    [str(x) for x in code_input_ids]))

        code_visit_length = [min(len(example.rx_code)-1, max_visit_len-1)]
        
        features.append(
            InputFeature(code_input_ids=code_input_ids,
                        code_type_ids=code_type_id_list,
                        code_visit_length=code_visit_length,
                        text_ids=txt_input_ids,
                        text_type_ids=txt_token_type_ids,
                        text_att_mask=txt_attention_mask,
                        y=y))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class EHR_Text_DataSet(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        code_input_ids = self.features[index].code_input_ids
        code_type_ids = self.features[index].code_type_ids
        code_visit_length = self.features[index].code_visit_length
        text_ids = self.features[index].text_input_ids
        text_type_ids = self.features[index].text_type_ids
        text_att_mask = self.features[index].text_att_mask
        label = self.features[index].y
        cur_tensors = (
            torch.tensor(code_input_ids, dtype=torch.long),
            torch.tensor(code_type_ids, dtype=torch.long),
            torch.tensor(code_visit_length, dtype=torch.long),
            torch.tensor(text_ids,dtype=torch.long),
            torch.tensor(text_type_ids, dtype=torch.long),
            torch.tensor(text_att_mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.float)
        )
        return cur_tensors

def collate_wraper(model_choice, predict_task):
    def pad_data(batch_data, max_len):
        paded_data = []
        for bi in batch_data:
            if bi.shape[0] < max_len:
                pad_bi = torch.zeros((max_len-bi.shape[0], bi.shape[1]), dtype=torch.long)
                paded_bi = torch.cat([bi, pad_bi], dim=0)
            else:
                paded_bi = bi
            paded_data.append(paded_bi)
        return torch.stack(paded_data, dim=0)

    def collate_fn(batch_data):
        batch_code_inputs = []
        batch_code_type_ids = []
        batch_code_visit_len = []
        batch_labels = []

        batch_txt_inputs = []
        batch_txt_type_ids = []
        batch_txt_att_mask = []
        for d in batch_data:
            code_input_ids, code_type_ids, code_visit_length, txt_inputs, txt_type_id, txt_att_mask, label = d
            if model_choice == 'gbert': 
                batch_code_inputs.append(code_input_ids)
                batch_code_visit_len.append(code_visit_length)
            else:
                batch_code_inputs.append(code_input_ids)
                batch_code_type_ids.append(code_type_ids)
                batch_code_visit_len.append(code_visit_length)
                batch_txt_inputs.append(txt_inputs)
                batch_txt_type_ids.append(txt_type_id)
                batch_txt_att_mask.append(txt_att_mask)
                if predict_task == 'rx' or predict_task == 'dx': 
                    batch_labels.append(label)
                else:
                    batch_labels.extend(label)
            
        max_len = np.max([len(i) for i in batch_code_inputs])
        batch_code_visit_len = torch.stack(batch_code_visit_len)
        if model_choice == 'gbert': 
            return batch_code_inputs, None, batch_code_visit_len, None, None, None, batch_labels
        else:        
            paded_code_input_ids = pad_data(batch_code_inputs, max_len)
            paded_code_type_ids = pad_data(batch_code_type_ids, max_len)
            paded_txt_input_ids = pad_data(batch_txt_inputs, max_len)
            paded_txt_type_ids = pad_data(batch_txt_type_ids, max_len)
            paded_txt_att_mask = pad_data(batch_txt_att_mask, max_len)
            if predict_task == 'rx' or predict_task == 'dx':
                batch_labels = pad_data(batch_labels, max_len-1)
            return paded_code_input_ids, paded_code_type_ids, batch_code_visit_len, paded_txt_input_ids, paded_txt_type_ids, paded_txt_att_mask, batch_labels
    return collate_fn


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='GBert-rx-predict', type=str, required=False,
                        help="model name")
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--pretrain_dir", default='../saved/Gbert_preTraining', type=str, required=False,
                        help="pretraining model")
    parser.add_argument("--train_file", default='data-multi-visit.pkl', type=str, required=False,
                        help="training data file.")
    parser.add_argument("--output_dir",
                        default='../saved_result/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--use_pretrain",
                        default=False,
                        action='store_true',
                        help="is use pretrain")
    parser.add_argument("--graph",
                        default=False,
                        action='store_true',
                        help="if use ontology embedding")
    parser.add_argument("--therhold",
                        default=0.3,
                        type=float,
                        help="therhold.")
    parser.add_argument("--max_seq_length",
                        default=61,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run on the dev set.")
    parser.add_argument("--do_test",
                        default=True,
                        action='store_true',
                        help="Whether to run on the test set.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help="random seed for initialization")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--use_DischargeSummary", action='store_true',
                        help='use text information of discharge summary')
    parser.add_argument("--clinical_bert_dir", 
                        default="$/ClinicalBert/model/pretraining/")
    parser.add_argument("--only_useDischargSummary", action='store_true',
                        help='use text information of discharge summary')
    parser.add_argument("--data_name", type=str, required=True, help='data name', choices=['multilabel'])
    parser.add_argument("--predict_task", type=str, required=True, choices=['rx'], help='choose in [rx, dx, readm, IHM(in hospital mortality)]')
    parser.add_argument("--model_choice", 
                        type=str, 
                        required=True, 
                        choices=['fusion_ml','fusion_bin'], 
                        help='use which model')
    parser.add_argument("--max_visit_len", type=int, default=29, help='how many visit number will be calculated')
    parser.add_argument('--partion', type=float, default=1.0)
    parser.add_argument('--saved_model_file', type=str, default=None, help='The saved fine-tuning stage.')

    args = parser.parse_args()
    args.output_dir_save = os.path.join(args.output_dir, args.predict_task, args.model_name, '{}_part'.format(args.partion))
    # args.output_dir = os.path.join(args.output_dir, args.predict_task, args.model_name, '{}_part'.format(args.partion), '{}_{}'.format(args.seed, args.learning_rate))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_save, exist_ok=True)

    # load tokenizer
    code_tokenizer = EHRTokenizer(args.data_dir)

    # text tokenizer
    txt_tokenizer = BertTokenizer.from_pretrained(args.clinical_bert_dir)

    processors = {
        'multilabel': MultilabelProcessor
    }
    model_box = {
        'fusion_ml':FusionBert_ML_predict, 
        'fusion_bin':FusionBert_binary_predict,
    }

    num_labels_task = {
        'rx': len(code_tokenizer.rx_voc.word2idx),
        'dx': len(code_tokenizer.dx_voc.word2idx),
        'readm': 2,
        'IHM': 2
    }

    logger.info("Loading Dataset")
    logger.info('TEST')

    with open('../dataset/rx_example.pk', 'rb') as f:
        test_dataset = pickle.load(f)
    num_labels = num_labels_task[args.predict_task]

    logger.info('Loading Model: ' + args.model_name)
    if args.use_pretrain:
        logger.info("Use Pretraining model")
        model = model_box[args.model_choice].from_pretrained(args.pretrain_dir, code_tokenizer=code_tokenizer, \
            num_labels=num_labels)       
    else:
        config = BertConfig(
            vocab_size_or_config_json_file=len(code_tokenizer.vocab.word2idx))
        config.graph = args.graph
        model = model_box[args.model_choice](config, code_tokenizer, num_labels=num_labels)
    logger.info('# of model parameters: ' + str(get_n_params(model)))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)
    

    # model_to_save = model.module if hasattr(
    #     model, 'module') else model  # Only save the model it-self
    model_to_save = model
    output_model_file = os.path.join(
        args.saved_model_file, "pytorch_model.bin")

    # Prepare optimizer
    # num_train_optimization_steps = int(
    #     len(train_dataset) / args.train_batch_size) * args.num_train_epochs
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=num_train_optimization_steps)
    if args.use_pretrain:
        if torch.cuda.device_count() > 1:
            for param in model.module.bert.embeddings.parameters():
                param.requires_grad = False
            for param in model.module.bert.encoder.layer[:10].parameters():
                param.requires_grad = False
        else:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
            for param in model.bert.encoder.layer[:10].parameters():
                param.requires_grad = False
        
    optimizer = Adam(
                    filter(lambda x : x.requires_grad, model.parameters()), 
                    lr=args.learning_rate)
    # optimizer = Adam(model.parameters(), lr=args.learning_rate)

    m = nn.Sigmoid()

    if args.do_test:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_dataset))
        logger.info("  Batch size = %d", 1)

        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(output_model_file, map_location=device)
        if list(model_state_dict.keys())[0].split('.')[0] == 'module':
            new_model_state_dict = {}
            for k, v in model_state_dict.items():
                new_model_state_dict[k[7:]] = v
            model.load_state_dict(new_model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
        model.to(device)

        model.eval()
        y_preds = []
        y_trues = []
        test_accuracy = 0
        nb_test_examples = 0
        test_features = convert_examples_to_features(test_dataset, txt_tokenizer, code_tokenizer, args.max_seq_length, \
                                                        args.max_visit_len, predict_task=args.predict_task, gbert_type_data=True, partion=1.0)
        test_data = EHR_Text_DataSet(test_features)
        test_dataloader = DataLoader(test_data, 
                                        sampler=SequentialSampler(test_data),
                                        batch_size=args.eval_batch_size,
                                        collate_fn=collate_wraper(args.model_choice, args.predict_task)
                                        )
        total_samples = {}
        excell_samples = {}
        for idx, test_input in tqdm(enumerate(test_dataloader), desc="Testing"):
            sample = {}
            test_input = tuple(t.cuda(device) for t in test_input)
            code_input_ids, code_type_ids, code_visit_length, txt_input_ids, txt_token_ids, txt_attention_mask, label = test_input

            if num_labels <= 2:
                code_visit_length = None

            loss, logits, truth = model(code_input_ids, code_type_ids, code_visit_length, txt_input_ids, txt_token_ids, txt_attention_mask, label)
            
            y_preds.append(t2n(m(logits)))
            y_trues.append(t2n(truth))
            
            label_ids = t2n(truth).flatten()
            scores = torch.squeeze(m(logits)).detach().cpu().numpy()
            outputs = np.asarray([1 if i else 0 for i in (scores.flatten()>=0.5)])
            tmp_test_accuracy = np.sum(outputs == label_ids)
            test_accuracy += tmp_test_accuracy
            nb_test_examples += scores.flatten().shape[0]
            last_output = np.array([1 if i else 0 for i in (t2n(m(logits))[-1,:]>=0.5)])
            last_label = t2n(truth.view_as(logits)[-1,:])
            output_idx = list(np.where(last_output==1)[0])
            label_idx = list(np.where(last_label==1)[0])
            pred_drug = [code_tokenizer.rx_voc.idx2word[i] for i in output_idx]
            label_drug = [code_tokenizer.rx_voc.idx2word[i] for i in label_idx]
            sample['predication'] = pred_drug
            sample['targets'] = label_drug
            sample['correct/total'] = len(set(label_drug).intersection(set(pred_drug))) / len(label_drug)
            sample['correct/predict'] = len(set(label_drug).intersection(set(pred_drug))) / len(pred_drug)
            total_samples['id_{}'.format(idx)] = sample


        print('')
        if num_labels > 2:
            acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                            args.therhold)
        else:
            acc_container = binary_metric_reporter(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                                args.therhold)

        # save report

        test_accuracy = test_accuracy / nb_test_examples
        logger.info("test/accuracy : {:.4f}".format(test_accuracy))
        for k, v in acc_container.items():
            logger.info("test:{}/{}".format(k,v))
        with open(os.path.join(args.output_dir_save, 'test_saved_sample.txt'), 'w', encoding='utf-8') as f:
            json.dump(total_samples, f, indent=4, separators=(',',':'))

               


if __name__ == "__main__":
    main()

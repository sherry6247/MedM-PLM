'''
Author: your name
Date: 2021-10-23 17:12:11
LastEditTime: 2021-11-03 14:54:37
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /model/lsc_code/preTrain_in_MIMIC-III/multi_modal_pretrain/src/multiModalDataset.py
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from requests.sessions import merge_cookies
from tqdm import tqdm, trange
import dill
import copy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam

from transformers import BertTokenizer, BertModel
import collections

import torch.nn as nn
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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

class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, rx_input_ids, dx_input_ids, text_ids, text_type_ids, text_att_mask, y_dx, y_rx):
        self.dx_input_ids = dx_input_ids
        self.rx_input_ids = rx_input_ids
        self.text_input_ids = text_ids
        self.text_type_ids = text_type_ids
        self.text_att_mask = text_att_mask
        self.y_dx = y_dx
        self.y_rx = y_rx

class InputExample(object):
    """A single training/test example for pre-training."""

    def __init__(self, guid, dx_code=None, rx_code=None, text=None):
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

class DataProcessor(object):
    """Base class for data converters for pre-trainings data sets."""

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
            input_file, 'data-multi-visit.pkl'))#[:500]
        data_single = pd.read_pickle(
            os.path.join(input_file, 'data-single-visit.pkl'))#[:50]
        return data_multi, data_single

class CodeTextProcessor(DataProcessor):
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
            #  only use the last one
            discharge_txt = dis_txt[-1]#' '.join(dis_txt)
        return discharge_txt    
    
    def preprocess1(self, x):
        y=re.sub('\\[(.*?)\\]','',x) #remove de-identified brackets
        y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
        y=re.sub('dr\.','doctor',y)
        y=re.sub('m\.d\.','md',y)
        y=re.sub('admission date:','',y)
        y=re.sub('discharge date:','',y)
        y=re.sub('--|__|==','',y)
        return y

    def preprocessing(self, df_less_n): 
        df_less_n['discharge_summary']=df_less_n['discharge_summary'].fillna(' ')
        df_less_n['discharge_summary']=df_less_n['discharge_summary'].str.replace('\n',' ')
        df_less_n['discharge_summary']=df_less_n['discharge_summary'].str.replace('\r',' ')
        df_less_n['discharge_summary']=df_less_n['discharge_summary'].apply(str.strip)
        df_less_n['discharge_summary']=df_less_n['discharge_summary'].str.lower()

        df_less_n['discharge_summary']=df_less_n['discharge_summary'].apply(lambda x: self.preprocess1(x))
        return df_less_n

    def split_dataset(self, data_dir):
        data_dict = {}
        data_multi, data_single = self._read_pickle(data_dir)
        data_multi['Set_NDC'] = data_multi.NDC.map(lambda x: self.get_set_data(x))
        data_multi['discharge_summary'] = data_multi.CATEGORY_TEXT.map(lambda x: self.get_discharge_summary_txt(x))

        data_single['Set_NDC'] = data_single.NDC.map(lambda x: self.get_set_data(x))
        data_single['discharge_summary'] = data_single.CATEGORY_TEXT.map(lambda x: self.get_discharge_summary_txt(x))

        data_single = self.preprocessing(data_single)
        data_multi = self.preprocessing(data_multi)

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

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} and {}".format(os.path.join(data_dir, "data-multi-visit.pkl"), os.path.join(data_dir, "data-single-visit.pkl")))
        
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
        for i, row in tqdm(pd_data.iterrows(), desc=set_type, ncols=100):
            guid = "%s-%s" % (set_type, row['SUBJECT_ID'])
            dx_code = list(row['DIAG_CODE'])
            rx_code = row['Set_NDC']
            text = row['discharge_summary']
            examples.append(
                InputExample(guid=guid, dx_code=dx_code, rx_code=rx_code, text=text))
        return examples

def convert_examples_to_features(examples, seq_len, max_predictions_per_seq, txt_tokenizer, code_tokenizer, rng):

    features = []
    max_len = []
    for (ex_index, example) in tqdm(enumerate(examples), desc='convert_examples_to_features', ncols=100):
        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l

        y_dx = np.zeros(len(code_tokenizer.dx_voc.word2idx))
        y_rx = np.zeros(len(code_tokenizer.rx_voc.word2idx))
        for item in example.dx_code:
            y_dx[code_tokenizer.dx_voc.word2idx[item]] = 1
        for item in example.rx_code:
            y_rx[code_tokenizer.rx_voc.word2idx[item]] = 1

        """replace tokens with [MASK]
        """
        dx_input, dx_masked_pos, dx_masked_labels = random_word_ids(example.dx_code, code_tokenizer.dx_voc, max_predictions_per_seq, rng)
        rx_input, rx_masked_pos, rx_masked_labels = random_word_ids(example.rx_code, code_tokenizer.rx_voc, max_predictions_per_seq, rng)


        """extract input and output tokens
        """
        dx_input_tokens = []
        rx_input_tokens = []
        dx_input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(dx_input), seq_len - 1))
        rx_input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(rx_input), seq_len - 1))

        """convert tokens to id
        """
        dx_input_ids = code_tokenizer.convert_tokens_to_ids(dx_input_tokens)
        rx_input_ids = code_tokenizer.convert_tokens_to_ids(rx_input_tokens)
        tokens_text = txt_tokenizer(example.text, padding="max_length", max_length=512, truncation=True)
        txt_input_ids = tokens_text['input_ids']
        txt_token_type_ids = tokens_text['token_type_ids']
        txt_attention_mask = tokens_text['attention_mask']
        
        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("dx_tokens: %s" % " ".join(
                    [str(x) for x in dx_input_ids]))
            logger.info("rx_tokens: %s" % " ".join(
                    [str(x) for x in rx_input_ids]))

        features.append(
            InputFeature(dx_input_ids=dx_input_ids,
                        rx_input_ids=rx_input_ids,
                        text_ids=txt_input_ids,
                        text_type_ids=txt_token_type_ids,
                        text_att_mask=txt_attention_mask,
                        y_dx=y_dx, y_rx=y_rx))
    return features


def random_word(tokens, vocab):
    for i, _ in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"
            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(vocab.word2idx.items()))[0]
            else:
                pass
        else:
            pass

    return tokens
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                            ["index", "label"])
def random_word_ids(tokens, vocab, max_predictions_per_seq, rng):
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens)*0.15))))
    masked_lm_positions = [] 
    masked_lm_labels = []
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab.idx2word[rng.randint(0, len(vocab.word2idx.items()) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, np.array(masked_lm_positions), masked_lm_labels

class MultiModalDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        dx_input_ids = self.features[index].dx_input_ids
        rx_input_ids = self.features[index].rx_input_ids
        text_input_ids = self.features[index].text_input_ids
        text_type_ids = self.features[index].text_type_ids
        text_att_mask = self.features[index].text_att_mask
        y_dx = self.features[index].y_dx
        y_rx = self.features[index].y_rx
        cur_tensors = (
            torch.tensor(dx_input_ids, dtype=torch.long),
            torch.tensor(rx_input_ids, dtype=torch.long),
            torch.tensor(text_input_ids, dtype=torch.long),
            torch.tensor(text_type_ids,dtype=torch.long),
            torch.tensor(text_att_mask, dtype=torch.long),
            torch.tensor(y_dx, dtype=torch.float),
            torch.tensor(y_rx, dtype=torch.float)
        )
        return cur_tensors
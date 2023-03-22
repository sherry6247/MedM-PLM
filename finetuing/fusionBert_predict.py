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

from utils import metric_report, t2n, get_n_params, vote_score, pr_curve_plot, vote_pr_curve
from fusion_config import FuseBertConfig as BertConfig
from fusion_predictive_models import FusionBert_ML_predict, FusionBert_binary_predict

from transformers import BertTokenizer, BertModel
import torch.nn as nn
import re


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
    """A single training/test example for simple sequence representation."""

    def __init__(self, guid, dx_code=None, rx_code=None, text=None, label_id=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            dx_code: for the diagnosis code of the example
            rx_code: for the ATC code of the example
            text: for the discharge summary text of the example
            label_id: for the binary classification label index of the example
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
    def _read_pickle_multi(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # load data
        data_multi = pd.read_pickle(os.path.join(
            input_file, 'data-multi-visit.pkl'))#
        data_single = pd.read_pickle(
            os.path.join(input_file, 'data-single-visit.pkl'))#

        return data_multi, data_single

    @classmethod
    def _read_pickle_binary(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # load data
        data_multi = pd.read_pickle(os.path.join(
            input_file, 'data-multi-visit.pkl'))#[:500]
        data_single = pd.read_pickle(
            os.path.join(input_file, 'data-single-visit.pkl'))#[:20]
        total_data = data_multi.append(data_single)
        readmit_ID = total_data[total_data.READMISSION_LABEL == 1].HADM_ID
        not_readmit_ID = total_data[total_data.READMISSION_LABEL == 0].HADM_ID
        #subsampling to get the balanced pos/neg numbers of patients for each dataset
        not_readmit_ID_use = not_readmit_ID.sample(n=len(readmit_ID), random_state=1)
        id_val_test_t=readmit_ID.sample(frac=0.2,random_state=1)
        id_val_test_f=not_readmit_ID_use.sample(frac=0.2,random_state=1)

        id_train_t = readmit_ID.drop(id_val_test_t.index)
        id_train_f = not_readmit_ID_use.drop(id_val_test_f.index)

        id_val_t=id_val_test_t.sample(frac=0.5,random_state=1)
        id_test_t=id_val_test_t.drop(id_val_t.index)

        id_val_f=id_val_test_f.sample(frac=0.5,random_state=1)
        id_test_f=id_val_test_f.drop(id_val_f.index)

        # test if there is overlap between train and test, should return "array([], dtype=int64)"
        (pd.Index(id_test_t).intersection(pd.Index(id_train_t))).values

        id_test = pd.concat([id_test_t, id_test_f])
        test_id_label = pd.DataFrame(data = list(zip(id_test, [1]*len(id_test_t)+[0]*len(id_test_f))), columns = ['id','label'])

        id_val = pd.concat([id_val_t, id_val_f])
        val_id_label = pd.DataFrame(data = list(zip(id_val, [1]*len(id_val_t)+[0]*len(id_val_f))), columns = ['id','label'])

        id_train = pd.concat([id_train_t, id_train_f])
        train_id_label = pd.DataFrame(data = list(zip(id_train, [1]*len(id_train_t)+[0]*len(id_train_f))), columns = ['id','label'])

        train_data = total_data[total_data.HADM_ID.isin(train_id_label.id)]
        val_data = total_data[total_data.HADM_ID.isin(val_id_label.id)]
        test_data = total_data[total_data.HADM_ID.isin(test_id_label.id)]
        # subsampling for training....since we obtain training on patient admission level so now we have same number of pos/neg readmission
        # but each admission is associated with different length of notes and we train on each chunks of notes, not on the admission, we need
        # to balance the pos/neg chunks on training set. (val and test set are fine) Usually, positive admissions have longer notes, so we need 
        # find some negative chunks of notes from not_readmit_ID that we haven't used yet

        df = pd.concat([not_readmit_ID_use, not_readmit_ID])
        df = df.drop_duplicates(keep=False)
        #check to see if there are overlaps
        (pd.Index(df).intersection(pd.Index(not_readmit_ID_use))).values

        # for this set of split with random_state=1, we find we need 400 more negative training samples
        not_readmit_ID_more = df.sample(n=400, random_state=1)
        train_snippets = pd.concat([total_data[total_data.HADM_ID.isin(not_readmit_ID_more)], train_data])

        #shuffle
        train_snippets = train_snippets.sample(frac=1, random_state=1).reset_index(drop=True)

        #check if balanced
        train_snippets.READMISSION_LABEL.value_counts()

        return train_snippets, val_data, test_data

class binarylabelProcessor(DataProcessor):
    """Processor for the code and text data of binary classification ."""
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

    def column_process(self, data_df):
        data_df['Set_NDC'] = data_df.NDC.map(lambda x: self.get_set_data(x))
        data_df['discharge_summary'] = data_df.CATEGORY_TEXT.map(lambda x: self.get_discharge_summary_txt(x))
        data_df = self.preprocessing(data_df)
        return data_df

    def split_dataset(self, data_dir, partion):
        data_dict = {}
        train_data, val_data, test_data = self._read_pickle_binary(data_dir)
        train_data = self.column_process(train_data)
        val_data = self.column_process(val_data)
        test_data = self.column_process(test_data)
        
        data_dict['train_data'] = train_data[:int(len(train_data)*partion)] # [:int(partion)]#
        data_dict['dev_data'] = val_data
        data_dict['test_data'] = test_data

        return data_dict

    def get_train_examples(self, data_dir, label_type, partion):
        """See base class."""
        logger.info("LOOKING AT {} and {}".format(os.path.join(data_dir, "data-multi-visit.pkl"), os.path.join(data_dir, "data-single-visit.pkl")))
        
        return self._create_examples(
            self.split_dataset(data_dir, partion)['train_data'], "train", label_type)

    def get_dev_examples(self, data_dir, label_type, partion):
        """See base class."""
        return self._create_examples(
            self.split_dataset(data_dir, partion)['dev_data'], "dev", label_type)

    def get_test_examples(self, data_dir, label_type, partion):
        """See base class."""
        return self._create_examples(
            self.split_dataset(data_dir, partion)['test_data'], "test", label_type)
    
    def get_df(self, data_dir, data_type, partion):
        return self.split_dataset(data_dir, partion)[data_type]

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
        data_multi, data_single = self._read_pickle_multi(data_dir)
        data_multi['Set_NDC'] = data_multi.NDC.map(lambda x: self.get_set_data(x))
        data_multi['discharge_summary'] = data_multi.CATEGORY_TEXT.map(lambda x: self.get_discharge_summary_txt(x))
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
        for subject_id in tqdm(pd_data['SUBJECT_ID'].unique(), desc=set_type, ncols=100):
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

def convert_examples_to_features_binary(examples, txt_tokenizer, code_tokenizer, seq_len, max_visit_len=None, partion=1.0):

    features = []
    max_len = []
    for (ex_index, example) in tqdm(enumerate(examples), desc='convert_examples_to_features', ncols=100):
        def fill_to_max(l, seq):
            t = l.copy()
            while len(t) < seq:
                t.append('[PAD]')
            return t
        def fill_to_zero(l, seq):
            t = l.copy()
            while len(t) < seq:
                t.append(0)
            return t
        
        """extract input and output tokens
        """

        dx_record = example.dx_code
        rx_record = example.rx_code
        
        code_tokens = ['[CLS]'] + fill_to_max(dx_record, seq_len - 1)
        code_tokens.extend(['[CLS]'] + fill_to_max(rx_record, seq_len - 1))
        code_type_ids = [0] * len(['[CLS]'] + fill_to_max(dx_record, seq_len - 1)) + [1] * len(['[CLS]'] + fill_to_max(rx_record, seq_len - 1))
        
        text_reocrd = example.text
                
        """convert tokens to id
        """
        code_input_ids = code_tokenizer.convert_tokens_to_ids(code_tokens)
        tokens_text = txt_tokenizer(text_reocrd, padding="max_length", max_length=512, truncation=True)
        txt_input_ids = tokens_text['input_ids']
        txt_token_type_ids = tokens_text['token_type_ids']
        txt_attention_mask = tokens_text['attention_mask']

        y = example.label_id        
            
        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("code_tokens: %s" % " ".join(
                    [str(x) for x in code_input_ids]))
        
        features.append(
            InputFeature(code_input_ids=code_input_ids,
                        code_type_ids=code_type_ids,
                        text_ids=txt_input_ids,
                        text_type_ids=txt_token_type_ids,
                        text_att_mask=txt_attention_mask,
                        y=y))
    return features

def convert_examples_to_features_multi(examples, txt_tokenizer, code_tokenizer, seq_len, max_visit_len=None, partion=1.0):

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
            
            code_tokens = ['[CLS]'] + fill_to_max(dx_record, seq_len - 1)
            code_tokens.extend(['[CLS]'] + fill_to_max(rx_record, seq_len - 1))
            code_type_ids = [0] * len(['[CLS]'] + fill_to_max(dx_record, seq_len - 1)) + [1] * len(['[CLS]'] + fill_to_max(rx_record, seq_len - 1))
            
            text_reocrd = example.text[idx]
            input_tokens_list.append(code_tokens)
            txt_input_tokens.append(text_reocrd)
            code_type_id_list.append(code_type_ids)

            if example.label_id is None:
                if idx != 0: 
                    y_list.append(example.rx_code[idx])
                
        """convert tokens to id
        """
        code_input_ids = [code_tokenizer.convert_tokens_to_ids(input_tokens) for input_tokens in input_tokens_list]
        tokens_text = txt_tokenizer(txt_input_tokens, padding="max_length", max_length=512, truncation=True)
        txt_input_ids = tokens_text['input_ids']
        txt_token_type_ids = tokens_text['token_type_ids']
        txt_attention_mask = tokens_text['attention_mask']

        y = []
        voc_size = len(code_tokenizer.rx_voc.word2idx)
        for tokens in y_list:
            tmp_labels = np.zeros(voc_size)
            tmp_labels[list(
                map(lambda x: code_tokenizer.rx_voc.word2idx[x], tokens))] = 1
            y.append(tmp_labels)               
                      
            
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

def collate_wraper_multi():
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

            batch_code_inputs.append(code_input_ids)
            batch_code_type_ids.append(code_type_ids)
            batch_code_visit_len.append(code_visit_length)
            batch_txt_inputs.append(txt_inputs)
            batch_txt_type_ids.append(txt_type_id)
            batch_txt_att_mask.append(txt_att_mask)
            batch_labels.append(label)
            
        max_len = np.max([len(i) for i in batch_code_inputs])
        batch_code_visit_len = torch.stack(batch_code_visit_len)
              
        paded_code_input_ids = pad_data(batch_code_inputs, max_len)
        paded_code_type_ids = pad_data(batch_code_type_ids, max_len)
        paded_txt_input_ids = pad_data(batch_txt_inputs, max_len)
        paded_txt_type_ids = pad_data(batch_txt_type_ids, max_len)
        paded_txt_att_mask = pad_data(batch_txt_att_mask, max_len)
        batch_labels = pad_data(batch_labels, max_len-1)
        return paded_code_input_ids, paded_code_type_ids, batch_code_visit_len, paded_txt_input_ids, paded_txt_type_ids, paded_txt_att_mask, batch_labels
    return collate_fn


def collate_wraper_binary():

    def collate_fn(batch_data):
        batch_code_inputs = []
        batch_code_type_ids = []
        batch_labels = []

        batch_txt_inputs = []
        batch_txt_type_ids = []
        batch_txt_att_mask = []
        for d in batch_data:
            code_input_ids, code_type_ids, txt_inputs, txt_type_id, txt_att_mask, label = d
            batch_code_inputs.append(code_input_ids)
            
            batch_code_type_ids.append(code_type_ids)
            batch_txt_inputs.append(txt_inputs)
            batch_txt_type_ids.append(txt_type_id)
            batch_txt_att_mask.append(txt_att_mask)
            batch_labels.append(label)
                          
        paded_code_input_ids = torch.stack(batch_code_inputs, dim=0)
        paded_code_type_ids = torch.stack(batch_code_type_ids, dim=0)
        paded_txt_input_ids = torch.stack(batch_txt_inputs, dim=0)
        paded_txt_type_ids = torch.stack(batch_txt_type_ids, dim=0)
        paded_txt_att_mask = torch.stack(batch_txt_att_mask, dim=0)
        batch_labels = torch.stack(batch_labels, dim=0)
        return paded_code_input_ids, paded_code_type_ids, paded_txt_input_ids, paded_txt_type_ids, paded_txt_att_mask, batch_labels
    return collate_fn


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='FusionBert-predict', type=str, required=False,
                        help="model name")
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--pretrain_dir", default='../saved/preTrained_model', type=str, required=False,
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
                        default=10,
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
                        default=2022,
                        help="random seed for initialization")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--use_DischargeSummary", action='store_true',
                        help='use text information of discharge summary')
    parser.add_argument("--clinical_bert_dir", 
                        default="../saved/ClinicalBert/model/pretraining/")
    parser.add_argument("--only_useDischargSummary", action='store_true',
                        help='use text information of discharge summary')
    parser.add_argument("--data_name", type=str, required=True, help='data name', choices=['multilabel','binarylabel'])
    parser.add_argument("--predict_task", type=str, required=True, choices=['rx', 'readm'], help='choose in [rx, dx, readm, IHM(in hospital mortality)]')
    parser.add_argument("--model_choice", 
                        type=str, 
                        required=True, 
                        choices=['fusion_ml','fusion_bin'], 
                        help='use which model')
    parser.add_argument("--max_visit_len", type=int, default=10, help='how many visit number will be calculated')
    parser.add_argument('--partion', type=float, default=1.0)

    args = parser.parse_args()
    args.output_dir_save = os.path.join(args.output_dir, args.predict_task, args.model_name, '{}_part'.format(args.partion))
    args.output_dir = os.path.join(args.output_dir, args.predict_task, args.model_name, '{}_part'.format(args.partion), '{}_{}'.format(args.seed, args.learning_rate))

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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_save, exist_ok=True)

    # Code tokenizer
    code_tokenizer = EHRTokenizer(args.data_dir)

    # Text tokenizer
    txt_tokenizer = BertTokenizer.from_pretrained(args.clinical_bert_dir)

    processors = {
        'multilabel': MultilabelProcessor,
        'binarylabel': binarylabelProcessor
    }
    model_box = {
        'fusion_ml':FusionBert_ML_predict, 
        'fusion_bin':FusionBert_binary_predict
    }

    collate_wraper_box = {
        "multilabel":collate_wraper_multi,
        "binarylabel":collate_wraper_binary
    }

    convert_examples_to_features_box = {
        "multilabel": convert_examples_to_features_multi,
        "binarylabel": convert_examples_to_features_binary
    }

    num_labels_task = {
        'rx': len(code_tokenizer.rx_voc.word2idx),
        'dx': len(code_tokenizer.dx_voc.word2idx),
        'readm': 1,
        'IHM': 1
    }

    logger.info("Loading {} Dataset".format(args.data_name))

    processor = processors[args.data_name]()
    
    collate_wraper = collate_wraper_box[args.data_name]
    convert_examples_to_features = convert_examples_to_features_box[args.data_name]
    if args.predict_task == 'rx':
        logger.info("TRAIN")
        train_dataset = processor.get_train_examples(args.data_dir)
        logger.info('DEV')
        eval_dataset = processor.get_dev_examples(args.data_dir)
        logger.info('TEST')
        test_dataset = processor.get_test_examples(args.data_dir)
    else:
        if args.predict_task == 'readm':
            label_type = 'READMISSION_LABEL'
        elif args.predict_task == 'IHM':
            label_type = 'HOSPITAL_EXPIRE_FLAG'
        logger.info("TRAIN")
        train_dataset = processor.get_train_examples(args.data_dir, label_type=label_type, partion=args.partion)
        logger.info('DEV')
        eval_dataset = processor.get_dev_examples(args.data_dir, label_type=label_type, partion=args.partion)
        eval_df = processor.get_df(args.data_dir, data_type='dev_data', partion=args.partion)
        logger.info('TEST')
        test_dataset = processor.get_test_examples(args.data_dir, label_type=label_type, partion=args.partion)
        test_df = processor.get_df(args.data_dir, data_type='test_data', partion=args.partion)


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
    

    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    # model_to_save = model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin")

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

    """ you can comment out this condition """
    if args.use_pretrain:
        if torch.cuda.device_count() > 1:
            for param in model.module.bert.embeddings.parameters():
                param.requires_grad = False
            for param in model.module.bert.encoder.layer[:10].parameters():
                param.requires_grad = False
            # for param in model.module.bert.pooler.parameters():
            #     param.requires_grad = False
        else:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
            for param in model.bert.encoder.layer[:10].parameters():
                param.requires_grad = False
        # for param in model.module.bert.pooler.parameters():
        #     param.requires_grad = False
        
    optimizer = Adam(
                    filter(lambda x : x.requires_grad, model.parameters()), 
                    lr=args.learning_rate)

    global_step = 0

    m = nn.Sigmoid()
    if args.do_train:
        train_features = convert_examples_to_features(train_dataset, txt_tokenizer, code_tokenizer, args.max_seq_length, \
                                                        max_visit_len=args.max_visit_len)
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info(" Num steps = %d", int(len(train_dataset)/args.train_batch_size))

        train_data = EHR_Text_DataSet(train_features)
        train_dataloader = DataLoader(train_data, 
                                        sampler=RandomSampler(train_data),
                                        batch_size=args.train_batch_size,
                                        collate_fn=collate_wraper)
        eval_features = convert_examples_to_features(eval_dataset, txt_tokenizer, code_tokenizer, args.max_seq_length, \
                                                        max_visit_len=args.max_visit_len)
        writer = SummaryWriter(args.output_dir)


        dx_acc_best, rx_acc_best = 0, 0
        early_stop_epoch = 0
        acc_name = 'prauc'
        dx_history = {'prauc': []}
        rx_history = {'prauc': []}

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            print('')
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            prog_iter = tqdm(train_dataloader, leave=False, desc='Training')
            model.train()
            for _, batch in enumerate(prog_iter):
                optimizer.zero_grad()
                batch = tuple(t.cuda(device) for t in batch)
                code_input_ids, code_type_ids, code_visit_length, txt_input_ids, txt_token_ids, txt_attention_mask, label = batch

                if num_labels <= 2:
                    code_visit_length = None

                loss, logits, truth = model(code_input_ids, code_type_ids=code_type_ids, input_lengths=code_visit_length, \
                                            txt_input_ids=txt_input_ids, txt_type_ids=txt_token_ids, txt_attention_mask=txt_attention_mask, labels=label)
                    
                if torch.cuda.device_count() > 1:
                    loss = torch.sum(loss)
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += 1
                nb_tr_steps += 1

                # Display loss
                prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))
                with open(os.path.join(args.output_dir, 'train_result.txt'), 'a', encoding='utf-8') as f:
                    f.write("Epoch:{}\tloss:{}".format(epoch, tr_loss / nb_tr_steps))

                optimizer.step()

                torch.cuda.empty_cache()

            writer.add_scalar('train/loss', tr_loss / nb_tr_steps, global_step)
            global_step += 1

            if args.do_eval:
                print('')
                logger.info("***** Running eval *****")
                logger.info("  Num examples = %d", len(eval_dataset))
                logger.info("  Batch size = %d", args.eval_batch_size)
                logger.info(" Num steps = %d", int(len(eval_dataset)/args.eval_batch_size))
                
                eval_data = EHR_Text_DataSet(eval_features)
                eval_dataloader = DataLoader(eval_data, 
                                            sampler=SequentialSampler(eval_data),
                                            batch_size=args.eval_batch_size,
                                            collate_fn=collate_wraper)
                model.eval()
                y_preds = []
                y_trues = []
                eval_loss = 0
                eval_step = 0
                
                eval_accuracy = 0
                nb_eval_examples = 0
                for eval_input in tqdm(eval_dataloader, desc="Evaluating"):
                    eval_input = tuple(t.cuda(device) for t in eval_input)
                    code_input_ids, code_type_ids, code_visit_length, txt_input_ids, txt_token_ids, txt_attention_mask, labels = eval_input

                    if num_labels <= 2:
                        code_visit_length = None

                    loss, logits, truth = model(code_input_ids, code_type_ids=code_type_ids, input_lengths=code_visit_length, \
                                            txt_input_ids=txt_input_ids, txt_type_ids=txt_token_ids, txt_attention_mask=txt_attention_mask, labels=labels)
                    if torch.cuda.device_count() > 1:
                        loss = torch.mean(loss)
                        
                    eval_step += 1
                    eval_loss += loss.item()

                    if args.predict_task == 'rx':
                        y_preds.append(t2n(torch.sigmoid(logits)))
                        y_trues.append(t2n(truth))
                        
                        scores = torch.squeeze(m(logits)).detach().cpu().numpy()
                        label_ids = t2n(truth).flatten()
                        outputs = np.asarray([1 if i else 0 for i in (scores.flatten()>=0.5)])
                        tmp_eval_accuracy = np.sum(outputs == label_ids)
                        eval_accuracy += tmp_eval_accuracy
                        nb_eval_examples += scores.flatten().shape[0]
                    else:
                        logits = torch.squeeze(m(logits)).detach().cpu().numpy()
                        label_ids = labels.to('cpu').numpy()

                        outputs = np.asarray([1 if i else 0 for i in (logits.flatten()>=0.5)])
                        tmp_eval_accuracy=np.sum(outputs == label_ids)
                        
                        true_labels = true_labels + label_ids.flatten().tolist()
                        pred_labels = pred_labels + outputs.flatten().tolist()
                        logits_history = logits_history + logits.flatten().tolist()

                        eval_accuracy += tmp_eval_accuracy

                        nb_eval_examples += labels.size(0)

                print('')
                # dx_acc_container = metric_report(np.concatenate(dx_y_preds, axis=0), np.concatenate(dx_y_trues, axis=0),
                #                                  args.therhold)
                if args.predict_task == 'rx':
                    eval_acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                                    args.therhold)
                
                
                    logger.info("eval/accuracy : {:.4f}".format(eval_accuracy/nb_eval_examples))

                    for k, v in eval_acc_container.items():
                        writer.add_scalar(
                            'eval/{}'.format(k), v, global_step)
                    writer.add_scalar('eval/loss', eval_loss / eval_step, global_step)
                else:
                    df = pd.DataFrame({'logits':logits_history, 'pred_label': pred_labels, 'label':true_labels})

                    string = 'logits_'+args.model_choice+'_chunk.csv'
                    df.to_csv(os.path.join(args.output_dir, string))
                    
                    writer.add_scalar('eval/loss', eval_loss / nb_eval_examples, global_step)

                    fpr, tpr, eval_auc = vote_score(df, args, flag='eval')
            
                    rp80, eval_ap, eval_aupr, rp70, rp60, rp50, f1 = vote_pr_curve(df, args, flag='eval')
                    result = {'eval_loss': eval_loss,
                            'eval_accuracy': eval_accuracy,                 
                            'global_step': global_step,
                            'auc' : eval_auc,
                            'aupr' : eval_aupr,
                            'ap': eval_ap,
                            'RP80': rp80,
                            'RP70': rp70,
                            'RP60':rp60,
                            'RP50': rp50,
                            'F1': f1}
                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "w") as f:
                        logger.info("***** {}_{}_{} Eval results *****".format(args.seed, args.learning_rate, args.model_name))
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            f.write("%s = %s\n" % (key, str(result[key])))

                

if __name__ == "__main__":
    main()

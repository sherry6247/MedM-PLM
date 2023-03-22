'''
Author: your name
Date: 2021-10-13 11:28:45
LastEditTime: 2022-01-18 17:47:26
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /model/lsc_code/preTrain_in_MIMIC-III/Fusion_bert/fusion_predictive_models.py
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss
from torch.nn import LayerNorm
import torch.nn.functional as F
from fusion_config import FuseBertConfig as BertConfig
from fusion_bert_model import CodeBERT, PreTrainedBertModel, Attention
import dill

from torch.autograd import Variable

from fusion_bert_model import BertModel, BertForSequenceClassification


logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def freeze_afterwards(model):
    for p in model.parameters():
        p.requires_grad = False


class MappingHead(nn.Module):
    def __init__(self, config: BertConfig):
        super(MappingHead, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.code_hidden_size, config.code_hidden_size),
                                   nn.ReLU())

    def forward(self, input):
        return self.dense(input)
        
class ClsHead(nn.Module):
    def __init__(self, config: BertConfig, voc_size):
        super(ClsHead, self).__init__()
        self.cls = nn.Sequential(nn.Linear(config.code_hidden_size, config.code_hidden_size), nn.ReLU(
        ), nn.Linear(config.code_hidden_size, voc_size))

    def forward(self, input):
        return self.cls(input)


class SelfSupervisedHead(nn.Module):
    def __init__(self, config: BertConfig, dx_voc_size, rx_voc_size):
        super(SelfSupervisedHead, self).__init__()
        self.multi_cls = nn.ModuleList([ClsHead(config, dx_voc_size), ClsHead(
            config, dx_voc_size), ClsHead(config, rx_voc_size), ClsHead(config, rx_voc_size)])

    def forward(self, dx_inputs, rx_inputs):
        # inputs (B, hidden)
        # output logits
        return self.multi_cls[0](dx_inputs), self.multi_cls[1](rx_inputs), self.multi_cls[2](dx_inputs), self.multi_cls[3](rx_inputs)

class text2code_attention(nn.Module):
    """
    计算text和code的attention，用scaled dot product attention
    """
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.wq = nn.Linear(input_size, hidden_size)
        self.wk = nn.Linear(output_size, hidden_size)
        self.wv = nn.Linear(output_size, hidden_size)
        self.att = Attention()
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wk.weight)
        nn.init.xavier_normal_(self.wv.weight)

    def forward(self, query, key, value, mask=None):
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)
        att_out, att_score = self.att(query, key, value)
        return att_out

class FusionBert_ML_predict(PreTrainedBertModel):
    def __init__(self, config: BertConfig, code_tokenizer, num_labels):
        super(FusionBert_ML_predict, self).__init__(config)
        self.codebert = CodeBERT(config, code_tokenizer.dx_voc, code_tokenizer.rx_voc)
        self.bert = BertModel(config) # 检查预训练任务加载的问题
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.text2dx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.text2rx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.dx2text_att = text2code_attention(config.code_hidden_size, 768, config.code_hidden_size)
        self.rx2text_att = text2code_attention(config.code_hidden_size, 768, config.code_hidden_size)
        self.classifier = nn.Linear(config.code_hidden_size*3+768*3, num_labels)
        self.apply(self.init_bert_weights)
    
    def forward(self,  code_input_ids, code_type_ids=None, input_lengths=None, txt_input_ids=None, txt_type_ids=None, txt_attention_mask=None, labels=None):
        """
        input_ids:[B*adm, seq_len]
        label: [B*(adm-1),num_class]
        input_lengths:[B]
        """
        batch_size, adm, code_len = code_input_ids.shape
        code_input, code_type = code_input_ids.view(-1, int(code_input_ids.shape[-1]//2)), code_type_ids.view(-1, code_type_ids.shape[-1])
        txt_input, txt_type, txt_att_mask = txt_input_ids.view(-1, txt_input_ids.shape[-1]), txt_type_ids.view(-1, txt_type_ids.shape[-1]), txt_attention_mask.view(-1, txt_attention_mask.shape[-1])
                
        token_types_ids = torch.cat([torch.zeros((1, code_input.size(1))), torch.ones(
            (1, code_input.size(1)))], dim=0).long().to(code_input.device)
        token_types_ids = token_types_ids.repeat(
            1 if code_input.size(0)//2 == 0 else code_input.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        code_all_layer, code_bert_pool = self.codebert(code_input, token_types_ids)
        txt_all_encode_layer_output, text_bert_pool = self.bert(txt_input, txt_type, txt_att_mask)

        code_all_layer = code_all_layer.view(2, -1, code_all_layer.size(1), code_all_layer.size(2))
        dx_code_all_layer = code_all_layer[0] # (B*adm, seq_len, H)
        rx_code_all_layer = code_all_layer[1] # (B*adm, seq_len, H)

        
        code_bert_pool = code_bert_pool.view(2, -1, code_bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = code_bert_pool[0] #self.dense[0](code_bert_pool[0])  # (B*adm, H)
        rx_bert_pool = code_bert_pool[1] #self.dense[1](code_bert_pool[1])  # (B*adm, H)

        dx_bert_att = self.dx2text_att(dx_bert_pool.unsqueeze(1), txt_all_encode_layer_output[-1], txt_all_encode_layer_output[-1], mask=txt_att_mask.unsqueeze(1)).squeeze(1)
        rx_bert_att = self.rx2text_att(rx_bert_pool.unsqueeze(1), txt_all_encode_layer_output[-1], txt_all_encode_layer_output[-1], mask=txt_att_mask.unsqueeze(1)).squeeze(1)
        
        # resudial
        dx_bert_pool += dx_bert_att
        rx_bert_pool += rx_bert_att

        dx_input = code_input_ids[:,:, :int(code_len//2)].view(-1, int(code_len//2))
        rx_input = code_input_ids[:,:, int(code_len//2):].view(-1, int(code_len//2))
        dx_mask = (dx_input>1).unsqueeze(1)
        rx_mask = (rx_input>1).unsqueeze(1)

        text_dx_att = self.text2dx_att(text_bert_pool.unsqueeze(1), dx_code_all_layer, dx_code_all_layer, mask=dx_mask).squeeze(1)
        text_rx_att = self.text2rx_att(text_bert_pool.unsqueeze(1), rx_code_all_layer, rx_code_all_layer, mask=rx_mask).squeeze(1)

        # resudial 
        txt2dx_bert_pool = text_bert_pool + text_dx_att
        txt2rx_bert_pool = text_bert_pool + text_rx_att

        dx_bert_pool = dx_bert_pool.view(code_input_ids.size(0), -1, dx_bert_pool.size(1)) # (B, adm, H)
        rx_bert_pool = rx_bert_pool.view(code_input_ids.size(0), -1, rx_bert_pool.size(1)) # (B, adm, H)
        
        text2dx_pool = txt2dx_bert_pool.view(code_input_ids.size(0), -1, txt2dx_bert_pool.size(1)) # (B, adm, H)
        text2rx_pool = txt2rx_bert_pool.view(code_input_ids.size(0), -1, txt2rx_bert_pool.size(1)) # (B, adm, H)

        
        if input_lengths is not None:
            logits = []
            truth = []
            for batch in range(input_lengths.shape[0]):
                len = input_lengths[batch]
                dx_pooler = dx_bert_pool[batch, :len+1, :]
                rx_pooler = rx_bert_pool[batch, :len+1, :]
                text2dx_pooler = text2dx_pool[batch, :len+1, :]
                text2rx_pooler = text2rx_pool[batch, :len+1, :]
                tmp_labels = labels[batch, :len, :]
                for i in range(len):
                    # mean
                    dx_mean = torch.mean(dx_pooler[0:i+1, :], dim=0, keepdim=True)
                    rx_mean = torch.mean(rx_pooler[0:i+1, :], dim=0, keepdim=True)
                    text2dx_mean = torch.mean(text2dx_pooler[0:i+1, :], dim=0, keepdim=True)
                    text2rx_mean = torch.mean(text2rx_pooler[0:i+1, :], dim=0, keepdim=True)
                    # concate
                    concat = torch.cat([dx_mean, rx_mean, dx_pooler[i+1, :].unsqueeze(dim=0), text2dx_mean, text2rx_mean, text2dx_pooler[i+1,:].unsqueeze(dim=0)], dim=-1) # (1, 2H)
                    logits.append(self.classifier(self.dropout(concat)))
                    truth.append(tmp_labels[i, :].unsqueeze(0))
            logits = torch.cat(logits, dim=0)
            truth = torch.cat(truth, dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, truth)
        return loss, logits, truth


class FusionBert_binary_predict(PreTrainedBertModel):
    def __init__(self, config: BertConfig, code_tokenizer, num_labels):
        super(FusionBert_binary_predict, self).__init__(config)
        self.codebert = CodeBERT(config, code_tokenizer.dx_voc, code_tokenizer.rx_voc)
        self.bert = BertModel(config)
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.text2dx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.text2rx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.dx2text_att = text2code_attention(config.code_hidden_size, 768, config.code_hidden_size)
        self.rx2text_att = text2code_attention(config.code_hidden_size, 768, config.code_hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear((768+config.code_hidden_size)*2, num_labels)
        self.apply(self.init_bert_weights)
    
    def forward(self,  code_input_ids, code_type_ids=None, txt_input_ids=None, txt_type_ids=None, txt_attention_mask=None, labels=None):
        """
        input_ids:[B, seq_len*2]
        label: [B,num_class]
        """
        batch_size, adm, code_len = code_input_ids.shape
        code_input, code_type = code_input_ids.view(-1, int(code_input_ids.shape[-1]//2)), code_type_ids.view(-1, code_type_ids.shape[-1])
        txt_input, txt_type, txt_att_mask = txt_input_ids.view(-1, txt_input_ids.shape[-1]), txt_type_ids.view(-1, txt_type_ids.shape[-1]), txt_attention_mask.view(-1, txt_attention_mask.shape[-1])
                
        token_types_ids = torch.cat([torch.zeros((1, code_input.size(1))), torch.ones(
            (1, code_input.size(1)))], dim=0).long().to(code_input.device)
        token_types_ids = token_types_ids.repeat(
            1 if code_input.size(0)//2 == 0 else code_input.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        code_all_layer, code_bert_pool = self.codebert(code_input, token_types_ids)
        txt_all_encode_layer_output, text_bert_pool = self.bert(txt_input, txt_type, txt_att_mask)

        code_all_layer = code_all_layer.view(2, -1, code_all_layer.size(1), code_all_layer.size(2))
        dx_code_all_layer = code_all_layer[0] # (B*adm, seq_len, H)
        rx_code_all_layer = code_all_layer[1] # (B*adm, seq_len, H)

        
        code_bert_pool = code_bert_pool.view(2, -1, code_bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = code_bert_pool[0] #self.dense[0](code_bert_pool[0])  # (B*adm, H)
        rx_bert_pool = code_bert_pool[1] #self.dense[1](code_bert_pool[1])  # (B*adm, H)

        dx_bert_att = self.dx2text_att(dx_bert_pool.unsqueeze(1), txt_all_encode_layer_output[-1], txt_all_encode_layer_output[-1], mask=txt_att_mask.unsqueeze(1)).squeeze(1)
        rx_bert_att = self.rx2text_att(rx_bert_pool.unsqueeze(1), txt_all_encode_layer_output[-1], txt_all_encode_layer_output[-1], mask=txt_att_mask.unsqueeze(1)).squeeze(1)
        
        # resudial
        dx_bert_pool += dx_bert_att
        rx_bert_pool += rx_bert_att

        dx_input = code_input_ids[:,:, :int(code_len//2)].view(-1, int(code_len//2))
        rx_input = code_input_ids[:,:, int(code_len//2):].view(-1, int(code_len//2))
        dx_mask = (dx_input>1).unsqueeze(1)
        rx_mask = (rx_input>1).unsqueeze(1)

        text_dx_att = self.text2dx_att(text_bert_pool.unsqueeze(1), dx_code_all_layer, dx_code_all_layer, mask=dx_mask).squeeze(1)
        text_rx_att = self.text2rx_att(text_bert_pool.unsqueeze(1), rx_code_all_layer, rx_code_all_layer, mask=rx_mask).squeeze(1)

        # resudial 
        txt2dx_bert_pool = text_bert_pool + text_dx_att
        txt2rx_bert_pool = text_bert_pool + text_rx_att

        concat = torch.cat([txt2dx_bert_pool, txt2rx_bert_pool, dx_bert_pool, rx_bert_pool], dim=-1)
        logits = self.classifier(self.dropout(concat)).view_as(labels)
                
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, logits, labels

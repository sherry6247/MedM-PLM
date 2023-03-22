'''
Author: your name
Date: 2021-10-23 17:03:08
LastEditTime: 2021-12-21 21:29:11
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /model/lsc_code/preTrain_in_MIMIC-III/multi_modal_pretrain/src/mutliModalBert.py
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
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import LayerNorm
import torch.nn.functional as F
from src.multiModal_config import FuseBertConfig as BertConfig
from src.multiModal_bert_model import CodeBERT, PreTrainedBertModel, Attention, FusionBertModel
import dill

from torch.autograd import Variable

from src.multiModal_bert_model import BertModel

logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def freeze_afterwards(model):
    for p in model.parameters():
        p.requires_grad = False


class TSNE(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(TSNE, self).__init__(config)

        self.bert = BERT(config, dx_voc, rx_voc)
        self.dx_voc = dx_voc
        self.rx_voc = rx_voc

        freeze_afterwards(self)

    def forward(self, output_dir, output_file='graph_embedding.tsv'):
        # dx_graph_emb = self.bert.embedding.ontology_embedding.dx_embedding.embedding
        # rx_graph_emb = self.bert.embedding.ontology_embedding.rx_embedding.embedding

        if not self.config.graph:
            print('save embedding not graph')
            rx_graph_emb = self.bert.embedding.word_embeddings(
                torch.arange(3, len(self.rx_voc.word2idx) + 3, dtype=torch.long))
            dx_graph_emb = self.bert.embedding.word_embeddings(
                torch.arange(len(self.rx_voc.word2idx) + 3, len(self.rx_voc.word2idx) + 3 + len(self.dx_voc.word2idx),
                             dtype=torch.long))
        else:
            print('save embedding graph')

            dx_graph_emb = self.bert.embedding.ontology_embedding.dx_embedding.get_all_graph_emb()
            rx_graph_emb = self.bert.embedding.ontology_embedding.rx_embedding.get_all_graph_emb()

        np.savetxt(os.path.join(output_dir, 'dx-' + output_file),
                   dx_graph_emb.detach().numpy(), delimiter='\t')
        np.savetxt(os.path.join(output_dir, 'rx-' + output_file),
                   rx_graph_emb.detach().numpy(), delimiter='\t')

        # def dump(prefix='dx-', emb):
        #     with open(prefix + output_file ,'w') as fout:
        #         m = emb.detach().cpu().numpy()
        #         for
        #         fout.write()

"""
只采用diagnosis和medication
"""
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

class textClsHead(nn.Module):
    def __init__(self, input_size, voc_size):
        super(textClsHead, self).__init__()
        self.cls = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(
        ), nn.Linear(input_size, voc_size))

    def forward(self, input):
        return self.cls(input)


class textSelfSupervisedHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(textSelfSupervisedHead, self).__init__()
        self.multi_cls = textClsHead(input_size, output_size)

    def forward(self, text_input):
        # inputs (B, hidden)
        # output logits
        return self.multi_cls(text_input)

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

# class Text2code_Attmodule(nn.Module):
#     def __init__(self, txt_size, dx_size, rx_size, hidden_size):
#         super(Text2code_Attmodule, self).__init__()
#         self.text2code_att = nn.ModuleList([text2code_attention(txt_size, hidden_size, dx_size),
#                                             text2code_attention(txt_size, hidden_size, rx_size)])
#     def forward(self, txt_input, dx_input, rx_input):
#         return self.text2code_att[0](txt_input, dx_input, dx_input), \
#                 self.text2code_att[1](txt_input, rx_input, rx_input)


class MultiModal_Pretrain(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(MultiModal_Pretrain, self).__init__(config)
        self.dx_voc_size = len(dx_voc.word2idx)
        self.rx_voc_size = len(rx_voc.word2idx)

        self.codebert = CodeBERT(config, dx_voc, rx_voc)
        self.cls = SelfSupervisedHead(
            config, self.dx_voc_size, self.rx_voc_size)

        self.bert = BertModel(config).cuda()

        self.text2dx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.text2rx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.Text2dxcls = textSelfSupervisedHead(768, self.dx_voc_size)
        self.Text2rxcls = textSelfSupervisedHead(768, self.rx_voc_size)

        self.apply(self.init_bert_weights)

    def forward(self, dx_inputs, rx_inputs, dx_labels=None, rx_labels=None, txt_input_ids=None, txt_type_ids=None, txt_attention_mask=None):
        # inputs (B, , max_len)
        # bert_pool (B, hidden)
        dx_all_encode_layer_output, dx_bert_pool = self.codebert(dx_inputs, torch.zeros(
            (dx_inputs.size(0), dx_inputs.size(1))).long().to(dx_inputs.device))
        rx_all_encode_layer_output, rx_bert_pool = self.codebert(rx_inputs, torch.zeros(
            (rx_inputs.size(0), rx_inputs.size(1))).long().to(rx_inputs.device))
        txt_all_encode_layer_output, txt_bert_pool = self.bert(txt_input_ids, txt_type_ids, txt_attention_mask)
        
        dx2dx, rx2dx, dx2rx, rx2rx = self.cls(dx_bert_pool, rx_bert_pool)
        text_dx_att = self.text2dx_att(txt_bert_pool.unsqueeze(1), dx_all_encode_layer_output, dx_all_encode_layer_output).squeeze(1)
        text_rx_att = self.text2rx_att(txt_bert_pool.unsqueeze(1), rx_all_encode_layer_output, rx_all_encode_layer_output).squeeze(1)
        text2dx = self.Text2dxcls(text_dx_att)
        text2rx = self.Text2rxcls(text_rx_att)
        # output logits
        if rx_labels is None or dx_labels is None:
            return F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx), F.sigmoid(text2dx), F.sigmoid(text2rx)
        else:
            loss = F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(rx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(dx2rx, rx_labels) + \
                F.binary_cross_entropy_with_logits(rx2rx, rx_labels) + \
                F.binary_cross_entropy_with_logits(text2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(text2rx, rx_labels)
            return loss, F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx), F.sigmoid(text2dx), F.sigmoid(text2rx)

class MultiModal_coAtt_Pretrain(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(MultiModal_coAtt_Pretrain, self).__init__(config)
        self.dx_voc_size = len(dx_voc.word2idx)
        self.rx_voc_size = len(rx_voc.word2idx)

        self.codebert = CodeBERT(config, dx_voc, rx_voc)
        self.cls = SelfSupervisedHead(
            config, self.dx_voc_size, self.rx_voc_size)

        self.bert = BertModel(config).cuda()

        self.text2dx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.text2rx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.Text2dxcls = textSelfSupervisedHead(768, self.dx_voc_size)
        self.Text2rxcls = textSelfSupervisedHead(768, self.rx_voc_size)

        self.dx2text_att = text2code_attention(config.code_hidden_size, 768, config.code_hidden_size)
        self.rx2text_att = text2code_attention(config.code_hidden_size, 768, config.code_hidden_size)

        self.apply(self.init_bert_weights)

    def forward(self, dx_inputs, rx_inputs, dx_labels=None, rx_labels=None, txt_input_ids=None, txt_type_ids=None, txt_attention_mask=None):
        # inputs (B, , max_len)
        # bert_pool (B, hidden)
        dx_all_encode_layer_output, dx_bert_pool = self.codebert(dx_inputs, torch.zeros(
            (dx_inputs.size(0), dx_inputs.size(1))).long().to(dx_inputs.device))
        rx_all_encode_layer_output, rx_bert_pool = self.codebert(rx_inputs, torch.zeros(
            (rx_inputs.size(0), rx_inputs.size(1))).long().to(rx_inputs.device))
        txt_all_encode_layer_output, txt_bert_pool = self.bert(txt_input_ids, txt_type_ids, txt_attention_mask)

        dx_bert_pool = self.dx2text_att(dx_bert_pool.unsqueeze(1), txt_all_encode_layer_output[-1], txt_all_encode_layer_output[-1], mask=txt_attention_mask.unsqueeze(1)).squeeze(1)
        rx_bert_pool = self.rx2text_att(rx_bert_pool.unsqueeze(1), txt_all_encode_layer_output[-1], txt_all_encode_layer_output[-1], mask=txt_attention_mask.unsqueeze(1)).squeeze(1)
               
        dx_mask = (dx_inputs>1).unsqueeze(1)
        rx_mask = (rx_inputs>1).unsqueeze(1)

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls(dx_bert_pool, rx_bert_pool)
        text_dx_att = self.text2dx_att(txt_bert_pool.unsqueeze(1), dx_all_encode_layer_output, dx_all_encode_layer_output, mask=dx_mask).squeeze(1)
        text_rx_att = self.text2rx_att(txt_bert_pool.unsqueeze(1), rx_all_encode_layer_output, rx_all_encode_layer_output, mask=rx_mask).squeeze(1)
        text2dx = self.Text2dxcls(text_dx_att)
        text2rx = self.Text2rxcls(text_rx_att)
        # output logits
        if rx_labels is None or dx_labels is None:
            return F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx), F.sigmoid(text2dx), F.sigmoid(text2rx)
        else:
            loss = F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(rx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(dx2rx, rx_labels) + \
                F.binary_cross_entropy_with_logits(rx2rx, rx_labels) + \
                F.binary_cross_entropy_with_logits(text2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(text2rx, rx_labels)
            return loss, F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx), F.sigmoid(text2dx), F.sigmoid(text2rx)


class MultiModal_coAtt_residual_Pretrain(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(MultiModal_coAtt_residual_Pretrain, self).__init__(config)
        self.dx_voc_size = len(dx_voc.word2idx)
        self.rx_voc_size = len(rx_voc.word2idx)

        self.codebert = CodeBERT(config, dx_voc, rx_voc)
        self.cls = SelfSupervisedHead(
            config, self.dx_voc_size, self.rx_voc_size)

        self.bert = BertModel(config).cuda()

        self.text2dx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.text2rx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.Text2dxcls = textSelfSupervisedHead(768, self.dx_voc_size)
        self.Text2rxcls = textSelfSupervisedHead(768, self.rx_voc_size)

        self.dx2text_att = text2code_attention(config.code_hidden_size, 768, config.code_hidden_size)
        self.rx2text_att = text2code_attention(config.code_hidden_size, 768, config.code_hidden_size)

        self.apply(self.init_bert_weights)

    def forward(self, dx_inputs, rx_inputs, dx_labels=None, rx_labels=None, txt_input_ids=None, txt_type_ids=None, txt_attention_mask=None):
        # inputs (B, , max_len)
        # bert_pool (B, hidden)
        dx_all_encode_layer_output, dx_bert_pool = self.codebert(dx_inputs, torch.zeros(
            (dx_inputs.size(0), dx_inputs.size(1))).long().to(dx_inputs.device))
        rx_all_encode_layer_output, rx_bert_pool = self.codebert(rx_inputs, torch.zeros(
            (rx_inputs.size(0), rx_inputs.size(1))).long().to(rx_inputs.device))
        txt_all_encode_layer_output, txt_bert_pool = self.bert(txt_input_ids, txt_type_ids, txt_attention_mask)

        dx_bert_att = self.dx2text_att(dx_bert_pool.unsqueeze(1), txt_all_encode_layer_output[-1], txt_all_encode_layer_output[-1]).squeeze(1)
        rx_bert_att = self.rx2text_att(rx_bert_pool.unsqueeze(1), txt_all_encode_layer_output[-1], txt_all_encode_layer_output[-1]).squeeze(1)
        
        # resudial
        dx_bert_pool += dx_bert_att
        rx_bert_pool += rx_bert_att

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls(dx_bert_pool, rx_bert_pool)
        text_dx_att = self.text2dx_att(txt_bert_pool.unsqueeze(1), dx_all_encode_layer_output, dx_all_encode_layer_output).squeeze(1)
        text_rx_att = self.text2rx_att(txt_bert_pool.unsqueeze(1), rx_all_encode_layer_output, rx_all_encode_layer_output).squeeze(1)

        # resudial 
        txt2dx_bert_pool = txt_bert_pool + text_dx_att
        txt2rx_bert_pool = txt_bert_pool + text_rx_att

        text2dx = self.Text2dxcls(txt2dx_bert_pool)
        text2rx = self.Text2rxcls(txt2rx_bert_pool)
        # output logits
        if rx_labels is None or dx_labels is None:
            return F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx), F.sigmoid(text2dx), F.sigmoid(text2rx)
        else:
            loss = F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(rx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(dx2rx, rx_labels) + \
                F.binary_cross_entropy_with_logits(rx2rx, rx_labels) + \
                F.binary_cross_entropy_with_logits(text2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(text2rx, rx_labels)
            return loss, F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx), F.sigmoid(text2dx), F.sigmoid(text2rx)


class FusionBert_ML_predict(PreTrainedBertModel):
    def __init__(self, config: BertConfig, code_tokenizer, num_labels):
        super(FusionBert_ML_predict, self).__init__(config)
        self.codebert = CodeBERT(config, code_tokenizer.dx_voc, code_tokenizer.rx_voc)
        self.bert = BertModel(config) # 检查预训练任务加载的问题
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.text2dx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.text2rx_att = text2code_attention(768, config.code_hidden_size, 768)
        self.classifier = nn.Linear(config.code_hidden_size*3+768*3, num_labels)
        self.apply(self.init_bert_weights)
    
    def forward(self,  code_input_ids, code_type_ids=None, input_lengths=None, txt_input_ids=None, txt_type_ids=None, txt_attention_mask=None, labels=None):
        """
        input_ids:[B*adm, seq_len]
        label: [B*(adm-1),num_class]
        input_lengths:[B]
        """
        code_input, code_type = code_input_ids.view(-1, int(code_input_ids.shape[-1]//2)), code_type_ids.view(-1, code_type_ids.shape[-1])
        txt_input, txt_type, txt_att_mask = txt_input_ids.view(-1, txt_input_ids.shape[-1]), txt_type_ids.view(-1, txt_type_ids.shape[-1]), txt_attention_mask.view(-1, txt_attention_mask.shape[-1])
                
        token_types_ids = torch.cat([torch.zeros((1, code_input.size(1))), torch.ones(
            (1, code_input.size(1)))], dim=0).long().to(code_input.device)
        token_types_ids = token_types_ids.repeat(
            1 if code_input.size(0)//2 == 0 else code_input.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        code_all_layer, code_bert_pool = self.codebert(code_input, token_types_ids)

        code_all_layer = code_all_layer.view(2, -1, code_all_layer.size(1), code_all_layer.size(2))
        dx_code_all_layer = code_all_layer[0] # (B*adm, seq_len, H)
        rx_code_all_layer = code_all_layer[1] # (B*adm, seq_len, H)

        
        code_bert_pool = code_bert_pool.view(2, -1, code_bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](code_bert_pool[0])  # (B*adm, H)
        rx_bert_pool = self.dense[1](code_bert_pool[1])  # (B*adm, H)
        dx_bert_pool = dx_bert_pool.view(code_input_ids.size(0), -1, dx_bert_pool.size(1)) # (B, adm, H)
        rx_bert_pool = rx_bert_pool.view(code_input_ids.size(0), -1, rx_bert_pool.size(1)) # (B, adm, H)
        
        # _, code_bert_pool = self.codebert(code_input, code_type)
        _, text_bert_pool = self.bert(txt_input, txt_type, txt_att_mask)
        # code_bert_pool = code_bert_pool.view(code_input_ids.size(0), -1, code_bert_pool.size(1)) # (B, adm, H)
        # text_bert_pool = text_bert_pool.view(code_input_ids.size(0), -1, text_bert_pool.size(1)) # (B, adm, H)
        text_dx_att = self.text2dx_att(text_bert_pool.unsqueeze(1), dx_code_all_layer, dx_code_all_layer).squeeze(1) # (B*adm, 768)
        text_rx_att = self.text2rx_att(text_bert_pool.unsqueeze(1), rx_code_all_layer, rx_code_all_layer).squeeze(1) # (B*adm, 768)
        # text2dx_pool = self.text2dx_trans(text_dx_att)
        # text2rx_pool = self.text2rx_trans(text_rx_att)
        text2dx_pool = text_dx_att
        text2rx_pool = text_rx_att
        text2dx_pool = text2dx_pool.view(code_input_ids.size(0), -1, text2dx_pool.size(1)) # (B, adm, H)
        text2rx_pool = text2rx_pool.view(code_input_ids.size(0), -1, text2rx_pool.size(1)) # (B, adm, H)

        
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
        self.textbert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.code_hidden_size+768, num_labels)
        self.apply(self.init_bert_weights)
    
    def forward(self,  code_input_ids, code_type_ids=None, input_lengths=None, txt_input_ids=None, txt_type_ids=None, txt_attention_mask=None, labels=None):
        """
        input_ids:[B, seq_len]
        label: [B,num_class]
        """
        code_input, code_type = code_input_ids.view(-1, code_input_ids.shape[-1]), code_type_ids.view(-1, code_type_ids.shape[-1])
        txt_input, txt_type, txt_att_mask = txt_input_ids.view(-1, txt_input_ids.shape[-1]), txt_type_ids.view(-1, txt_type_ids.shape[-1]), txt_attention_mask.view(-1, txt_attention_mask.shape[-1])
        _, code_bert_pool = self.codebert(code_input, code_type)
        _, text_bert_pool = self.bert(txt_input, txt_type, txt_att_mask)
        concat = torch.concat([code_bert_pool, text_bert_pool], dim=-1)
        logits = self.classifier(self.dropout(concat))
                
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, logits


class MappingHead(nn.Module):
    def __init__(self, config: BertConfig):
        super(MappingHead, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.code_hidden_size, config.code_hidden_size),
                                   nn.ReLU())

    def forward(self, input):
        return self.dense(input)

class GBERT_Predict(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer, num_labels):
        super(GBERT_Predict, self).__init__(config)
        self.bert = CodeBERT(config, tokenizer.dx_voc, tokenizer.rx_voc)
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, 2*config.hidden_size),
                                 nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.rx_voc.word2idx)))
        self.dropout = nn.Dropout(config.code_hidden_dropout_prob)
        self.apply(self.init_bert_weights)

    def forward(self, code_input_ids, code_type_ids=None, input_lengths=None, txt_input_ids=None, txt_type_ids=None, txt_attention_mask=None, labels=None):
        """
        :param input_ids: [B, max_seq_len] where B = 2*adm
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]
        :return:
        """
        token_types_ids = torch.cat([torch.zeros((1, code_input_ids.size(1))), torch.ones(
            (1, code_input_ids.size(1)))], dim=0).long().to(code_input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if code_input_ids.size(0)//2 == 0 else code_input_ids.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.bert(code_input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)

        # mean and concat for rx prediction task
        if input_lengths is None:
            concat = torch.concat([dx_bert_pool, rx_bert_pool], dim=-1)
            logits = self.cls(self.dropout(concat))
        else:
            rx_logits = []
            for i in range(labels.size(0)):
                # mean
                dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
                rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
                # concat
                concat = torch.cat(
                    [dx_mean, rx_mean, dx_bert_pool[i+1, :].unsqueeze(dim=0)], dim=-1)
                rx_logits.append(self.cls(concat))

            logits = torch.cat(rx_logits, dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, rx_logits


class OnlyText_Predict(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer, num_labels):
        super(OnlyText_Predict, self).__init__(config)
        self.TextBert = BertModel(config)
        self.txt_dense = nn.Sequential(nn.Linear(768, config.hidden_size),
                                   nn.ReLU())
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self,  code_input_ids=None, code_type_ids=None, input_lengths=None, txt_input_ids=None, txt_type_ids=None, txt_attention_mask=None, labels=None):
        """
        :param input_ids: [B, max_seq_len] where B = 2*adm
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]
        :return:
        """
        
        _, txt_bert_pool = self.TextBert(txt_input_ids, txt_type_ids, txt_attention_mask)
        txt_bert_pool = self.txt_dense(txt_bert_pool)
        if input_lengths is None:
            logits = self.out(self.dropout(txt_bert_pool))
        else:
            logits = txt_bert_pool[:txt_input_ids.size(0), :]
            logits = self.out(logits)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, logits
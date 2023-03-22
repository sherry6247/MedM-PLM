'''
Author: your name
Date: 2021-10-23 16:42:26
LastEditTime: 2021-10-28 15:27:08
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /model/lsc_code/preTrain_in_MIMIC-III/multi_modal_pretrain/main.py
'''
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import random
import numpy as np
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from src.get_options import args_parser
from src.mutliModalBert import MultiModal_Pretrain, MultiModal_coAtt_Pretrain, MultiModal_coAtt_residual_Pretrain
from transformers import BertTokenizer, BertModel

from src.utils import metric_report, t2n, get_n_params
from src.multiModal_config import FuseBertConfig as BertConfig
from src.multiModalDataset import MultiModalDataset, CodeTextProcessor, EHRTokenizer, convert_examples_to_features
from src.Trainer import Trainer

import os

n_gpus = torch.cuda.device_count()
assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main():
    args = args_parser()

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")
    args.nprocs = torch.cuda.device_count()
#     mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

# def main_worker(local_rank, nprocs, args):

    # local_rank = args.local_rank
    # nprocs = args.nprocs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    processors = {
        "codeText":CodeTextProcessor,
    }
    print("Loading Dataset")
    # load tokenizer
    code_tokenizer = EHRTokenizer(args.data_dir)

    # text tokenizer
    txt_tokenizer = BertTokenizer.from_pretrained(args.clinical_bert_dir)

    processor = processors['codeText']()
    print("TRAIN")
    train_dataset = processor.get_train_examples(args.data_dir)
    print('DEV')
    eval_dataset = processor.get_dev_examples(args.data_dir)
    # print('TEST')
    # test_dataset = processor.get_test_examples(args.data_dir)

    print('Loading Model: ' + args.model_name)
    if args.use_pretrain:
        print("Use Pretraining model")
        model = MultiModal_Pretrain.from_pretrained(args.pretrain_dir, dx_voc=code_tokenizer.dx_voc,
                                            rx_voc=code_tokenizer.rx_voc)
    else:
        config = BertConfig(
            vocab_size_or_config_json_file=len(code_tokenizer.vocab.word2idx))
        config.graph = args.graph
        
        print("Use clinicalBert pretraining model parameters...")
          
        model = MultiModal_coAtt_residual_Pretrain.from_pretrained(args.clinical_bert_dir, dx_voc=code_tokenizer.dx_voc,
                                            rx_voc=code_tokenizer.rx_voc)
    
    # setup(local_rank, nprocs)
    
    # torch.cuda.set_device(local_rank)
    # model.cuda(local_rank)
    # args.train_batch_size = int(args.train_batch_size / nprocs)
    # args.eval_batch_size = int(args.eval_batch_size / nprocs)
    # model = DistributedDataParallel(model, device_ids=[local_rank],
    #                 find_unused_parameters=True)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        model.to('cuda')

    print('# of model parameters: ' + str(get_n_params(model)))
    for param in model.parameters():
        param.requires_grad = True
    for param in model.module.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.module.bert.encoder.layer[:10].parameters():
        param.requires_grad = False
        
    optimizer = Adam(
                    filter(lambda x : x.requires_grad, model.parameters()), 
                    lr=args.learning_rate)  
    
    rng = random.Random(args.seed)
    train_features = convert_examples_to_features(train_dataset, args.max_seq_length, args.max_predictions_per_seq,
                                                    txt_tokenizer, code_tokenizer, rng)
    
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Batch size = %d", args.train_batch_size)
    # print(" Num steps = %d", int(len(train_dataset)/args.train_batch_size))

    train_data = MultiModalDataset(train_features)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, 
                                    sampler=train_sampler,
                                    batch_size=args.train_batch_size,
                                    num_workers=0,
                                    pin_memory=True)
    eval_features = convert_examples_to_features(eval_dataset, args.max_seq_length, args.max_predictions_per_seq,
                                                    txt_tokenizer, code_tokenizer, rng)
    eval_data = MultiModalDataset(eval_features)
    # eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_data)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, 
                                sampler=eval_sampler,
                                batch_size=args.eval_batch_size,
                                num_workers=0,
                                pin_memory=True)
    # Trainer
    trainer = Trainer(train_dataloader, eval_dataloader, model, optimizer, args)
    trainer.train()
        

if __name__=='__main__':
    main()
    

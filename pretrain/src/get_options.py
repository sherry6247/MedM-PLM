'''
Author: your name
Date: 2021-10-23 16:45:48
LastEditTime: 2021-10-26 21:58:05
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /model/lsc_code/preTrain_in_MIMIC-III/multi_modal_pretrain/src/get_options.py
'''

import argparse
def args_parser():
     parser = argparse.ArgumentParser()

     # Required parameters
     parser.add_argument("--model_name", default='GBert-pretraining', type=str, required=False,
                         help="model name")
     parser.add_argument("--data_dir",
                         default='../data',
                         type=str,
                         required=False,
                         help="The input data dir.")
     parser.add_argument("--pretrain_dir", default='../saved/GBert-predict', type=str, required=False,
                         help="pretraining model dir.")
     parser.add_argument("--train_file", default='data-multi-visit.pkl', type=str, required=False,
                         help="training data file.")
     parser.add_argument("--output_dir",
                         default='../saved/',
                         type=str,
                         required=False,
                         help="The output directory where the model checkpoints will be written.")

     # Other parameters
     parser.add_argument("--use_pretrain",
                         default=False,
                         action='store_true',
                         help="if use ontology embedding")
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
                              "than this will be padded. default 55")
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
                         default=64,
                         type=int,
                         help="Total batch size for training.")
     parser.add_argument("--eval_batch_size",
                         default=64,
                         type=int,
                         help="Total batch size for training.")
     parser.add_argument("--learning_rate",
                         default=5e-4,
                         type=float,
                         help="The initial learning rate for Adam.")
     parser.add_argument("--num_train_epochs",
                         default=10.0,
                         type=float,
                         help="Total number of training epochs to perform.")
     parser.add_argument("--no_cuda",
                         action='store_true',
                         help="Whether not to use CUDA when available")
     parser.add_argument('--seed',
                         type=int,
                         default=1203,
                         help="random seed for initialization")
     parser.add_argument("--warmup_proportion",
                         default=0.1,
                         type=float,
                         help="Proportion of training to perform linear learning rate warmup for. "
                              "E.g., 0.1 = 10%% of training.")

     parser.add_argument("--max_predictions_per_seq", default=20)
     parser.add_argument("--clinical_bert_dir", 
                         default="./ClinicalBert/model/pretraining/")
     parser.add_argument("--local_rank", default=-1, type=int,
                    help='node rank for distributed training')
     parser.add_argument("--max_visit_len", type=int, default=29, help='how many visit number will be calculated')
     parser.add_argument('--DDP', action='store_true')

     args = parser.parse_args()
    
     return args
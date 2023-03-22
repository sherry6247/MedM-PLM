'''
Author: your name
Date: 2021-10-23 17:33:30
LastEditTime: 2021-10-24 19:22:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /model/lsc_code/preTrain_in_MIMIC-III/multi_modal_pretrain/src/Trainer.py
'''
import os
import torch
from tensorboardX import SummaryWriter
import shutil
import numpy as np
from src.mutliModalBert import MultiModal_Pretrain
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
from src.utils import metric_report, t2n
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, train_dataloader:DataLoader,  evel_dataloader: DataLoader, model:MultiModal_Pretrain, optimizer=None, args=None):
        self.train_dataloader = train_dataloader
        self.evel_dataloader = evel_dataloader
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.local_rank = args.local_rank
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_local_0 = self.local_rank is None or self.local_rank == 0
        print("Trainer rank {}, is_local_0:{}".format(self.local_rank, self.is_local_0))

        if self.device == 'cuda':
            self.model = self.model.cuda()
        self.args.output_dir = os.path.join(args.output_dir, args.model_name)
        os.makedirs(args.output_dir, exist_ok=True)

        print("save model in {}".format(self.args.output_dir))

        
        self.model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        self.output_model_file = os.path.join(
            args.output_dir, "pytorch_model.bin")

        # if self.is_local_0:
        self.board = SummaryWriter(self.args.output_dir)
        # else:
        #     self.board = None
                    
    def train(self):
        print("Start Training")
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        dx_acc_best, rx_acc_best = 0, 0
        acc_name = 'prauc'
        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch", ncols=100):
            print('')
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            prog_iter = tqdm(self.train_dataloader, desc='Training', ncols=100, colour='blue') #if self.is_local_0 else self.train_dataloader
            
            self.model.train()
            for _, batch in enumerate(prog_iter):
                batch = tuple(t.to('cuda') for t in batch)

                dx_input_ids, rx_input_ids, txt_input_ids, txt_type_ids, txt_att_mask, dx_labels, rx_labels = batch
                loss, dx2dx, rx2dx, dx2rx, rx2rx, text2dx, text2rx = self.model(
                    dx_input_ids, rx_input_ids, dx_labels, rx_labels, txt_input_ids, txt_type_ids, txt_att_mask)
                if torch.cuda.device_count() > 1:
                    loss = torch.sum(loss)
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += 1
                nb_tr_steps += 1

                # Display loss
                # if self.is_local_0 == 0:
                prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

                self.optimizer.step()
                self.optimizer.zero_grad()

                torch.cuda.empty_cache()
                # if self.is_local_0:
                with open(os.path.join(self.args.output_dir, 'train_result.txt'), 'a', encoding='utf-8') as f:
                    f.write("epoch:{}\ttrian loss:{}\n".format(epoch, loss.item()))

            # if self.is_local_0:
            self.board.add_scalar('train/loss', tr_loss / nb_tr_steps, global_step)
            global_step += 1


            if self.args.do_eval:
                print('')
                logger.info("***** Running eval *****")
                
                self.model.eval()
                dx2dx_y_preds = []
                rx2dx_y_preds = []
                text2dx_y_preds = []
                dx_y_trues = []

                dx2rx_y_preds = []
                rx2rx_y_preds = []
                text2rx_y_preds = []
                rx_y_trues = []
                eval_iterator_bar = tqdm(self.evel_dataloader, ncols=100, desc='Evaluating') #if self.is_local_0 else self.eval_dataloader
                for batch in eval_iterator_bar:
                    with torch.no_grad():
                        batch = tuple(t.to('cuda') for t in batch)
                        
                        dx_input_ids, rx_input_ids, txt_input_ids, txt_type_ids, txt_att_mask, dx_labels, rx_labels = batch
                        dx2dx, rx2dx, dx2rx, rx2rx, text2dx, text2rx = self.model(
                        dx_input_ids, rx_input_ids, \
                            txt_input_ids=txt_input_ids, txt_type_ids=txt_type_ids, txt_attention_mask=txt_att_mask)
                        text2dx_y_preds.append(t2n(text2dx))
                        text2rx_y_preds.append(t2n(text2rx))
                        
                    dx2dx_y_preds.append(t2n(dx2dx))
                    rx2dx_y_preds.append(t2n(rx2dx))
                    dx2rx_y_preds.append(t2n(dx2rx))
                    rx2rx_y_preds.append(t2n(rx2rx))

                    dx_y_trues.append(
                        t2n(dx_labels))
                    rx_y_trues.append(
                        t2n(rx_labels))

                print('')
                print('dx2dx')
                dx2dx_acc_container = metric_report(
                    np.concatenate(dx2dx_y_preds, axis=0), np.concatenate(dx_y_trues, axis=0), self.args.therhold)
                print('rx2dx')
                rx2dx_acc_container = metric_report(
                    np.concatenate(rx2dx_y_preds, axis=0), np.concatenate(dx_y_trues, axis=0), self.args.therhold)
                print('dx2rx')
                dx2rx_acc_container = metric_report(
                    np.concatenate(dx2rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0), self.args.therhold)
                print('rx2rx')
                rx2rx_acc_container = metric_report(
                    np.concatenate(rx2rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0), self.args.therhold)
                
                print('text2dx')
                text2dx_acc_container = metric_report(
                    np.concatenate(text2dx_y_preds, axis=0), np.concatenate(dx_y_trues, axis=0), self.args.therhold
                )
                print('text2rx')
                text2rx_acc_container = metric_report(
                    np.concatenate(text2rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0), self.args.therhold
                )
                # if self.is_local_0:
                for k, v in text2dx_acc_container.items():
                    self.board.add_scalar(
                        'eval_text2dx/{}'.format(k), v, global_step)
                for k, v in text2rx_acc_container.items():
                    self.board.add_scalar(
                        'eval_text2rx/{}'.format(k), v, global_step)

                # keep in history
                # if self.is_local_0:
                for k, v in dx2dx_acc_container.items():
                    self.board.add_scalar(
                        'eval_dx2dx/{}'.format(k), v, global_step)
                for k, v in rx2dx_acc_container.items():
                    self.board.add_scalar(
                        'eval_rx2dx/{}'.format(k), v, global_step)
                for k, v in dx2rx_acc_container.items():
                    self.board.add_scalar(
                        'eval_dx2rx/{}'.format(k), v, global_step)
                for k, v in rx2rx_acc_container.items():
                    self.board.add_scalar(
                        'eval_rx2rx/{}'.format(k), v, global_step)
                # if self.args.local_rank==0:
                if rx2rx_acc_container[acc_name] > rx_acc_best:
                    rx_acc_best = rx2rx_acc_container[acc_name]
                    early_stop_epoch = epoch
                    # save model
                    torch.save(self.model_to_save.state_dict(),
                            self.output_model_file)

                    with open(os.path.join(self.args.output_dir, 'bert_config.json'), 'w', encoding='utf-8') as fout:
                        fout.write(self.model_to_save.config.to_json_string())
                    with open(os.path.join(self.args.output_dir, 'eval_result.txt'), 'a', encoding='utf-8') as ef:
                        for k, v in dx2dx_acc_container.items():
                            ef.write('eval_dx2dx/{}=={}=={}\n'.format(k, v, global_step))
                        for k, v in rx2dx_acc_container.items():
                            ef.write('eval_rx2dx/{}=={}=={}\n'.format(k, v, global_step))
                        for k, v in dx2rx_acc_container.items():
                            ef.write('eval_dx2rx/{}=={}=={}\n'.format(k, v, global_step))
                        for k, v in rx2rx_acc_container.items():
                            ef.write('eval_rx2rx/{}=={}=={}\n'.format(k, v, global_step))
                        for k, v in text2dx_acc_container.items():
                            ef.write('eval_text2dx/{}=={}=={}\n'.format(k, v, global_step))
                        for k, v in text2rx_acc_container.items():
                            ef.write('eval_text2rx/{}=={}=={}\n'.format(k, v, global_step))
                if epoch - early_stop_epoch > 10:
                    break
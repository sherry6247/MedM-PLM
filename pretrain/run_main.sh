CUDA_VISIBLE_DEVICES=0,1 python main.py \
                --model_name multi_modal_coAtt_residual_pretrain \
				--num_train_epochs 200  \
				--do_train \
				--graph  \
				--max_visit_len 10

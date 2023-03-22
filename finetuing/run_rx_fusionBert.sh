#!/bin/bash
for se in 2022 2023 2024 ; do 
    for lr in 5e-5 1e-5 ; do
        CUDA_VISIBLE_DEVICES=0,1 python fusionBert_predict.py \
            --partion $spr \
            --learning_rate $lr \
            --seed $se \
            --model_name FusionBert_coAtt_residual_rx_predict \
            --graph  \
            --do_train  \
            --data_name  multilabel  \
            --predict_task  rx  \
            --model_choice  fusion_ml \
            --max_visit_len 10 \
            --use_pretrain  \
            --pretrain_dir /home/lsc/model/lsc_code/preTrain_in_MIMIC-III/pretrained_model/multi_modal_coAtt_residual_pretrain
    done
done
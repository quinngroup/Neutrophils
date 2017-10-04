#!/usr/bin/env bash
ROOT_PATH='..'
RESULT_PATH="${ROOT_PATH}/models/size-64-64_wgangp-adam"

mkdir -p ${RESULT_PATH}

python ${ROOT_PATH}/code/main.py \
--dataroot /media/narita/Data/Neutrophils/train_test_images/train_validation_combined/neutrophils_train \
--image_height 64 --image_width 64 --cuda \
--num_iter 50000 \
--experiment ${RESULT_PATH} --dataset_type folder \
--model_type DCGAN --GAN_algorithm WGAN-GP \
--num_disc_iters 10 --optimizer adam \
--lrD 0.0001 --lrG 0.0001 --beta1 0.0 --beta2 0.9 \
--weight_decay 1e-5 | tee ${RESULT_PATH}/train_stdout.txt

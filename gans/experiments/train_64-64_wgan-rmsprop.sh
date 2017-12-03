#!/usr/bin/env bash
ROOT_PATH='..'
RESULT_PATH="${ROOT_PATH}/models/size-64-64_wgan-rmsprop_10000"

mkdir -p ${RESULT_PATH}

python ${ROOT_PATH}/code/main.py \
--dataroot /media/narita/Data/Neutrophils/train_test_images/train_validation_combined/training_region_crops/neutrophils_train \
--image_height 64 --image_width 64 --cuda \
--num_iter 10000 --experiment ${RESULT_PATH} \
--dataset_type folder \
--model_type DCGAN --GAN_algorithm WGAN \
--num_disc_iters 5 --optimizer rmsprop \
--lrD 0.00005 --lrG 0.00005 | tee ${RESULT_PATH}/train_stdout.txt

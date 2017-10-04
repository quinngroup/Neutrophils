#!/usr/bin/env bash
#!/usr/bin/env bash
ROOT_PATH='..'
MODEL_NAME='size-64-64_wgan-rmsprop'
NET_PATH="${ROOT_PATH}/models/${MODEL_NAME}/saved_models/netG_iter_50000.pth"

RESULT_PATH="./reconstruction/${MODEL_NAME}_50000"
mkdir -p ${RESULT_PATH}

python ${ROOT_PATH}/code/main.py \
--dataroot /media/narita/Data/Neutrophils/train_test_images/train_validation_combined/training_region_crops/neutrophils_train \
--image_height 64 --image_width 64 --cuda \
--num_iter 50000 --experiment ${RESULT_PATH} \
--model_type DCGAN --mode reconstruction \
--netG ${NET_PATH} \
--test_data /media/narita/Data/Neutrophils/train_test_images/testing_region_crops/neutrophils_test | tee ${RESULT_PATH}/reconstruction_stdout.txt

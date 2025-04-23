#!/bin/bash

# This is your argument

gpu_device="0"

nproc_per_node=1
master_port=$(shuf -i 10000-45000 -n 1)
echo "Using port $master_port for communication."

export TOKENIZERS_PARALLELISM=false


data_root=/home/yzhang/research/nanobody/data
model_root=./checkpoint



MODEL_TYPE='nanobert'

task='vhh_affinity'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=256
lr=5e-3
data=''
data_file_train=train_score.csv; data_file_val=val_score.csv; data_file_test=test_score.csv
MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/ft/${task}/score/opensource/${MODEL_TYPE}  
seed=12345

        

# TODO: 需要修改
master_port=$(shuf -i 10000-45000 -n 1)
echo "Using port $master_port for communication."
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"

echo ${MODEL_PATH}

${EXEC_PREFIX} \
    downstream/train_vhh_affinity.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data}_seed${seed} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps ${gradient_accumulation} \
    --learning_rate ${lr} \
    --num_train_epochs 100 \
    --save_steps 400 \
    --output_dir ${OUTPUT_PATH}/${data} \
    --evaluation_strategy steps \
    --eval_steps 2 \
    --warmup_steps 50 \
    --logging_steps 1 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --model_type ${MODEL_TYPE} \
    --fp16 \

data_file_train=train_seq.csv; data_file_val=val_seq.csv; data_file_test=test_seq.csv
OUTPUT_PATH=./outputs/ft/${task}/seq/opensource/${MODEL_TYPE}  

${EXEC_PREFIX} \
    downstream/train_vhh_affinity.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data}_seed${seed} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps ${gradient_accumulation} \
    --learning_rate ${lr} \
    --num_train_epochs 100 \
    --save_steps 400 \
    --output_dir ${OUTPUT_PATH}/${data} \
    --evaluation_strategy steps \
    --eval_steps 2 \
    --warmup_steps 50 \
    --logging_steps 1 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --model_type ${MODEL_TYPE} \
    --fp16 \
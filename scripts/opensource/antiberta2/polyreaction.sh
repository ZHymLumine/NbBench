#!/bin/bash
SECONDS=0
# Common settings
export TOKENIZERS_PARALLELISM=false
gpu_device="0"
nproc_per_node=1
data_root="/home/yzhang/research/nanobody_benchmark/data"
model_root="./checkpoint"
MODEL_TYPE='antiberta2'
seed=12345

master_port=$(shuf -i 10000-45000 -n 1)
echo "Using port $master_port for communication."

EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"


echo "Starting polyreaction prediction task..."
task='polyreaction'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=256
lr=5e-3
data=''
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/probe/${task}/opensource/${MODEL_TYPE}_lr_${lr}


${EXEC_PREFIX} \
downstream/train_polyreaction.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data}_seed${seed} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps ${gradient_accumulation} \
    --learning_rate ${lr} \
    --num_train_epochs 50 \
    --save_steps 200 \
    --output_dir ${OUTPUT_PATH}/${data} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 200 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --model_type ${MODEL_TYPE} \

echo "All tasks completed successfully!" 

duration=$SECONDS
echo "Total runtime: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s"
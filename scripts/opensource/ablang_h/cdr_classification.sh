#!/bin/bash
SECONDS=0
# Common settings
export TOKENIZERS_PARALLELISM=false
gpu_device="4"
nproc_per_node=1
data_root="/home/yzhang/research/nanobody_benchmark/data"
model_root="./checkpoint"
MODEL_TYPE='ablang_h'
seed=12345

master_port=$(shuf -i 10000-45000 -n 1)
echo "Using port $master_port for communication."

EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"

echo "Starting CDR classification task..."
task='CDRs_classification'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=160
lr=5e-3
data=''
data_file_train=train_filtered.csv; data_file_val=val_filtered.csv; data_file_test=test_filtered.csv
MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/probe/${task}/opensource/${MODEL_TYPE}_lr_${lr}


${EXEC_PREFIX} \
downstream/train_cdr_classification.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data}_seed${seed} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${gradient_accumulation} \
    --learning_rate ${lr} \
    --num_train_epochs 1 \
    --save_steps 300 \
    --output_dir ${OUTPUT_PATH}/${data} \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --warmup_steps 50 \
    --logging_steps 200 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --model_type ${MODEL_TYPE} \

echo "All tasks completed successfully!" 

duration=$SECONDS
echo "Total runtime: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s"
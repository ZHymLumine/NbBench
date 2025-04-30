#!/bin/bash
SECONDS=0
# Common settings
export TOKENIZERS_PARALLELISM=false
gpu_device="3"
nproc_per_node=1
data_root="/home/yzhang/research/nanobody_benchmark/data"
model_root="./checkpoint"
MODEL_TYPE='esm-2'
seed=12345

master_port=$(shuf -i 10000-45000 -n 1)
echo "Using port $master_port for communication."

EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"


echo "Starting thermostability prediction task..."
task='thermo'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=256
lr=5e-3
data=''
data_file_train=train_tm.csv; data_file_val=val_tm.csv; data_file_test=test_tm.csv
# MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
MODEL_PATH="facebook/esm2_t33_650M_UR50D"
OUTPUT_PATH=./outputs/ft/${task}/tm_lr_${lr}/opensource/${MODEL_TYPE}_lr_${lr}

${EXEC_PREFIX} \
    downstream/train_thermo.py \
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
    --freeze False

data_file_train=train_seq.csv; data_file_val=val_seq.csv; data_file_test=test_seq.csv
OUTPUT_PATH=./outputs/ft/${task}/seq_lr_${lr}/opensource/${MODEL_TYPE}_lr_${lr}

${EXEC_PREFIX} \
    downstream/train_thermo.py \
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
    --freeze False

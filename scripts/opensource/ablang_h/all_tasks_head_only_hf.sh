#!/bin/bash
SECONDS=0
# Common settings
#!/bin/bash
SECONDS=0
# Common settings
export TOKENIZERS_PARALLELISM=false
gpu_device="1"
nproc_per_node=1
data_root="/home/yzhang/research/nanobody_benchmark/data"
model_root="./checkpoint"
MODEL_TYPE='ablang_h'
seed=1234

master_port=$(shuf -i 10000-45000 -n 1)
echo "Using port $master_port for communication."

EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port "

# 1. CDR Classification Task
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
dataset_name="ZYMScott/VRClassification"

echo ${MODEL_PATH}

${EXEC_PREFIX} \
downstream/train_cdr_classification_hug.py \
    --model_name_or_path $MODEL_PATH \
    --use_hf_dataset True \
    --dataset_name ${dataset_name} \
    --data_path  $DATA_PATH/$data \
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
    --model_type ${MODEL_TYPE}

# 2. CDR Infilling Task
echo "Starting CDR infilling task..."
task='CDRs_infilling'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=160
lr=5e-3
data=''
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test_filtered.csv
MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/probe/${task}/opensource/${MODEL_TYPE}_lr_${lr}
dataset_name="ZYMScott/CDRInfilling"

${EXEC_PREFIX} \
downstream/train_cdr_infilling_hug.py \
    --model_name_or_path $MODEL_PATH \
    --use_hf_dataset True \
    --dataset_name ${dataset_name} \
    --data_path  $DATA_PATH/$data \
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
    --model_type ${MODEL_TYPE}

# 3. Interaction Prediction Task
echo "Starting interaction prediction task..."

# 3.1 AVIDa-SARS-CoV-2

task='AVIDa-SARS-CoV-2'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=160
lr=5e-3
data=''
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/probe/${task}/opensource/${MODEL_TYPE}_lr_${lr}
dataset_name="ZYMScott/AVIDa-SARS-CoV-2"

${EXEC_PREFIX} \
    downstream/train_interaction_hug.py \
    --model_name_or_path $MODEL_PATH \
    --use_hf_dataset True \
    --dataset_name ${dataset_name} \
    --data_path  $DATA_PATH/$data \
    --run_name ${MODEL_TYPE}_${data}_seed${seed} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${gradient_accumulation} \
    --learning_rate ${lr} \
    --num_train_epochs 50 \
    --save_steps 400 \
    --output_dir ${OUTPUT_PATH}/${data} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 200 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --model_type ${MODEL_TYPE} \

# 3.2 AVIDa-hIL6
task='AVIDa-hIL6'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=256
lr=5e-3
data=''
data_file_train=train.csv; data_file_val=val_sampled.csv; data_file_test=test.csv
MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/probe/${task}/opensource/${MODEL_TYPE}_lr_${lr}
dataset_name="ZYMScott/AVIDa-hIL6"

${EXEC_PREFIX} \
    downstream/train_interaction_hug.py \
    --model_name_or_path $MODEL_PATH \
    --use_hf_dataset True \
    --dataset_name ${dataset_name} \
    --data_path  $DATA_PATH/$data \
    --run_name ${MODEL_TYPE}_${data}_seed${seed} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps ${gradient_accumulation} \
    --learning_rate ${lr} \
    --num_train_epochs 30 \
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


# 4. Paratope Prediction Task
echo "Starting paratope prediction task..."
task='paratope'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=160
lr=5e-3
data=''
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/probe/${task}/opensource/${MODEL_TYPE}_lr_${lr}
dataset_name="ZYMScott/Paratope"

${EXEC_PREFIX} \
downstream/train_paratope_hug.py \
    --model_name_or_path $MODEL_PATH \
    --use_hf_dataset True \
    --dataset_name ${dataset_name} \
    --data_path  $DATA_PATH/$data \
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

# # 5. Polyreactivity Prediction Task
echo "Starting polyreactivity prediction task..."
task='polyreaction'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=160
lr=5e-3
data=''
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/probe/${task}/opensource/${MODEL_TYPE}_lr_${lr}
dataset_name="ZYMScott/polyreaction"

${EXEC_PREFIX} \
downstream/train_polyreaction_hug.py \
    --model_name_or_path $MODEL_PATH \
    --use_hf_dataset True \
    --dataset_name ${dataset_name} \
    --data_path  $DATA_PATH/$data \
    --run_name ${MODEL_TYPE}_${data}_seed${seed} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 8 \
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

#6. Sdab Type Prediction Task
echo "Starting sdab type prediction task..."
task='sdabtype'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=160
lr=5e-3
data=''
data_file_train=train_filtered.csv; data_file_val=val_filtered.csv; data_file_test=test_filtered.csv
MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/probe/${task}/opensource/${MODEL_TYPE}_lr_${lr}
dataset_name="ZYMScott/nanobody_type"

${EXEC_PREFIX} \
downstream/train_sdab_type_hug.py \
    --model_name_or_path $MODEL_PATH \
    --use_hf_dataset True \
    --dataset_name ${dataset_name} \
    --data_path  $DATA_PATH/$data \
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
    --model_type ${MODEL_TYPE}

# 7. VHH Affinity Prediction Task
echo "Starting affinity prediction task..."
task='vhh_affinity'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=160
lr=5e-3
data=''
data_file_train=train_score.csv; data_file_val=val_score.csv; data_file_test=test_score.csv
MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/probe/${task}/score/opensource/${MODEL_TYPE}_lr_${lr}
dataset_name="ZYMScott/vhh_affinity-score"

${EXEC_PREFIX} \
    downstream/train_vhh_affinity_hug.py \
    --model_name_or_path $MODEL_PATH \
    --use_hf_dataset True \
    --dataset_name ${dataset_name} \
    --data_path  $DATA_PATH/$data \
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

data_file_train=train_seq.csv; data_file_val=val_seq.csv; data_file_test=test_seq.csv
OUTPUT_PATH=./outputs/probe/${task}/seq/opensource/${MODEL_TYPE}_lr_${lr}
dataset_name="ZYMScott/vhh_affinity-seq"

${EXEC_PREFIX} \
    downstream/train_vhh_affinity_hug.py \
    --model_name_or_path $MODEL_PATH \
    --use_hf_dataset True \
    --dataset_name ${dataset_name} \
    --data_path  $DATA_PATH/$data \
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

# 8. Thermostability Prediction Task
echo "Starting thermostability prediction task..."
task='thermo'
DATA_PATH=${data_root}/downstream/${task}
batch_size=32
gradient_accumulation=2
model_max_length=160
lr=5e-3
data=''
data_file_train=train_tm.csv; data_file_val=val_tm.csv; data_file_test=test_tm.csv
MODEL_PATH=${model_root}/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/probe/${task}/tm_lr_${lr}/opensource/${MODEL_TYPE}_lr_${lr}
dataset_name="ZYMScott/thermo-tm"

${EXEC_PREFIX} \
    downstream/train_thermo_hug.py \
    --model_name_or_path $MODEL_PATH \
    --use_hf_dataset True \
    --dataset_name ${dataset_name} \
    --data_path  $DATA_PATH/$data \
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
    --model_type ${MODEL_TYPE}

data_file_train=train_seq.csv; data_file_val=val_seq.csv; data_file_test=test_seq.csv
OUTPUT_PATH=./outputs/probe/${task}/seq_lr_${lr}/opensource/${MODEL_TYPE}_lr_${lr}
dataset_name="ZYMScott/thermo-seq"
${EXEC_PREFIX} \
    downstream/train_thermo_hug.py \
    --model_name_or_path $MODEL_PATH \
    --use_hf_dataset True \
    --dataset_name ${dataset_name} \
    --data_path  $DATA_PATH/$data \
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
    --model_type ${MODEL_TYPE}

echo "All tasks completed successfully!" 


duration=$SECONDS
echo "Total runtime: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s"
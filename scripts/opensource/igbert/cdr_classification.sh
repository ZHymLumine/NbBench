# Common settings
export TOKENIZERS_PARALLELISM=false
gpu_device="0"
nproc_per_node=1
data_root="/home/yzhang/research/nanobody_benchmark/data"
model_root="./checkpoint"
MODEL_TYPE='igbert'
seed=12345

# Training script for IgBert CDR Classification
python train.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $model_root/$MODEL_TYPE \
    --task_name cdr_classification \
    --do_train \
    --do_eval \
    --data_dir $data_root \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir $model_root/$MODEL_TYPE/output \
    --overwrite_output_dir \
    --logging_steps 500 \
    --save_steps 1000 \
    --seed $seed \
    --gpu_device $gpu_device \
    --nproc_per_node $nproc_per_node \
    --fp16 
#!/bin/bash

hostname
nvidia-smi
task=mQA

for data in  TyDiQA XQuAD MLQA
  do
model=../saved_models/llama-7b-hf
#  model=../saved_models/llama_P3_16_2e-5
CUDA_VISIBLE_DEVICES=0 python  $task/train-QA.py \
        --output_dir ${model} \
        --model_type llama \
        --model_name_or_path ${model} --cache_dir ./cache \
        --data_path ${data} \
        --do_eval --do_lower_case \
        --per_gpu_eval_batch_size=16  \
        --overwrite_output_dir --flash_attn
done

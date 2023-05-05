#!/bin/bash

hostname
nvidia-smi
task=mCLS

for data in copa_winogrande_wic_rte_mnlim_mnlimm_obqa_SST-2 # xwinograd_xstorycloze_xnli_xcopa
  do
    for i in 90000
do
  model=../saved_models/llama-7b-hf
#  model=../saved_models/llama_P3_16_2e-5
CUDA_VISIBLE_DEVICES=0 python $task/train-CLS.py \
        --output_dir ${model} \
        --model_type llama \
        --model_name_or_path ${model} --cache_dir ../cache \
        --data_path ${data} \
         --do_eval --do_lower_case \
        --per_gpu_eval_batch_size=4  \
        --overwrite_output_dir --flash_attn
done
done


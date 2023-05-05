"""
Enable flash attention during fine-tuning.
Adopted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_mem.py
"""
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from train import train

if __name__ == "__main__":
    train()

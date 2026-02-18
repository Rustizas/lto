from datasets import load_dataset
import re

ds = load_dataset("uonlp/CulturaX", "lt", split="train")

def clean(example):
    text = example["text"]
    #If too short
    if len(text) < 200:
        return False
    # Spam url
    if text.count("http") > 3:
        return False
    # Spam cleaning
    digits = sum(x.isdigit() for x in text)
    if digits / len(text) > 0.2:
        return False
    #seo spam
    words = text.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return False
    return True

#filtering
ds_clean = ds.filter(clean, num_proc=8)

ds_clean.save_to_disk("/home/rustis/projektai/Res/lto/data/processed/culturax_clean")
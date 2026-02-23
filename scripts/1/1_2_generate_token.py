import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import torch
import pandas as pd
import tiktoken

load_dotenv()
dir = os.getenv("dir")

tokenizers = {
    "llama3" : "meta-llama/Meta-Llama-3-8B",
    "qwen2": "Qwen/Qwen2.5-7B",
    "mistral": "mistralai/Mistral-7B-v0.3",
}

df = pd.read_csv(f"{dir}/raw/wikipedia_lt_eng.csv")

def has_repetition(text, threshold=5):
    words = str(text).lower().split()
    counts = Counter(words)
    return any(c > threshold for c in counts.values())

df = df[~df["en"].apply(has_repetition)]

# tiktoken gpt-4
enc = tiktoken.get_encoding("cl100k_base")
df["lt_gpt4"] = df["lt"].apply(lambda x: len(enc.encode(str(x))))
df["en_gpt4"] = df["en"].apply(lambda x: len(enc.encode(str(x))))
df["ratio_gpt4"] = round(df["lt_gpt4"] / df["en_gpt4"], 2)
#avg chars per token
df["avg_char_per_token_gpt4_lt"] = round(df["lt"].str.len() / df["lt_gpt4"],2)
df["avg_char_per_token_gpt4_en"] = round(df["en"].str.len() / df["en_gpt4"],2)

# cycle through tokenizers
for name, path in tokenizers.items():
    tokenizer = AutoTokenizer.from_pretrained(path)
    df[f"lt_{name}"] = df["lt"].apply(lambda x: len(tokenizer.encode(str(x))))
    df[f"en_{name}"] = df["en"].apply(lambda x: len(tokenizer.encode(str(x))))
    df[f"ratio_{name}"] = round(df[f"lt_{name}"]/df[f"en_{name}"], 2)
    df[f"avg_char_per_token_{name}_lt"] = round(df["lt"].str.len() / df[f"lt_{name}"], 2)
    df[f"avg_char_per_token_{name}_en"] = round(df["en"].str.len() / df[f"en_{name}"], 2)

df = df[(df["ratio_llama3"] >= 0.5) & (df["ratio_llama3"] <= 5)]

df.to_csv(f"{dir}/raw/token_counts_5k.csv", index=False)
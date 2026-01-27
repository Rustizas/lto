from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import tiktoken

tokenizers = {
    "llama3" : "meta-llama/Meta-Llama-3-8B",
    "qwen2": "Qwen/Qwen2.5-7B",
    "mistral": "mistralai/Mistral-7B-v0.3",
}

df = pd.read_csv("/home/rustis/projektai/Res/lto/data/raw/wikipedia_lt_eng.csv")

#tiktoken gpt-4
enc = tiktoken.get_encoding("cl100k_base")
df["lt_gpt4"] = df["lt"].apply(lambda x: len(enc.encode(str(x))))
df["en_gpt4"] = df["en"].apply(lambda x: len(enc.encode(str(x))))
df["ratio_gpt4"] = round(df["lt_gpt4"] / df["en_gpt4"], 2)

#cycle through tokenizers
for name, path in tokenizers.items():
    tokenizer = AutoTokenizer.from_pretrained(path)
    df[f"lt_{name}"] = df["lt"].apply(lambda x: len(tokenizer.encode(str(x))))
    df[f"en_{name}"] = df["en"].apply(lambda x: len(tokenizer.encode(str(x))))
    df[f"ratio_{name}"] = round(df[f"lt_{name}"]/df[f"en_{name}"], 2)

df.to_csv("/home/rustis/projektai/Res/lto/data/raw/token_counts_5k.csv", index=False)
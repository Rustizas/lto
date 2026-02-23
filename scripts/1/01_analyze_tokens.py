import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

load_dotenv()
dir = os.getenv("dir")

tokenizers = {
    "llama3" : "meta-llama/Meta-Llama-3-8B",
    "qwen2": "Qwen/Qwen2.5-7B",
    "mistral": "mistralai/Mistral-7B-v0.3",
}

df = pd.read_csv(f'{dir}/raw/sentences.csv', engine='python', on_bad_lines='warn')

for name, path in tokenizers.items():
    tokenizer = AutoTokenizer.from_pretrained(path)
    df[f'lt_{name}'] = df['lt'].apply(lambda x: len(tokenizer.encode(str(x))))
    df[f'en_{name}'] = df['en'].apply(lambda x: len(tokenizer.encode(str(x))))
    df[f'ratio_{name}'] = round(df[f'lt_{name}']/df[f'en_{name}'], 2)

df.to_csv(f'{dir}/processed/token_counts.csv', index=False)
import os
from dotenv import load_dotenv
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch
import pandas as pd
from datasets import Dataset

load_dotenv()
dir = os.getenv("dir")

df = pd.read_csv(f"{dir}/raw/wikipedia_lithuanian_5k.csv")
hf_dataset = Dataset.from_pandas(df)

translator = pipeline(
    task="translation",
    model="facebook/nllb-200-distilled-600M",
    torch_dtype=torch.bfloat16,
    device=0,
    max_length=1024)
translated = []

for out in translator(KeyDataset(hf_dataset, "lt"), src_lang="lit_Latn", tgt_lang="eng_Latn", batch_size=32, max_length=1024):
    translated.append(out[0]["translation_text"])

df["en"] = translated
df.to_csv(f"{dir}/raw/wikipedia_lt_eng.csv", index=False)
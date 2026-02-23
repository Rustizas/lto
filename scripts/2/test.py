import os
from dotenv import load_dotenv
from datasets import load_from_disk

load_dotenv()
dir = os.getenv("dir")

ds_clean = load_from_disk(f"{dir}/processed/culturax_clean")

total_chars = sum(len(x["text"]) for x in ds_clean)
total_gb = total_chars / (1024**3)
print(f"Teksto: {total_gb:.1f} GB")
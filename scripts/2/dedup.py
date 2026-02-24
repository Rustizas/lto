import hashlib
from datasets import load_from_disk
from dotenv import load_dotenv
import os 
load_dotenv()
dir = os.getenv("dir")
ds = load_from_disk(f"{dir}/processed/culturax_normalized")

def hash(row):
    vals = hashlib.md5(row["text"].encode()).hexdigest()
    return {"hash": vals}

ds_hashed = ds.map(hash, num_proc=8)

#dup searching
df = ds_hashed.select_columns(["hash"]).to_pandas()
unique_indeksai = df.drop_duplicates(subset=["hash"]).index.tolist()

ds_clean = ds_hashed.select(unique_indeksai)
ds_clean = ds_clean.remove_columns(["hash"])
ds_clean.save_to_disk(f"{dir}/processed/culturax_dedup")
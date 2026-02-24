from datasets import load_from_disk
import os 
from dotenv import load_dotenv

load_dotenv()
dir = os.getenv("dir")
ds = load_from_disk(f"{dir}/processed/culturax_dedup")
#sampling 3m 
ds_sample = ds.shuffle(seed=42).select(range(min(5_000_000, len(ds))))

ds_split = ds_sample.train_test_split(test_size=0.05, seed=42)
ds_val_test = ds_split["test"].train_test_split(test_size=0.5, seed=42)

#split 95% train 2.5% val 2.5% test
train = ds_split["train"]
val = ds_val_test["train"]
test = ds_val_test["test"]

train.save_to_disk(f"{dir}/processed/train")
val.save_to_disk(f"{dir}/processed/val")
test.save_to_disk(f"{dir}/processed/test")
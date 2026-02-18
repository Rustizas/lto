from datasets import load_from_disk

ds_clean = load_from_disk("/home/rustis/projektai/Res/lto/data/processed/culturax_clean")

total_chars = sum(len(x["text"]) for x in ds_clean)
total_gb = total_chars / (1024**3)
print(f"Teksto: {total_gb:.1f} GB")
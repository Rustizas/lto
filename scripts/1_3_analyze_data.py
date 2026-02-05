import pandas as pd
import numpy
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM

df = pd.read_csv("/home/rustis/projektai/Res/lto/data/raw/token_counts_5k.csv")

model_names = ["llama3", "qwen2", "mistral", "gpt4"]

data = []

for name in model_names:
	data.append({
		"avg_ratio": round(df[f"ratio_{name}"].mean(), 2), #Avg ratio of Lithuanian to English tokens
		"std_dev_ratio": round(df[f"ratio_{name}"].std(), 2), #Standart deviation of the ratio
		"avg_char_per_token_lt": round(df[f"avg_char_per_token_{name}_lt"].mean(), 2), #Avg symbols per token in Lithuanian
		"min_ratio": round(df[f"ratio_{name}"].min(), 2), #Min ratio of Lithuanian to English tokens
		"max_ratio": round(df[f"ratio_{name}"].max(), 2), #Max ratio of Lithuanian to English tokens
		"percentile_95": round(df[f"ratio_{name}"].quantile(0.95), 2), #95th percentile of the ratio
		"percentile_5": round(df[f"ratio_{name}"].quantile(0.05), 2),
		"median": round(df[f"ratio_{name}"].median(), 2) #Median of the ratio
		"n_samples": len(df)
	})

results_df = pd.DataFrame(data, index=model_names)
results_df.to_csv("/home/rustis/projektai/Res/lto/data/processed/summary_stats.csv")

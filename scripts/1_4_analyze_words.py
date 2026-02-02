import pandas as pd
import numpy
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

df = pd.read_csv("/home/rustis/projektai/Res/lto/data/raw/wikipedia_lt_eng.csv")

letters = "ąčęėįšųūžĄČĘĖĮŠŲŪŽ"
tokenizers = {
    "llama3" : "meta-llama/Meta-Llama-3-8B",
    "qwen2": "Qwen/Qwen2.5-7B",
    "mistral": "mistralai/Mistral-7B-v0.3",
}
utf8_rez = []
hard_words=[]

enc = tiktoken.get_encoding("cl100k_base")


#Counting huggingface tokenizers
for name, path in tokenizers.items():
	tokenizer = AutoTokenizer.from_pretrained(path)
	for char in letters:
		ids = tokenizer.encode(char, add_special_tokens=False)
		if len(ids) > 1:
			utf8_rez.append({
				"model": name,
				"char": char,
				"token_count": len(ids),
				"readable": str([tokenizer.decode([i]) for i in ids])
			})
	x_counter = Counter()
	for text in df["lt"]:
		words = str(text).split()
		for word in words:
			clean = word.strip(".,!?()\"':;-")
			ids = tokenizer.encode(clean, add_special_tokens=False)
			if len(ids) >= 5:
				x_counter[clean] += 1
	for word, count in x_counter.most_common(100):
		ids = tokenizer.encode(word, add_special_tokens=False)
		breakdown = [tokenizer.decode([i]) for i in ids]
		hard_words.append({
			"model": name,
			"word": word,
			"token_count": count,
			"breakdown": breakdown
		})
#distinct gpt-4 counting
enc = tiktoken.get_encoding("cl100k_base")
model_name = "gpt4"
for char in letters:
	ids = enc.encode(char)
	if len(ids) > 1:
		utf8_rez.append({
			"model": model_name,
			"char": char,
			"token_count": len(ids),
			"readable": str([enc.decode([i]) for i in ids])
		})
gpt4_counter = Counter()
for text in df["lt"]:
	words = str(text).split()
	for word in words:
		clean = word.strip(".,!?()\"':;-")
		ids = enc.encode(clean)
		if len(ids) >= 5:
			gpt4_counter[clean] += 1
for word, count in gpt4_counter.most_common(100):
	ids = enc.encode(word)
	breakdown = [enc.decode([i]) for i in ids]
	hard_words.append({
		"model": model_name,
		"word": word,
		"token_count": count,
		"breakdown": breakdown
	})

pd.DataFrame(utf8_rez).to_csv("/home/rustis/projektai/Res/lto/data/processed/utf8_tokenization.csv", index=False)
pd.DataFrame(hard_words).to_csv("/home/rustis/projektai/Res/lto/data/processed/hard_words_tokenization.csv", index=False)
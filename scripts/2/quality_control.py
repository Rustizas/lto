import os
import re
import html
from dotenv import load_dotenv
from datasets import load_from_disk
import hashlib

load_dotenv()
dir = os.getenv("dir")

ds = load_from_disk(f"{dir}/processed/culturax_clean")

def normalize(text):
	#URL removing
	text = re.sub(r"http\S+", "", text) 
	#HTML removing
	text = re.sub(r"<[^>]+>", "", text)
	#HTML entities
	text = html.unescape(text)
	#whitespace normalization
	text = re.sub(r'\s+', ' ', text)
	text = text.strip()
	return text

def lithuanian(text, threshold=0.005): 
	#checking if the text is in Lithuanian based on the ratio of Lithuanian characters
	if not text: return False
	letters = set("ąčęėįšųūžĄČĘĖĮŠŲŪŽ")
	count = sum(char in letters for char in text)
	return (count / len(text)) >= threshold

def quality_control(row):
	text = row["text"]
	#Normalizing
	text = normalize(text)
	# Too short after normalization
	if len(text) < 100:
		return False
	#not lithuanian
	if not lithuanian(text):
		return False
	# too much non-alphanumeric characters
	non_alphanumeric = len(re.findall(r'[^\w\s]', text))
	if (non_alphanumeric / len(text)) > 0.3:
		return False
	return True

ds_filtered = ds.filter(quality_control, num_proc=8)
ds_normalized = ds_filtered.map(lambda x: {"text": normalize(x["text"])}, num_proc=8)
ds_normalized.save_to_disk(f"{dir}/processed/culturax_normalized")
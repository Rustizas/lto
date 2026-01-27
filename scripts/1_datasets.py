import spacy
import re
from datasets import load_dataset
import pandas as pd 

nlp = spacy.load("lt_core_news_sm")
ds = load_dataset("wikimedia/wikipedia", "20231101.lt", split="train")

def clean_text(text):
	#wikipedia text cleaning
    text = re.sub(r'\[\d+\]', '', text)
    text = text.replace('\n', ' ')
    return " ".join(text.split())

def prose(y):
    #filtering out only valid sentences
    if not y.endswith('.'): 
        return False
    if y.endswith(' a.') or y.endswith(' m.'): 
        return False
    if not y[0].isupper(): 
        return False
    
    return True
	
def sentences_list(dataset, target=5000):
    sentences = []
    for row in dataset:
        text = clean_text(row['text'])
        doc = nlp(text)
        for sent in doc.sents:
            x = sent.text.strip()
            # filtering out short sentences/dictionary definitions
            if len(x) > 45 and prose(x):
                sentences.append(x)
            if len(sentences) >= target:
                return sentences
    return sentences

sentences_5k = sentences_list(ds, 5000)
df = pd.DataFrame(sentences_5k, columns=['sentences'])
df.to_csv("/home/rustis/projektai/Res/lto/data/raw/wikipedia_lithuanian_5k.csv", index=False, encoding='utf-8')
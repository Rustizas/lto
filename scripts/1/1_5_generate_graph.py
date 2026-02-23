import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

load_dotenv()
dir = os.getenv("dir")

df = pd.read_csv(f"{dir}/raw/token_counts_5k.csv")
sns.set_theme(style="whitegrid")

#histogram for gpt-4 token counts comparison between Lithuanian and English languages
plt.figure(figsize=(12, 8))
sns.histplot(df["lt_gpt4"], color = "blue", label="Lithuanian(LT)", kde=True, element="step", alpha=0.5)
sns.histplot(df['en_gpt4'], color="orange", label="English(EN)", kde=True, element="step", alpha=0.5)
plt.title("GPT-4 Tokenų count difference (LT vs EN)", fontsize=14)
plt.xlabel("Token count in a sentence", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.savefig(f"{dir}/graphs/histogram_lt_en_gpt4_token_counts.png")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))
df['lt_char_count'] = df['lt'].astype(str).str.len()
df['en_char_count'] = df['en'].astype(str).str.len()
sns.scatterplot(
    x=df['lt_char_count'], 
    y=df['lt_gpt4'], 
    color="blue", 
    alpha=0.5, 
    label="Lithuanian(LT)", 
    s=15 
)
sns.scatterplot(
    x=df['en_char_count'], 
    y=df['en_gpt4'], 
    color="orange", 
    alpha=0.5, 
    label="English(EN)", 
    s=15
)
plt.title("Sentence lenght vs token count", fontsize=14)
plt.xlabel("Character count in a sentence", fontsize=12)
plt.ylabel("Token count in a sentence", fontsize=12)
plt.legend()
plt.savefig(f"{dir}/graphs/scatter_lt_en_gpt4_token_counts.png")
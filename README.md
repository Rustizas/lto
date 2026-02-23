# LTO

Analyzing tokenization efficiency of Lithuanian language across LLM tokenizers (GPT-4, LLaMA 3, Qwen2, Mistral) compared to English.

## Key Findings

Lithuanian text requires ~2x more tokens than equivalent English across all tested tokenizers.

Several Lithuanian characters (ė, į, ų, Ą, Ę...) are split into multiple byte-level tokens, especially in LLaMA 3 and GPT-4.

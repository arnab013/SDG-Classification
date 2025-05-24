import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import aggregate_logits_chunkwise

def predict_sdg_labels(claims, abstract, description, model, tokenizer, thresholds, device='cuda'):
    text = f"{claims} [SEP] {abstract} [SEP] {description}"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    max_chunk_len = 512 - 2
    chunks = [tokens[i:i + max_chunk_len] for i in range(0, len(tokens), max_chunk_len)]
    chunk_texts = [tokenizer.decode(chunk) for chunk in chunks]
    encodings = tokenizer(chunk_texts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu().numpy()
    agg_logits = aggregate_logits_chunkwise(logits)
    probs = 1 / (1 + np.exp(-agg_logits))
    pred = (probs >= thresholds).astype(int)
    if pred.sum() == 0:
        return ['sdg_0']
    return [f"sdg_{i+1}" for i, v in enumerate(pred) if v == 1]

# Example usage:
# model = AutoModelForSequenceClassification.from_pretrained('./sdg_bert_model').to('cuda')
# tokenizer = AutoTokenizer.from_pretrained('./sdg_bert_model')
# thresholds = np.load('sdg_thresholds.npy')
# sdgs = predict_sdg_labels(claims, abstract, description, model, tokenizer, thresholds)

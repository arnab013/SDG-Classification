import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import pandas as pd

from patent_sdg_dataset import load_patent_data, PatentSDGDataset
from collate_fn import patent_sdg_collate_fn
from model import get_model
from utils import evaluate_multilabel, find_optimal_thresholds, aggregate_logits_chunkwise

DATA_PATH = "/home/jovyan/epo-sdg/output/*.csv.gz"
TEXT_COLS = ["claims", "abstract_text", "description_text"]
LABEL_COLS = [f"sdg_{i}" for i in range(1, 18)]
META_COLS = ["publication_number"]
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3
PRETRAINED_MODEL = "anferico/bert-for-patents"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading data...")
df = load_patent_data(DATA_PATH, TEXT_COLS, LABEL_COLS, META_COLS)
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
train_set = PatentSDGDataset(train_df, tokenizer, TEXT_COLS, LABEL_COLS, max_length=MAX_LENGTH)
val_set   = PatentSDGDataset(val_df, tokenizer, TEXT_COLS, LABEL_COLS, max_length=MAX_LENGTH)
test_set  = PatentSDGDataset(test_df, tokenizer, TEXT_COLS, LABEL_COLS, max_length=MAX_LENGTH)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=patent_sdg_collate_fn)
val_loader   = DataLoader(val_set, batch_size=1, collate_fn=patent_sdg_collate_fn)
test_loader  = DataLoader(test_set, batch_size=1, collate_fn=patent_sdg_collate_fn)

model = get_model(PRETRAINED_MODEL, num_labels=len(LABEL_COLS)).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=2.5e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
loss_fn = torch.nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        batch_size = len(batch['input_ids'])
        optimizer.zero_grad()
        batch_loss = 0
        for i in range(batch_size):
            input_ids = batch['input_ids'][i].to(DEVICE)
            attention_mask = batch['attention_mask'][i].to(DEVICE)
            labels = batch['labels'][i].to(DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(logits, labels.unsqueeze(0).repeat(logits.shape[0], 1))
                sample_loss = loss.mean()
            scaler.scale(sample_loss).backward()
            batch_loss += sample_loss.item()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += batch_loss / batch_size
    print(f"Epoch {epoch+1} training loss: {running_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'][0].to(DEVICE)
            attention_mask = batch['attention_mask'][0].to(DEVICE)
            labels = batch['labels'][0].cpu().numpy()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu().numpy()
            agg_logits = aggregate_logits_chunkwise(logits)
            probs = 1 / (1 + np.exp(-agg_logits))
            all_probs.append(probs)
            all_labels.append(labels)
    all_probs = np.stack(all_probs)
    all_labels = np.stack(all_labels)
    metrics = evaluate_multilabel(all_labels, all_probs, threshold=0.6)
    print("Validation Macro F1:", metrics['macro_f1'])

optimal_thresholds = find_optimal_thresholds(all_labels, all_probs)
print("Optimal thresholds per SDG:", optimal_thresholds)

# Test evaluation
model.eval()
test_probs, test_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'][0].to(DEVICE)
        attention_mask = batch['attention_mask'][0].to(DEVICE)
        labels = batch['labels'][0].cpu().numpy()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu().numpy()
        agg_logits = aggregate_logits_chunkwise(logits)
        probs = 1 / (1 + np.exp(-agg_logits))
        test_probs.append(probs)
        test_labels.append(labels)
test_probs = np.stack(test_probs)
test_labels = np.stack(test_labels)
test_metrics = evaluate_multilabel(test_labels, test_probs, threshold=optimal_thresholds)
print("Test Macro F1 (optimal threshold):", test_metrics['macro_f1'])

model.save_pretrained('/home/jovyan/Training_Model/models/sdg_bert_model')
tokenizer.save_pretrained('/home/jovyan/Training_Model/models/sdg_bert_model')
np.save('/home/jovyan/Training_Model/models/sdg_thresholds.npy', optimal_thresholds)

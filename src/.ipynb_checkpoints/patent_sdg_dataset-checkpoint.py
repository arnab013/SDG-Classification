import pandas as pd
import torch
from torch.utils.data import Dataset

def load_patent_data(path_pattern, text_cols, label_cols, meta_cols):
    import glob
    files = glob.glob(path_pattern)
    dfs = [pd.read_csv(f, usecols=(text_cols + label_cols + meta_cols), compression='gzip') for f in files]
    df = pd.concat(dfs, ignore_index=True)
    for col in label_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    #df[label_cols] = df[label_cols].fillna(0).astype(int)
    return df

def combine_text_for_bert(row, text_cols):
    texts = [str(row[c]) if pd.notnull(row[c]) else "" for c in text_cols]
    return " [SEP] ".join(texts)
import torch
from torch.utils.data import Dataset

class PatentSDGDataset(Dataset):
    def __init__(self, df, tokenizer, text_cols, label_cols, max_length=512, max_chunks=20):
        self.df = df.reset_index(drop=True)  # Avoid pandas warning
        self.tokenizer = tokenizer
        self.text_cols = text_cols
        self.label_cols = label_cols
        self.max_length = max_length
        self.max_chunks = max_chunks

    def __len__(self):
        return len(self.df)

    def _combine_text(self, row):
        # Combine using [SEP] just-in-time
        return " [SEP] ".join([str(row[c]) if pd.notnull(row[c]) else "" for c in self.text_cols])

    def _chunk_text(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        max_chunk_len = self.max_length - 2
        chunks = [tokens[i:i + max_chunk_len] for i in range(0, len(tokens), max_chunk_len)]
        return [self.tokenizer.decode(chunk) for chunk in chunks[:self.max_chunks]]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self._combine_text(row)
        label = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float)
        chunk_texts = self._chunk_text(text)
        encodings = self.tokenizer(chunk_texts, padding='max_length', truncation=True,
                                   max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': label
        }



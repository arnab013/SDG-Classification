#!/usr/bin/env python
"""train_sdg_classifier_v3.py – **with multiple classifier heads**
========================================================
This script fine-tunes BERT with different classifier architectures:
1. Single dense layer (default)
2. Two dense layers
3. Attention-based head (NOAH-style)

Includes functionality to train and compare multiple architectures.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    logging as hf_logging,
)

# Make 🤗 Transformers emit progress bars & info
hf_logging.set_verbosity_info()

###############################################################################
# 1️⃣  CONFIGURATION - DEFINE YOUR PATHS HERE
###############################################################################

class Config:
    # Data paths - Modify these for your dataset
    DATA_DIR = "/home/jovyan/Training_Model/data/"
    CSV_FILENAME = "dataset.csv"  # Updated to match your file
    
    # Model output paths
    MODELS_DIR = "/home/jovyan/Training_Model/models/"
    EXPERIMENT_NAME = "sdg_bert_v1"  # Match your directory structure
    
    # Derived paths (automatically calculated)
    @property
    def csv_path(self):
        return os.path.join(self.DATA_DIR, self.CSV_FILENAME)
    
    @property
    def output_dir(self):
        return os.path.join(self.MODELS_DIR, self.EXPERIMENT_NAME)

# Create global config instance
config = Config()

###############################################################################
# 2️⃣  CLASSIFIER ARCHITECTURES
###############################################################################

class SingleDenseClassifier(PreTrainedModel):
    """BERT with single dense classification layer"""
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained("anferico/bert-for-patents")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config.problem_type = "multi_label_classification"
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class TwoDenseClassifier(PreTrainedModel):
    """BERT with two dense classification layers"""
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained("anferico/bert-for-patents")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.dense2 = nn.Linear(config.hidden_size // 2, config.num_labels)
        self.relu = nn.ReLU()
        self.config.problem_type = "multi_label_classification"
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        hidden = self.relu(self.dense1(pooled_output))
        hidden = self.dropout(hidden)
        logits = self.dense2(hidden)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class AttentionBasedClassifier(PreTrainedModel):
    """BERT with attention-based classification head (NOAH-style)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained("anferico/bert-for-patents")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Attention mechanism for sequence-level representation
        self.attention = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config.problem_type = "multi_label_classification"
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # All token representations
        
        # Compute attention weights for each token
        attention_weights = self.attention(sequence_output)
        
        # Apply attention mask to prevent attending to padding
        if attention_mask is not None:
            attention_weights = attention_weights + (1.0 - attention_mask.unsqueeze(-1)) * -10000.0
        
        # Compute attention-weighted representation
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_output = torch.sum(attention_weights * sequence_output, dim=1)
        
        # Apply dropout and classification
        weighted_output = self.dropout(weighted_output)
        logits = self.classifier(weighted_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

###############################################################################
# 3️⃣  CLI ARGUMENTS
###############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT with different classifier architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Classifier architecture choice
    parser.add_argument("--classifier_type", type=str, default="all",
                        choices=["single", "double", "attention", "all"],
                        help="Classifier architecture: single/double dense or attention-based")

    # Column names
    parser.add_argument("--text_col", type=str, default="text",
                        help="Column that contains the free-text patent")

    # Text processing
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length for BERT input")
    parser.add_argument("--sliding_window", action="store_true",
                        help="Use sliding window for long texts")
    parser.add_argument("--window_stride", type=int, default=128,
                        help="Stride for sliding window")

    # Training hyper-parameters
    parser.add_argument("--num_epochs", type=int, default=4,
                        help="Training epochs")
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Per-GPU batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Validation/test batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Initial learning rate")
                        
    # Disk space saving options
    parser.add_argument("--save_space", action="store_true",
                        help="Minimize disk space usage (overwrite checkpoints)")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps", "no"],
                        help="When to save checkpoints: 'epoch', 'steps', or 'no'")
    parser.add_argument("--final_only", action="store_true",
                        help="Only save the final model (no intermediate checkpoints)")

    # Misc
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Sigmoid cutoff for prediction")
    parser.add_argument("--fp16", action="store_true", help="Mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility")

    return parser.parse_args()

###############################################################################
# 4️⃣  UTILITY FUNCTIONS
###############################################################################

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def build_metrics_fn(threshold: float):
    def compute(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits))
        preds = (probs > threshold).int().numpy()
        labels = labels.astype(int)

        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0)
        subset_acc = accuracy_score(labels, preds)
        
        return {
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "subset_accuracy": subset_acc,
        }
    return compute

###############################################################################
# 5️⃣  DATA PROCESSING (Same as before)
###############################################################################

SDG_LABELS: Dict[int, str] = {
    0: "SDG0 – No SDG / Misc",
    1: "SDG1 – No Poverty",
    2: "SDG2 – Zero Hunger",
    3: "SDG3 – Good Health & Well-Being",
    4: "SDG4 – Quality Education",
    5: "SDG5 – Gender Equality",
    6: "SDG6 – Clean Water & Sanitation",
    7: "SDG7 – Affordable & Clean Energy",
    8: "SDG8 – Decent Work & Economic Growth",
    9: "SDG9 – Industry, Innovation & Infrastructure",
    10: "SDG10 – Reduced Inequalities",
    11: "SDG11 – Sustainable Cities & Communities",
    12: "SDG12 – Responsible Consumption & Production",
    13: "SDG13 – Climate Action",
    14: "SDG14 – Life Below Water",
    15: "SDG15 – Life on Land",
    16: "SDG16 – Peace, Justice & Strong Institutions",
    17: "SDG17 – Partnerships for the Goals",
}
NUM_LABELS = len(SDG_LABELS)

def ensure_sdg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all SDG columns exist, add missing ones with 0."""
    for i in range(NUM_LABELS):
        col = f"sdg_{i}"
        if col not in df.columns:
            print(f"Adding missing column: {col} (filled with 0)")
            df[col] = 0
    return df

def load_and_preprocess(csv_path: str, text_col: str) -> pd.DataFrame:
    """Load and preprocess CSV data."""
    df = pd.read_csv(csv_path)
    
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not in CSV")
    
    # Ensure all SDG columns exist
    df = ensure_sdg_columns(df)
    sdg_cols = [f"sdg_{i}" for i in range(NUM_LABELS)]
    
    # Drop rows with missing text
    df = df.dropna(subset=[text_col])
    
    # Build multi-hot vector
    def row_to_vec(row):
        vec = []
        for col in sdg_cols:
            try:
                val = float(row[col])
                vec.append(1.0 if val > 0.5 else 0.0)
            except:
                vec.append(0.0)
        return np.array(vec, dtype=np.float32).tolist()
    
    df["labels"] = df.apply(row_to_vec, axis=1)
    df = df[[text_col, "labels"]].rename(columns={text_col: "text"})
    
    return df

def tokenize_dataset(ds: DatasetDict, tokenizer, max_len: int = 512):
    """Tokenize dataset for BERT."""
    def tok_fn(batch):
        return tokenizer(
            batch["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_len
        )
    
    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

###############################################################################
# 6️⃣  TRAINING FUNCTION
###############################################################################

def train_single_model(model_class, model_name: str, ds_tok, config_obj, args):
    """Train a single model architecture."""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    # Create model configuration
    model_config = AutoConfig.from_pretrained("anferico/bert-for-patents")
    model_config.num_labels = NUM_LABELS
    model_config.hidden_dropout_prob = 0.1
    
    # Initialize model
    model = model_class(model_config)
    
    # Data collator
    tokenizer = AutoTokenizer.from_pretrained("anferico/bert-for-patents")
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Prepare output subdirectory
    output_subdir = os.path.join(config_obj.output_dir, model_name)
    
    # Clean existing directory if in space-saving mode
    if args.save_space and os.path.exists(output_subdir):
        import shutil
        print(f"Space-saving mode: Cleaning existing directory {output_subdir}")
        # Keep any metrics files but remove model files
        for item in os.listdir(output_subdir):
            item_path = os.path.join(output_subdir, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                print(f"  Removing checkpoint: {item}")
                shutil.rmtree(item_path)
    
    # Create output directory
    Path(output_subdir).mkdir(parents=True, exist_ok=True)
    Path(output_subdir).mkdir(parents=True, exist_ok=True)  # Create directory
    
    # Handle different parameter names across transformers versions
    eval_strategy_param = {}
    try:
        # Try the newer parameter name first
        TrainingArguments(output_dir=".", eval_strategy="epoch")
        eval_strategy_param["eval_strategy"] = "epoch"
    except TypeError:
        # Fall back to older parameter name
        eval_strategy_param["evaluation_strategy"] = "epoch"
    
    # Configure disk space saving options
    if args.save_space:
        print(f"Running with space-saving options enabled")
        save_total_limit = 1
        overwrite_output_dir = True
    else:
        save_total_limit = 2
        overwrite_output_dir = False
    
    # Set save strategy based on args
    if args.final_only:
        save_strategy = "no"
        print(f"Only saving final model (no intermediate checkpoints)")
    else:
        save_strategy = args.save_strategy
        
    # Check if TensorBoard is available
    try:
        import tensorboard
        report_to = ["tensorboard"]
    except ImportError:
        print("Warning: TensorBoard not found, disabling TensorBoard logging")
        report_to = []
    
    training_args = TrainingArguments(
        output_dir=output_subdir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        overwrite_output_dir=overwrite_output_dir,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        weight_decay=0.01,
        fp16=args.fp16,
        logging_steps=10,
        report_to=report_to,  # Use dynamic report_to
        seed=args.seed,
        **eval_strategy_param  # Use the appropriate parameter name
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=data_collator,
        compute_metrics=build_metrics_fn(args.threshold),
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    test_metrics = trainer.evaluate(ds_tok["test"])
    
    # Save results
    Path(output_subdir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_subdir) / "metrics.json", "w") as fp:
        json.dump({"test": test_metrics}, fp, indent=2)
    
    return test_metrics

###############################################################################
# 7️⃣  MAIN PIPELINE WITH COMPARISON
###############################################################################

def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=== CONFIGURATION ===")
    print(f"Input CSV: {config.csv_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Classifier types to train: {args.classifier_type}")
    if args.save_space:
        print(f"Space-saving mode: ON (overwriting checkpoints)")
    if args.final_only:
        print(f"Saving mode: Final model only")
    else:
        print(f"Saving mode: {args.save_strategy}")
    print("====================\n")
    
    # Load data
    print("Loading and preprocessing data...")
    df = load_and_preprocess(config.csv_path, args.text_col)
    print(f"Loaded {len(df)} samples")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("anferico/bert-for-patents")
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=args.seed)
    
    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
    })
    
    # Tokenize
    ds_tok = tokenize_dataset(ds, tokenizer, args.max_length)
    
    # Define models to train
    models_to_train = []
    if args.classifier_type == "all":
        models_to_train = [
            (SingleDenseClassifier, "single_dense"),
            (TwoDenseClassifier, "two_dense"),
            (AttentionBasedClassifier, "attention_based")
        ]
    else:
        model_map = {
            "single": (SingleDenseClassifier, "single_dense"),
            "double": (TwoDenseClassifier, "two_dense"),
            "attention": (AttentionBasedClassifier, "attention_based")
        }
        models_to_train = [model_map[args.classifier_type]]
    
    # Train all selected models
    results = {}
    for model_class, model_name in models_to_train:
        try:
            metrics = train_single_model(model_class, model_name, ds_tok, config, args)
            results[model_name] = metrics["test_macro_f1"]
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = 0.0
    
    # Compare results
    print(f"\n{'='*50}")
    print("FINAL COMPARISON")
    print(f"{'='*50}")
    print(f"{'Model':<20} {'Macro F1':>10}")
    print("-" * 35)
    for model_name, f1_score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:<20} {f1_score:>10.4f}")
    
    # Create output directory if it doesn't exist
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save comparison results
    if results:  # Only save if we have results
        comparison_results = {
            "models": results,
            "best_model": max(results.items(), key=lambda x: x[1])[0] if results else None,
            "best_f1": max(results.values()) if results else 0.0
        }
        
        with open(Path(config.output_dir) / "comparison_results.json", "w") as fp:
            json.dump(comparison_results, fp, indent=2)
    else:
        print("Warning: No results to save")
    
    print(f"\n🏁 Training complete. Results saved to '{config.output_dir}'")

if __name__ == "__main__":
    main()
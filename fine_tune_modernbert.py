# filename: fine_tune_modernbert_optuna.py

# === OPTUNA HYPERPARAMETER TUNING SCRIPT ===
import optuna
import evaluate
import numpy as np
import os
import time
import torch
import pandas as pd
from scipy import stats

from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerFast
)
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score
)

# Disable WandB and other loggers for a clean run
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === Configuration ===
MODEL_DIRECTORY = "/projects/lz25/navyat/nt/model_files_05"
TASK_NAME = "promoter_all"
NUM_TRIALS = 10
TIMEOUT = 3600  # 1 hour timeout
SEEDS = [42, 123, 2024, 3407, 999]  # Seeds for multi-run evaluation

# === 1. Load and preprocess data only ONCE ===
print("Step 1: Loading and preprocessing data...")
full_dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks")

filtered_dataset = full_dataset.filter(lambda example: example['task'] == TASK_NAME)
filtered_dataset = filtered_dataset.remove_columns(["task"])

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIRECTORY)

def tokenize_function(examples):
    return tokenizer(examples['sequence'], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = filtered_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sequence", "name"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
eval_dataset  = tokenized_datasets["test"]

# === Auto-detect binary vs multiclass ===
NUM_LABELS = len(set(train_dataset["labels"].tolist()) | set(eval_dataset["labels"].tolist()))
IS_BINARY  = NUM_LABELS == 2
AVG        = 'binary' if IS_BINARY else 'macro'

print(f"  Number of labels for '{TASK_NAME}': {NUM_LABELS}")
print(f"  Task type: {'Binary' if IS_BINARY else 'Multiclass'}")
print("Data preprocessing complete.")

# === 2. Metrics Function ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Softmax probabilities for AUC
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    try:
        if IS_BINARY:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')

    accuracy  = accuracy_score(labels, predictions)
    f1        = f1_score(labels, predictions, average=AVG)
    precision = precision_score(labels, predictions, average=AVG, zero_division=0)
    recall    = recall_score(labels, predictions, average=AVG, zero_division=0)
    mcc       = matthews_corrcoef(labels, predictions)

    return {
        "accuracy":  accuracy,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
        "matthews_corrcoef": mcc,
        "auc":       auc,
    }

# === 3. Objective Function for Optuna ===
def objective(trial):

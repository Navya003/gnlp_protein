# filename: gena_lm_st.py

import os
import torch
import numpy as np
import evaluate
import time
import optuna
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import pandas as pd
from scipy import stats
from transformers import set_seed

# === Configuration ===
MODEL_NAME = "AIRI-Institute/gena-lm-bert-base-t2t-multi"
TASK_NAME = "promoter_all"
OUTPUT_DIR = "./final_model_gena_lm"
NUM_TRIALS = 10 # Number of Optuna trials
TIMEOUT = 3600 # 1 hour timeout for tuning

# Disable WandB and other loggers for a clean run
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === 1. Load and preprocess data ===
print("Step 1: Loading and preprocessing data...")
full_dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks")

filtered_dataset = full_dataset.filter(lambda example: example['task'] == TASK_NAME)
dataset = filtered_dataset.remove_columns(["task"])

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def tokenize_function(examples):
    # CORRECTED: Added explicit max_length
    return tokenizer(examples['sequence'], padding='max_length', truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
tokenized_eval_dataset = tokenized_eval_dataset.rename_column("label", "labels")

tokenized_train_dataset = tokenized_train_dataset.remove_columns(["sequence"])
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["sequence"])

# === 2. Define metrics ===
print("Step 2: Defining metrics...")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1]
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    mcc = matthews_corrcoef(labels, predictions)
    auc = roc_auc_score(labels, probabilities)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "matthews_corrcoef": mcc,
        "auc": auc,
    }

# === 3. Hyperparameter Tuning with Optuna ===
print("Step 3: Starting hyperparameter tuning with Optuna...")

def model_init(trial):
    num_labels = np.unique(tokenized_train_dataset['labels']).shape[0]
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, trust_remote_code=True)

def objective(trial):
    # Hyperparameter ranges to be tuned
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_categorical("num_train_epochs", [1, 2, 3])

    # Trainer arguments
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}_tuning",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_result = trainer.evaluate()
    
    return eval_result["eval_f1"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=NUM_TRIALS, timeout=TIMEOUT)

print("\n=======================================================")
print("Best hyperparameters found by Optuna:")
best_params = study.best_trial.params
print(best_params)
print("=======================================================")

# === 4. Train the final model across multiple seeds ===
print("\nStep 4: Training the final model across multiple seeds...")
best_model_params = study.best_trial.params
SEEDS = [42, 123, 2024, 3407, 999]
results_list = []

for s in SEEDS:
    print(f"\n--- Training with seed: {s} ---")
    set_seed(s)
    
    final_training_args = TrainingArguments(
        output_dir=f"./final_model/{TASK_NAME}/seed_{s}",
        num_train_epochs=best_model_params["num_train_epochs"],
        per_device_train_batch_size=best_model_params["per_device_train_batch_size"],
        per_device_eval_batch_size=best_model_params["per_device_train_batch_size"],
        learning_rate=best_model_params["learning_rate"],
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=s  # Ensure the trainer respects the seed
    )

    num_labels = np.unique(tokenized_train_dataset['labels']).shape[0]
    final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, trust_remote_code=True)

    final_trainer = Trainer(
        model=final_model,
        args=final_training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics
    )

    final_trainer.train()
    metrics = final_trainer.evaluate()
    
    results_list.append({
        "seed": s,
        "F1": metrics["eval_f1"],
        "Precision": metrics["eval_precision"],
        "Recall": metrics["eval_recall"],
        "MCC": metrics["eval_mcc"],
        "Accuracy": metrics["eval_accuracy"],
        "AUC": metrics["eval_auc"]
    })

# === 5. Save and Statistical Analysis ===
df = pd.DataFrame(results_list)
df.to_csv("genalm_results.csv", index=False)
print("\nResults saved to genalm_results.csv")

metrics_list = ["F1", "Precision", "Recall", "MCC", "Accuracy", "AUC"]
print("\n--- Final Statistical Report ---")
for metric in metrics_list:
    values = df[metric].values
    n = len(values)
    mean = np.mean(values)
    sd = np.std(values, ddof=1)
    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=n-1,
        loc=mean,
        scale=sd / np.sqrt(n)
    )
    print(f"{metric}: {mean:.4f} ± {sd:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

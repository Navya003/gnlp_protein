import os
import torch
import numpy as np
import evaluate
import time
import optuna
import pandas as pd
from scipy import stats
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

# === Configuration ===
MODEL_NAME = "AIRI-Institute/gena-lm-bert-base-t2t-multi"
TASK_NAME = "promoter_all"
OUTPUT_DIR = "./final_model_gena_lm"
NUM_TRIALS = 10  # Number of Optuna trials
TIMEOUT = 3600   # 1 hour timeout for tuning
SEEDS = [42, 123, 2024, 3407, 999]  # Seeds for multi-run evaluation

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
    predictions = np.argmax(logits, axis=-1)

    # Compute softmax probabilities for AUC
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    # Use probability of positive class (index 1) for binary AUC
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float('nan')

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    mcc = matthews_corrcoef(labels, predictions)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "matthews_corrcoef": mcc,
        "auc": auc,
    }

# === 3. Hyperparameter Tuning with Optuna (done once, seed-independent) ===
print("Step 3: Starting hyperparameter tuning with Optuna...")

def model_init(trial=None):
    num_labels = np.unique(tokenized_train_dataset['labels']).shape[0]
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, trust_remote_code=True)

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_categorical("num_train_epochs", [1, 2, 3])

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

# === 4. Multi-seed training and evaluation with best hyperparameters ===
print("\nStep 4: Training with multiple seeds using best hyperparameters...")

all_seed_results = []

for seed in SEEDS:
    print(f"\n--- Running with seed: {seed} ---")

    seed_output_dir = f"./final_model/{TASK_NAME}/seed_{seed}"

    final_training_args = TrainingArguments(
        output_dir=seed_output_dir,
        num_train_epochs=best_params["num_train_epochs"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        per_device_eval_batch_size=best_params["per_device_train_batch_size"],
        learning_rate=best_params["learning_rate"],
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs/seed_{seed}",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=seed,               # <-- Set the training seed
        data_seed=seed,          # <-- Set the data seed for reproducibility
    )

    num_labels = np.unique(tokenized_train_dataset['labels']).shape[0]
    seed_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, trust_remote_code=True
    )

    seed_trainer = Trainer(
        model=seed_model,
        args=final_training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    seed_trainer.train()
    end_time = time.time()

    duration = end_time - start_time
    print(f"Seed {seed} training time: {int(duration // 60)}m {int(duration % 60)}s")

    eval_results = seed_trainer.evaluate()

    seed_row = {
        "seed":     seed,
        "F1":       eval_results.get("eval_f1",               float('nan')),
        "MCC":      eval_results.get("eval_matthews_corrcoef", float('nan')),
        "Accuracy": eval_results.get("eval_accuracy",          float('nan')),
        "AUC":      eval_results.get("eval_auc",               float('nan')),
    }
    all_seed_results.append(seed_row)

    print(f"Seed {seed} results: F1={seed_row['F1']:.4f}, MCC={seed_row['MCC']:.4f}, "
          f"Accuracy={seed_row['Accuracy']:.4f}, AUC={seed_row['AUC']:.4f}")

    # Save the model for this seed
    seed_trainer.save_model(f"{seed_output_dir}/{TASK_NAME}_seed_{seed}_final")

# === 5. Save per-seed results to CSV ===
print("\nStep 5: Saving per-seed results...")
results_csv_path = f"./{OUTPUT_DIR}/{TASK_NAME}_seed_results.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
results_df = pd.DataFrame(all_seed_results)
results_df.to_csv(results_csv_path, index=False)
print(f"Seed results saved to: {results_csv_path}")
print(results_df.to_string(index=False))

# === 6. Compute mean, std, and 95% CI across seeds ===
print("\nStep 6: Computing statistics across seeds...")
metrics = ["F1", "MCC", "Accuracy", "AUC"]

print("\n=======================================================")
print(f"Aggregated results for task: {TASK_NAME}")
print("=======================================================")

for metric in metrics:
    values = results_df[metric].values
    n = len(values)

    mean = np.mean(values)
    sd = np.std(values, ddof=1)

    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=n - 1,
        loc=mean,
        scale=sd / np.sqrt(n)
    )

    print(f"{metric}: {mean:.4f} ± {sd:.4f}")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

print("=======================================================")
print("All done!")

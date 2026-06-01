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
#NUM_LABELS = len(set(train_dataset["labels"].tolist()) | set(eval_dataset["labels"].tolist()))
#NUM_LABELS = len(set(train_dataset["labels"].numpy().tolist()) | set(eval_dataset["labels"].numpy().tolist()))
NUM_LABELS = len(set(int(x) for x in train_dataset["labels"]) | set(int(x) for x in eval_dataset["labels"]))
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
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIRECTORY,
        num_labels=NUM_LABELS
    )

    training_args = TrainingArguments(
        output_dir=f"./optuna_results/trial_{trial.number}",
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        per_device_eval_batch_size=trial.suggest_categorical("per_device_eval_batch_size", [8, 16]),
        num_train_epochs=trial.suggest_int("num_train_epochs", 2, 4),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
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
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_f1"]

# === 4. Run Optuna Study + Multi-seed Evaluation ===
def run_hyperparameter_tuning_and_evaluate():

    # --- Optuna tuning ---
    print("Step 2: Starting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction="maximize", study_name=TASK_NAME)
    study.optimize(objective, n_trials=NUM_TRIALS, timeout=TIMEOUT)

    print("\n=======================================================")
    print(f"Hyperparameter tuning for {TASK_NAME} complete.")
    print(f"  Best value: {study.best_trial.value:.4f}")
    print(f"  Best params: {study.best_trial.params}")
    print("=======================================================")

    best_params = study.best_trial.params

    # --- Multi-seed training ---
    print("\nStep 3: Training with multiple seeds using best hyperparameters...")
    all_seed_results = []

    for seed in SEEDS:
        print(f"\n--- Running with seed: {seed} ---")

        seed_output_dir = f"./final_model/{TASK_NAME}/seed_{seed}"

        final_training_args = TrainingArguments(
            output_dir=seed_output_dir,
            num_train_epochs=best_params["num_train_epochs"],
            per_device_train_batch_size=best_params["per_device_train_batch_size"],
            per_device_eval_batch_size=best_params["per_device_eval_batch_size"],
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
            seed=seed,       # training seed
            data_seed=seed,  # data shuffling seed
        )

        seed_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIRECTORY,
            num_labels=NUM_LABELS
        )

        seed_trainer = Trainer(
            model=seed_model,
            args=final_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
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

        seed_trainer.save_model(f"{seed_output_dir}/{TASK_NAME}_seed_{seed}_final")

    # --- Save CSV ---
    print("\nStep 4: Saving per-seed results...")
    os.makedirs("./modernbert_results", exist_ok=True)
    results_csv_path = f"./modernbert_results/{TASK_NAME}_seed_results.csv"
    results_df = pd.DataFrame(all_seed_results)
    results_df.to_csv(results_csv_path, index=False)
    print(f"Seed results saved to: {results_csv_path}")
    print(results_df.to_string(index=False))

    # --- Compute statistics ---
    print("\nStep 5: Computing statistics across seeds...")
    metrics = ["F1", "MCC", "Accuracy", "AUC"]

    print("\n=======================================================")
    print(f"Aggregated results for task: {TASK_NAME} (ModernBERT)")
    print("=======================================================")

    for metric in metrics:
        values = results_df[metric].values
        n      = len(values)
        mean   = np.mean(values)
        sd     = np.std(values, ddof=1)

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

# === Execute ===
if __name__ == "__main__":
    run_hyperparameter_tuning_and_evaluate()

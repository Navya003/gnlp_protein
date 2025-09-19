# filename: gena_lm_fine_tuning_script.py

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
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from tqdm import tqdm

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
    return tokenizer(examples['sequence'], padding='max_length', truncation=True)

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
        eval_strategy="epoch", # CORRECTED: Changed from 'evaluation_strategy'
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

# === 4. Train the final model with the best parameters ===
print("\nStep 4: Training the final model with best hyperparameters...")
best_model_params = study.best_trial.params

final_training_args = TrainingArguments(
    output_dir=f"./final_model/{TASK_NAME}",
    num_train_epochs=best_model_params["num_train_epochs"],
    per_device_train_batch_size=best_model_params["per_device_train_batch_size"],
    per_device_eval_batch_size=best_model_params["per_device_train_batch_size"],
    learning_rate=best_model_params["learning_rate"],
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch", # CORRECTED: Changed from 'evaluation_strategy'
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

# Load the model and tokenizer again for final training
num_labels = np.unique(tokenized_train_dataset['labels']).shape[0]
final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, trust_remote_code=True)

final_trainer = Trainer(
    model=final_model,
    args=final_training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics
)

# Time the final training process
start_time = time.time()
final_trainer.train()
end_time = time.time()

training_duration = end_time - start_time
minutes = int(training_duration // 60)
seconds = int(training_duration % 60)

print("\n=======================================================")
print(f"Final training complete. Time taken: {minutes} minutes and {seconds} seconds.")
print("=======================================================")

# Final evaluation on the best model
final_evaluation_results = final_trainer.evaluate()

print("\n=======================================================")
print(f"Final evaluation results for {TASK_NAME}:")
print(final_evaluation_results)
print("=======================================================")

# Save the final fine-tuned model
final_trainer.save_model(f"./{OUTPUT_DIR}/{TASK_NAME}_final")

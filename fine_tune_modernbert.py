# filename: fine_tune_modernbert_optuna.py

# === OPTUNA HYPERPARAMETER TUNING SCRIPT ===
import optuna
import evaluate
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerFast
)
from datasets import load_dataset
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
import numpy as np
import os

# Disable WandB and other loggers for a clean run
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === Configuration ===
MODEL_DIRECTORY = "/projects/lz25/navyat/nt/model_files_05"
TASK_NAME = "promoter_all" 
NUM_TRIALS = 10 
TIMEOUT = 3600 # 1 hour timeout

# === 1. Load and preprocess data only ONCE ===
print("Step 1: Loading and preprocessing data...")
full_dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks")

# Filter for the specific task
filtered_dataset = full_dataset.filter(lambda example: example['task'] == TASK_NAME)
filtered_dataset = filtered_dataset.remove_columns(["task"])

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIRECTORY)

# Preprocessing function
def tokenize_function(examples):
    return tokenizer(examples['sequence'], padding="max_length", truncation=True, max_length=512)

# Tokenize and format the datasets
tokenized_datasets = filtered_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sequence", "name"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# Get number of labels
num_labels = max(train_dataset["labels"]) + 1
print(f"  Number of labels for '{TASK_NAME}': {num_labels}")
print("Data preprocessing complete.")

# === 2. Metrics Function ===
# The Trainer will call this function during evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate all metrics and return them in a dictionary
    metrics = {}
    metrics["accuracy"] = evaluate.load("accuracy").compute(predictions=predictions, references=labels)["accuracy"]
    metrics["f1"] = f1_score(labels, predictions, average="weighted")
    metrics["mcc"] = matthews_corrcoef(labels, predictions)
    metrics["precision"] = precision_score(labels, predictions, average="weighted", zero_division=0)
    metrics["recall"] = recall_score(labels, predictions, average="weighted", zero_division=0)
    
    return metrics

# === 3. Objective Function for Optuna ===
def objective(trial):
    # Load a fresh model for each trial
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIRECTORY,
        num_labels=num_labels
    )
    
    # Suggest hyperparameters to the trial
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
        # Metric to maximize for hyperparameter search
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
    
    # Return the metric to be maximized by Optuna
    return eval_result["eval_f1"]

# === 4. Run Optuna Study and Final Evaluation ===
def run_hyperparameter_tuning_and_evaluate():
    print("Step 2: Starting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction="maximize", study_name=TASK_NAME)
    study.optimize(objective, n_trials=NUM_TRIALS, timeout=TIMEOUT)

    print("\n=======================================================")
    print(f"Hyperparameter tuning for {TASK_NAME} complete.")
    print("Best trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print("  Params: ", study.best_trial.params)
    print("=======================================================")

    # Get the best hyperparameters
    best_params = study.best_trial.params

    # Run one final training session with the best parameters
    print("\nStep 3: Training and evaluating the final model with best hyperparameters...")
    
    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIRECTORY, 
        num_labels=num_labels
    )
    final_training_args = TrainingArguments(
        output_dir=f"./final_model/{TASK_NAME}",
        num_train_epochs=best_params["num_train_epochs"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        per_device_eval_batch_size=best_params["per_device_eval_batch_size"],
        learning_rate=best_params["learning_rate"],
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
    final_trainer = Trainer(
        model=final_model,
        args=final_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    final_trainer.train()
    final_evaluation_results = final_trainer.evaluate()
    
    print("\n=======================================================")
    print(f"Final evaluation results for {TASK_NAME}:")
    print(final_evaluation_results)
    print("=======================================================")

    # Save the final model
    final_trainer.save_model(f"./{TASK_NAME}_final_model")
    print(f"Final model saved to ./{TASK_NAME}_final_model")

# === Execute ===
if __name__ == "__main__":
    run_hyperparameter_tuning_and_evaluate()

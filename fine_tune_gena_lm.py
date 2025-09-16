# filename: fine_tune_gena_lm.py

# === OPTUNA HYPERPARAMETER TUNING SCRIPT ===
import optuna
import evaluate
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer, 
)
from datasets import load_dataset
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
import numpy as np
import os

# Disable WandB and other loggers for a clean run
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HOME"] = "/projects/lz25/navyat/"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === Configuration ===
# Use the model identifier for GENA-LM
MODEL_NAME = "AIRI-Institute/gena-lm-bert-base-t2t-dna" 
TASK_NAME = "splice_sites_acceptor" # Example task, update as needed
NUM_TRIALS = 10 
TIMEOUT = 3600 # 1 hour timeout

# === 1. Load and preprocess data only ONCE ===
print("Step 1: Loading and preprocessing data...")
full_dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks")

# Filter for the specific task
filtered_dataset = full_dataset.filter(lambda example: example['task'] == TASK_NAME)
filtered_dataset = filtered_dataset.remove_columns(["task"])

# Add a check to prevent the ValueError
if len(filtered_dataset['train']) == 0:
    print(f"\nError: The dataset for TASK_NAME '{TASK_NAME}' is empty.")
    print("Please check the task name and try again.")
    exit()

# Split the data
train_dataset = filtered_dataset["train"].shuffle(seed=42)
eval_dataset = filtered_dataset["validation"].shuffle(seed=42)

# Get the number of labels for the task
num_labels = max(train_dataset["labels"]) + 1

# Load GENA-LM tokenizer with trust_remote_code=True
print("Step 2: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Tokenization function
def tokenize_function(examples):
    # Ensure correct padding and truncation
    return tokenizer(
        examples["sequence"], 
        padding="max_length", 
        truncation=True, 
        max_length=600 # Adjust max_length according to your task's sequence length
    )

# Apply tokenization
print("Step 3: Tokenizing dataset...")
tokenized_datasets = filtered_dataset.map(tokenize_function, batched=True)

# Set the format for PyTorch and remove unwanted columns
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_train_dataset = tokenized_datasets["train"]
tokenized_eval_dataset = tokenized_datasets["validation"]

# === 2. Hyperparameter Search with Optuna ===
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # The predictions are logits, so we take the argmax
    predictions = np.argmax(predictions, axis=1)
    
    f1 = f1_score(labels, predictions, average="weighted")
    mcc = matthews_corrcoef(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    
    return {
        "f1": f1,
        "matthews_corrcoef": mcc,
        "precision": precision,
        "recall": recall
    }

def model_init(trial):
    config = AutoConfig.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        trust_remote_code=True
    )
    # Load GENA-LM model with trust_remote_code=True
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        config=config,
        trust_remote_code=True
    )
    return model

def objective(trial):
    # Suggest hyperparameters to Optuna
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size", [8, 16]
    )
    per_device_eval_batch_size = trial.suggest_categorical(
        "per_device_eval_batch_size", [8, 16]
    )
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 3)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
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
        tokenizer=tokenizer,
    )

    trainer.train()
    
    # Evaluate on the best model from the current trial
    eval_results = trainer.evaluate()
    return eval_results["eval_f1"]

# Run Optuna study
print("Step 4: Starting Optuna hyperparameter search...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=NUM_TRIALS, timeout=TIMEOUT)

print("\n=======================================================")
print("Best trial:")
best_trial = study.best_trial
print(f"  Value (F1 Score): {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
print("=======================================================\n")

# === 3. Final Training with Best Hyperparameters ===
if best_trial:
    best_params = best_trial.params
    print("Step 5: Training final model with best hyperparameters...")
    
    config = AutoConfig.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        trust_remote_code=True
    )
    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        config=config,
        trust_remote_code=True
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
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics
    )
    
    final_trainer.train()
    final_evaluation_results = final_trainer.evaluate()
    
    print("\n=======================================================")
    print(f"Final evaluation results for {TASK_NAME}:")
    print(final_evaluation_results)
    print("=======================================================")

    final_trainer.save_model(f"./{TASK_NAME}_final_model")

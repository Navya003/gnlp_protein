# filename: fine_tune_modernbert_rna.py

# === OPTUNA HYPERPARAMETER TUNING SCRIPT FOR MODERNBERT (RNA) ===
import optuna
import evaluate
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoConfig,
    PreTrainedTokenizerFast # Correct tokenizer for ModernBERT
)
from datasets import load_dataset
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
import numpy as np
import os
import time

# Disable WandB and standardizing HF_HOME path
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HOME"] = "/projects/lz25/navyat/" # <-- CORRECTED HF_HOME PATH
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === Configuration ===
MODEL_DIRECTORY = "/projects/lz25/navyat/nt/model_files_05" # <-- CORRECTED MODEL DIRECTORY
TASK_NAME = "ncrna_family_bnoise0" 
NUM_TRIALS = 10 
TIMEOUT = 3600 # 1 hour timeout
MAX_LENGTH = 512

# === 1. Load and preprocess data only ONCE ===
print("Step 1: Loading and preprocessing data...")
# Loading the RNA dataset
full_dataset = load_dataset("genbio-ai/rna-downstream-tasks", TASK_NAME)

# The RNA dataset task is loaded directly via the config
filtered_dataset = full_dataset 
print(filtered_dataset)

# Load ModernBERT tokenizer from the local directory
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIRECTORY) # <-- Uses MODEL_DIRECTORY
if not tokenizer:
    raise ValueError("Tokenizer failed to load. Please check the model path.")
    
# Function to tokenize the sequences
def tokenize_function(examples):
    # This uses the 'sequences' column, correct for the RNA dataset
    return tokenizer(
        examples['sequences'], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH
    )

print("Step 2: Tokenizing the dataset...")
# Apply tokenization to the entire dataset
tokenized_dataset = filtered_dataset.map(tokenize_function, batched=True, num_proc=10, remove_columns=["sequences", "family"])
tokenized_dataset.set_format("torch")

# Split the dataset into train and test
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

# Set num_labels dynamically based on the dataset
num_labels = len(np.unique(train_dataset['labels']))
print(f"Number of labels detected: {num_labels}")

# --- 2. Define compute_metrics for evaluation ---
def compute_metrics(eval_pred):
    # Access the prediction logits and true labels
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    predictions = np.argmax(predictions, axis=1)
    
    # Compute and return the metrics (macro average for multi-class classification)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    mcc = matthews_corrcoef(labels, predictions)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        "f1": f1,
        "mcc": mcc,
        "precision": precision,
        "recall": recall
    }

# --- 3. Define the objective function for Optuna ---
def objective(trial):
    # Suggest hyperparameters (re-introduced batch size for Optuna search)
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)

    # Load a fresh ModernBERT model for each trial
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIRECTORY, # <-- Uses MODEL_DIRECTORY
        num_labels=num_labels # Pass num_labels directly
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{TASK_NAME}/trial_{trial.number}",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size, 
        per_device_eval_batch_size=per_device_train_batch_size * 2,
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

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()

    # Optuna needs to maximize a value
    return eval_results["eval_f1"]

# --- 4. Run the Optuna study and Final Evaluation ---
if __name__ == "__main__":
    print("Step 3: Running Optuna hyperparameter optimization...")
    study_name = f"{TASK_NAME}_optimization_modernbert"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=NUM_TRIALS, timeout=TIMEOUT)

    # --- 5. Print best hyperparameters and train the final model ---
    print("\n=======================================================")
    print("Optimization finished.")
    print("Best hyperparameters found: ", study.best_params)
    print("Best F1 score: ", study.best_value)
    print("=======================================================")

    best_params = study.best_params
    
    # Train the final model with the best hyperparameters
    print("\nStep 4: Training final model with best hyperparameters...")
    
    # Load the final ModernBERT model
    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIRECTORY, # <-- Uses MODEL_DIRECTORY
        num_labels=num_labels
    )
    
    # Use the best hyperparameters found by Optuna
    final_training_args = TrainingArguments(
        output_dir=f"./final_model/{TASK_NAME}",
        num_train_epochs=best_params["num_train_epochs"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"], 
        per_device_eval_batch_size=best_params["per_device_train_batch_size"] * 2, 
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
    
    # Measure the training time
    start_time = time.time()
    final_trainer.train()
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    final_evaluation_results = final_trainer.evaluate()

    print(f"\nExecution time: {elapsed_time:.4f} seconds")
    print("\n=======================================================")
    print(f"Final evaluation results for {TASK_NAME}:")
    print(final_evaluation_results)
    print("=======================================================")

    final_trainer.save_model(f"./{TASK_NAME}_final_model")

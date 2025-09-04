# filename: fine_tune_dnabert2.py

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
MODEL_NAME = "quietflamingo/dnabert2-no-flashattention"
TASK_NAME = "promoter_all" 
NUM_TRIALS = 10 
TIMEOUT = 3600 # 1 hour timeout

# === 1. Load and preprocess data only ONCE ===
print("Step 1: Loading and preprocessing data...")
full_dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks")

# Filter for the specific task
filtered_dataset = full_dataset.filter(lambda example: example['task'] == TASK_NAME)
filtered_dataset = filtered_dataset.remove_columns(["task"])

# Load DNABERT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if not tokenizer:
    raise ValueError("Tokenizer failed to load. Please check the model name and internet connection.")
    
# Function to tokenize the sequences
def tokenize_function(examples):
    # CORRECTED: Use 'sequence' column instead of 'text'
    if 'sequence' not in examples or not isinstance(examples['sequence'], list):
        raise ValueError("Dataset does not contain a 'sequence' column or it's not a list.")
    # Pad sequences to the maximum length of the batch to speed up processing
    return tokenizer(examples['sequence'], padding="max_length", truncation=True, max_length=512)

print("Step 2: Tokenizing the dataset...")
# Apply tokenization to the entire dataset
# CORRECTED: Remove 'sequence' column after tokenization
tokenized_dataset = filtered_dataset.map(tokenize_function, batched=True, num_proc=10, remove_columns=["sequence"])
tokenized_dataset.set_format("torch")

# Split the dataset into train and test
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]
# Set num_labels dynamically based on the dataset
num_labels = len(np.unique(train_dataset['label']))
print(f"Number of labels detected: {num_labels}")

# --- 2. Define compute_metrics for evaluation ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Check if predictions are logits (for classification)
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
        
    f1 = f1_score(labels, predictions, average='macro')
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
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32])
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)

    # Load a fresh model for each trial
    # 1. Load the model's configuration
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.num_labels = num_labels
    # 2. Explicitly set the attention implementation to 'eager'
    config._attn_implementation = "eager"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        config=config,
        trust_remote_code=True
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

    # Optuna needs to maximize a value, so we'll use F1-score
    return eval_results["eval_f1"]

# --- 4. Run the Optuna study ---
if __name__ == "__main__":
    print("Step 3: Running Optuna hyperparameter optimization...")
    study_name = f"{TASK_NAME}_optimization"
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
    
    # Load the final model with the best configuration from the study
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.num_labels = num_labels
    config._attn_implementation = "eager"

    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        config=config,
        trust_remote_code=True
    )
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
    
    final_trainer.train()
    final_evaluation_results = final_trainer.evaluate()
    
    print("\n=======================================================")
    print(f"Final evaluation results for {TASK_NAME}:")
    print(final_evaluation_results)
    print("=======================================================")

    final_trainer.save_model(f"./{TASK_NAME}_final_model")

# filename: fine_tune_modernbert.py

# --- Required imports
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate
import os

# --- 1. Define model and tokenizer paths ---
model_directory = "/projects/lz25/navyat/nt/model_files_05"

# --- 2. Define the list of tasks to fine-tune on ---
# Choose any 5 tasks from the 18 available
tasks_to_finetune = [
    "promoter_tata",
    "H3",
    "H4",
    "enhancers",
    "enhancer_types"
]

# --- 3. Load the full dataset (default configuration)
print("Step 1: Loading the full 'nucleotide_transformer_downstream_tasks' dataset...")
try:
    full_dataset = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks"
    )
    print("Full dataset loaded successfully.")
except Exception as e:
    print(f"Error loading the dataset: {e}")
    exit()

# --- 4. Load the pre-trained tokenizer and model ---
print("\nStep 2: Loading pre-trained tokenizer and model...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_directory)
# We load the model once, and it will be fine-tuned for each task.
# The number of labels will be updated for each task.
base_model = AutoModelForSequenceClassification.from_pretrained(model_directory)

# --- 5. Define evaluation metrics
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# --- 6. Iterate through each task and run the fine-tuning process ---
for task_name in tasks_to_finetune:
    print(f"\n=======================================================")
    print(f"Step 3: Fine-tuning for task: {task_name}")
    print(f"=======================================================")

    # Filter the dataset for the current task
    # A single dataset contains all the tasks, differentiated by the 'task' column.
    filtered_dataset = full_dataset.filter(lambda example: example['task'] == task_name)
    
    # We need to drop the 'task' column before training
    filtered_dataset = filtered_dataset.remove_columns(["task"])

    # Update the number of labels for the current task
    num_labels = filtered_dataset["train"].features["labels"].num_classes
    
    # Reload the model with the correct number of labels for the current task.
    # We use a fresh model to ensure each fine-tuning process is independent.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_directory, 
        num_labels=num_labels
    )

    # Preprocess the data for the current task
    def tokenize_function(examples):
        return tokenizer(examples['sequence'], padding="max_length", truncation=True)

    print("  Tokenizing the dataset...")
    tokenized_datasets = filtered_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sequence", "id"])
    tokenized_datasets = tokenized_datasets.rename_column("labels", "labels")
    tokenized_datasets.set_format("torch")

    # Set up TrainingArguments for the current task
    print("  Setting up training arguments...")
    output_dir = f"./results/{task_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs/{task_name}',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Instantiate and run the Trainer for the current task
    print("  Initializing and starting the Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("  Fine-tuning complete!")

    # Evaluate the fine-tuned model
    print("  Evaluating the model on the test set...")
    evaluation_results = trainer.evaluate()
    print("  Evaluation results:", evaluation_results)
    
    # You may also want to save the final fine-tuned model for this task
    trainer.save_model(f"./{task_name}_final_model")

print("\nAll tasks completed!")

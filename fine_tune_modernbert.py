# filename: fine_tune_modernbert.py

# --- Required imports
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate

# --- 1. Define model and tokenizer paths ---
# Set the path to the directory that contains all the model and tokenizer files.
model_directory = "./path/to/your/modernbert-dna-model/"

# --- 2. Load the dataset for fine-tuning

print("Step 1: Loading the fine-tuning dataset...")

# Promoter/Non-Promoter dataset
dataset = load_dataset(
    "InstaDeepAI/nucleotide_transformer_downstream_tasks", 
    split="promoter_all", streaming=True
)
# dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks", split="splice_sites_donor")
# dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks", split="H3K4me3")

print(f"Dataset loaded successfully with splits: {dataset.keys()}")
print(dataset['train'])

# --- 2. Load the pre-trained tokenizer and model ---
print("\nStep 2: Loading pre-trained tokenizer and model...")
# The from_pretrained method will find all the necessary files within this directory.
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_directory)

# Ensure the model is loaded with the correct number of labels for your task.
num_labels = dataset["train"].features["labels"].num_classes
model = AutoModelForSequenceClassification.from_pretrained(
    model_directory, 
    num_labels=num_labels
)

# --- 4. Define data preprocessing function
# This function tokenizes the sequences and prepares them for the model.
def tokenize_function(examples):
    # 'sequence' is the column name for the DNA sequences
    return tokenizer(examples['sequence'], padding="max_length", truncation=True)

print("\nStep 3: Tokenizing the dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove the original columns that are not needed for training
tokenized_datasets = tokenized_datasets.remove_columns(["sequence", "id"])
# Rename the 'labels' column to 'labels' as required by the Trainer
tokenized_datasets = tokenized_datasets.rename_column("labels", "labels")

# Set the format for PyTorch
tokenized_datasets.set_format("torch")

# --- 5. Define evaluation metrics
# We will use accuracy as a simple metric.
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# --- 6. Set up TrainingArguments
print("\nStep 4: Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",                   # Directory where the model and logs will be saved
    num_train_epochs=3,                       # Number of full passes over the training data
    per_device_train_batch_size=8,            # Batch size per GPU/device for training
    per_device_eval_batch_size=8,             # Batch size per GPU/device for evaluation
    warmup_steps=500,                         # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                        # Strength of weight decay
    logging_dir='./logs',                     # Directory for storing logs
    logging_steps=100,                        # Log every 100 steps
    eval_strategy="epoch",                    # Run evaluation at the end of each epoch
    save_strategy="epoch",                    # Save model checkpoint at the end of each epoch
    load_best_model_at_end=True,              # Load the best model from a saved checkpoint
    metric_for_best_model="accuracy",         # Metric to monitor for the best model
)

# --- 7. Instantiate the Trainer
print("\nStep 5: Initializing the Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# --- 8. Start fine-tuning
print("\nStep 6: Starting the fine-tuning process...")
trainer.train()
print("\nFine-tuning complete!")

# --- 9. Evaluate the fine-tuned model
print("\nStep 7: Evaluating the model on the test set...")
evaluation_results = trainer.evaluate()
print("Evaluation results:", evaluation_results)

# --- End of script ---

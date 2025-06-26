# finetune_matscibert.py

# --- Imports ---
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from huggingface_hub import HfFolder
import numpy as np
from sklearn.metrics import f1_score
import os

# --- Configuration ---
# Dataset ID from huggingface.co/datasets
# This remains the same as it's the synthetic domain classification dataset.
dataset_id = "argilla/synthetic-domain-text-classification"

# Model ID for MatSciBERT (or a general SciBERT for materials science context)
# You might need to replace this with a more specific MatSciBERT if available.
# Example using a general SciBERT:
model_id = "allenai/scibert_scivocab_uncased" # A common SciBERT model

# Output directory for the fine-tuned model
output_dir = "MatSciBERT-domain-classifier"

# --- Data Loading and Preparation ---
print(f"Loading dataset: {dataset_id}")
train_dataset = load_dataset(dataset_id, split='train')

print("Splitting dataset into train and test sets...")
# Using a reduced subset for faster iteration, as discussed.
# Adjust these numbers based on your available resources and desired test run length.
train_subset_size = 1000 # Example: Take 1000 training examples
test_subset_size = 100   # Example: Take 100 test examples

print(f"Original training set size: {len(train_dataset)}")

# Reduce training dataset size
train_dataset = train_dataset.select(range(min(train_subset_size, len(train_dataset))))
print(f"Reduced training set to {len(train_dataset)} examples.")

# Perform train-test split on the potentially reduced dataset
split_dataset = train_dataset.train_test_split(test_size=min(test_subset_size, len(train_dataset)) / len(train_dataset) if len(train_dataset) > 0 else 0.1, seed=42)

# Ensure the test split is not empty if the original subset was very small
if len(split_dataset['test']) == 0 and len(split_dataset['train']) > 0:
    # If test_subset_size was larger than what's left after train_subset_size,
    # or if train_subset_size itself was too small, this ensures a minimal test set.
    # Re-split with a fixed test_size proportion or handle no test set for very small datasets.
    print("Warning: Test set is empty after subsetting. Re-splitting with 10% test size if possible.")
    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    if len(split_dataset['test']) == 0:
        print("Error: Still no test set after re-split. Please increase train_subset_size.")
        # You might want to exit or handle this more gracefully.
        exit()


print("Example data point from training set:")
print(split_dataset['train'][0])

print(f"Loading tokenizer for model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Tokenize helper function
def tokenize(batch):
    # SciBERT typically uses a 512 token context length like BERT,
    # so truncation=True is important.
    return tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt")

# Tokenize dataset
print("Tokenizing dataset...")
if "label" in split_dataset["train"].features.keys():
    split_dataset = split_dataset.rename_column("label", "labels") # to match Trainer
tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=["text"])

print("Tokenized dataset features:")
print(tokenized_dataset["train"].features.keys())

# --- Model Preparation ---
print(f"Loading model: {model_id}")
# Prepare model labels - useful for inference
labels = tokenized_dataset["train"].features["labels"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# Download the model from huggingface.co/models
# Note: MatSciBERT/SciBERT are BERT-based, so AutoModelForSequenceClassification is correct.
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label,
)

# --- Metric Definition ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )
    return {"f1": float(score) if score == 1 else score}

# --- Training Arguments and Trainer ---
print("Defining training arguments...")

# Determine if CUDA is available for GPU training
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    print("CUDA is available. Training will use GPU.")
else:
    print("CUDA is not available. Training will use CPU.")

# Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=5,
    # bf16 is typically supported by newer NVIDIA GPUs (Ampere architecture and above).
    # If you have an older GPU or are on CPU, set to False or fp16=True if your GPU supports it.
    bf16=use_cuda,
    optim="adamw_torch_fused", # improved optimizer 
    # logging & evaluation strategies
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    no_cuda=not use_cuda, # Set to True to force CPU, False to allow GPU if available
    metric_for_best_model="f1",
    # push to hub parameters
    push_to_hub=True,
    hub_strategy="every_save",
    hub_token=HfFolder.get_token(),
)

# Create a Trainer instance
print("Creating Trainer instance...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# --- Train the Model ---
print("Starting model training...")
trainer.train()

# --- Run Inference ---
print("\n--- Running Inference ---")
from transformers import pipeline

# Load model from huggingface.co/models using our repository id
# device=0 for first GPU, -1 for CPU
inference_device = 0 if use_cuda else -1 
classifier = pipeline(
    task="text-classification", 
    model=os.path.join(output_dir, "checkpoint-final") if os.path.exists(os.path.join(output_dir, "checkpoint-final")) else output_dir, # Load from final checkpoint or output_dir
    device=inference_device,
)
 
sample = "This material exhibits superconductivity at extremely low temperatures due to its unique crystalline structure and electron phonon coupling."
 
print(f"Sample text: '{sample}'")
print(classifier(sample))

print("\n--- Another Sample ---")
sample_2 = "The latest research indicates a novel approach to synthesize quantum dots for improved solar cell efficiency."
print(f"Sample text: '{sample_2}'")
print(classifier(sample_2))

print("\nConclusion: Model fine-tuning and inference complete.")

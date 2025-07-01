#!/usr/bin/env python3

import os
import zipfile
import torch
import datasets
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import (
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import evaluate # For evaluation (e.g., perplexity)

# Import ModernBert specific classes.
# These *must* be available in your Python environment, likely from the transformers
# library version you installed or a custom local package.
try:
    from transformers import ModernBertConfig, ModernBertForMaskedLM
except ImportError:
    print("WARNING: ModernBertConfig or ModernBertForMaskedLM not found directly in transformers.")
    print("Please ensure your 'transformers' installation from GitHub includes these classes,")
    print("or that you have a custom 'modern_bert' package installed if they are external.")
    # Exit or raise error if these are critical and not found
    # For a robust script, you might want to sys.exit(1) here
    exit("Required ModernBERT classes not found. Exiting.")


# --- Paths and Configuration ---
# IMPORTANT: Adjust these paths for your HPC environment.
# If your dataset is a zip, uncomment the unzipping block and adjust dataset_zip_path.
# If 'genome_subset_1000' is already an extracted directory, keep it as is.
# Example if genome_subset_1000.zip exists (adjust source path):
# dataset_zip_path = "/path/to/your/data/genome_subset_1000.zip"
extracted_dataset_dir = "/home/navyat/nt/genome_subset_1000" # Where the dataset will be loaded from
output_dir = "/home/navyat/nt/modernbert-pretrain_out" # Output directory for the pre-trained model

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Pre-training specific configurations
tokenizer_name = "answerdotai/ModernBERT-base" # Tokenizer for pre-training
mlm_probability = 0.3
max_sequence_length = 512 # Chunk size for pre-training

learning_rate = 5e-5 # Common learning rate for pre-training
batch_size = 8 # Batch size might need to be smaller for pre-training due to sequence length
num_epochs = 3 # Or more, pre-training usually requires many epochs

# --- Unzip Dataset (If your dataset is in a zip file) ---
# Uncomment and modify this block if your 'genome_subset_1000' dataset
# is provided as a zip file that needs extraction.
# Make sure the zip file path is correct for your HPC.
# try:
#     print(f"Unzipping dataset from {dataset_zip_path} to {extracted_dataset_dir}...")
#     with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
#         zip_ref.extractall("/path/to/your/data/") # Adjust extraction root if needed
#     print("Dataset unzipped.")
# except FileNotFoundError:
#     print(f"WARNING: Dataset zip file not found at {dataset_zip_path}. Assuming dataset is already extracted at {extracted_dataset_dir}.")
# except Exception as e:
#     print(f"Error unzipping dataset: {e}. Please check path and permissions.")
#     exit(1)


# --- Load Dataset ---
print(f"Loading dataset from: {extracted_dataset_dir}")
try:
    raw_dataset = load_from_disk(extracted_dataset_dir)
except Exception as e:
    print(f"Error loading dataset from {extracted_dataset_dir}: {e}")
    print("Please ensure the path is correct and the dataset is properly formatted.")
    exit(1)

# Check if the loaded object is a DatasetDict or a Dataset
if isinstance(raw_dataset, DatasetDict):
    if "train" not in raw_dataset:
        raise ValueError("DatasetDict must contain a 'train' split for pre-training.")
    if "test" not in raw_dataset and "validation" not in raw_dataset:
        print("No 'test' or 'validation' split found. Creating a 90/10 train/test split for evaluation.")
        train_test_split = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
        split_datasets = DatasetDict({
            "train": train_test_split["train"],
            "test": train_test_split["test"]
        })
    else:
        split_datasets = raw_dataset # Use the existing DatasetDict
elif isinstance(raw_dataset, Dataset):
    print("Loaded dataset is a single Dataset. Creating a 90/10 train/test split.")
    train_test_split = raw_dataset.train_test_split(test_size=0.1, seed=42)
    split_datasets = datasets.DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })
else:
    raise TypeError(f"Unexpected dataset type: {type(raw_dataset)}. Expected Dataset or DatasetDict.")


print("Dataset structure for pre-training:")
print(split_datasets)
print(f"Train dataset size: {len(split_datasets['train'])}")
print(f"Test dataset size: {len(split_datasets['test'])}")

# --- Tokenizer Loading ---
print(f"Loading tokenizer: {tokenizer_name}")
hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)

# Identify the text/sequence column name from the dataset's features
# Assuming it's 'text' based on your previous logs for 'genome_subset_1000'
# Verify this by inspecting split_datasets['train'].column_names if uncertain
sequence_column_name = "text"

def tokenize_function(examples):
    return hf_tokenizer(examples[sequence_column_name], return_attention_mask=False)

# Apply tokenization to the dataset
print("Tokenizing dataset...")
# Use num_proc for parallel processing, adjust based on available cores/resources
num_workers = os.cpu_count() if os.cpu_count() else 1 # Default to 1 if cpu_count is None
tokenized_dataset = split_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=num_workers,
    remove_columns=split_datasets["train"].column_names,
    load_from_cache_file=False # Always re-process if underlying data might change
)
print("Dataset tokenization complete.")

# --- Grouping texts into fixed-size chunks ---
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_sequence_length) * max_sequence_length
    result = {
        k: [t[i : i + max_sequence_length] for i in range(0, total_length, max_sequence_length)]
        for k, t in concatenated_examples.items()
    }
    # Ensure 'labels' are present and are 'input_ids' for MLM
    result["labels"] = result["input_ids"].copy()
    return result

print(f"Grouping texts into chunks of {max_sequence_length}...")
lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=num_workers,
    load_from_cache_file=False, # Important, as this changes sample count
)
print("Text grouping complete.")

print("Pre-training dataset structure after chunking:")
print(lm_dataset)
print(f"Train dataset size after chunking: {len(lm_dataset['train'])}")
print(f"Test dataset size after chunking: {len(lm_dataset['test'])}")


# --- Initialize ModernBERT Model ---
bert_config = ModernBertConfig(
    vocab_size=hf_tokenizer.vocab_size,
    max_position_embeddings=max_sequence_length,
    global_rope_theta=10000,
    pad_token_id=hf_tokenizer.pad_token_id,
    bos_token_id=hf_tokenizer.bos_token_id,
    eos_token_id=hf_tokenizer.eos_token_id,
    cls_token_id=hf_tokenizer.cls_token_id,
    sep_token_id=hf_tokenizer.sep_token_id,
    # You might want to add hidden_size, num_hidden_layers, num_attention_heads
    # to match the base ModernBERT configuration if you are not loading
    # from a pretrained checkpoint. For pre-training from scratch, these are key.
    # hidden_size=768, # Example for a 'base' model
    # num_hidden_layers=12, # Example for a 'base' model
    # num_attention_heads=12, # Example for a 'base' model
)
model = ModernBertForMaskedLM(bert_config)
print("ModernBERT model initialized for Masked Language Modeling.")

# --- Data Collator ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer, mlm=True, mlm_probability=mlm_probability
)
print(f"Data collator for MLM initialized with mlm_probability={mlm_probability}.")

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", # For pre-training, usually minimize eval_loss
    greater_is_better=False, # Smaller loss is better
    push_to_hub=False,
    logging_dir=os.path.join(output_dir, "logs"), # Logs within output_dir
    logging_steps=500,
    report_to="none", # Change to "tensorboard" if you want to use TensorBoard
    fp16=torch.cuda.is_available(), # Enable mixed precision if GPU is available
    remove_unused_columns=False, # Essential for DataCollatorForLanguageModeling
    save_total_limit=2, # Keep only the last 2 checkpoints
)
print("Training arguments configured.")

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    tokenizer=hf_tokenizer, # Pass tokenizer for saving along with the model
    # For pre-training, compute_metrics is less common as loss is the primary metric.
    # You can remove compute_metrics if you only care about loss.
    # If you want perplexity during evaluation steps, you would define compute_metrics
    # to calculate it from eval_loss. The eval_loss is already logged by Trainer.
)
print("Trainer initialized.")

# --- Train the model ---
print("Starting pre-training...")
try:
    trainer.train()
    print("Pre-training complete.")
except Exception as e:
    print(f"An error occurred during training: {e}")
    exit(1)

# Save the pre-trained model and tokenizer
trainer.save_model(output_dir)
hf_tokenizer.save_pretrained(output_dir)
print(f"Pre-trained model and tokenizer saved to {output_dir}")

# --- Optional: Perplexity Evaluation on the final model ---
print("Calculating final perplexity on the test set...")
try:
    eval_results = trainer.evaluate()
    # Perplexity is exp(loss)
    if 'eval_loss' in eval_results:
        perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
        print(f"Final Test Perplexity: {perplexity:.2f}")
    else:
        print("Evaluation results did not contain 'eval_loss'.")
except Exception as e:
    print(f"Error during final evaluation: {e}")

print("\nPre-training script finished successfully.")

#!/usr/bin/env python3

import os
import torch
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

try:
    from transformers import ModernBertConfig, ModernBertForMaskedLM
except ImportError:
    print("ERROR: ModernBertConfig or ModernBertForMaskedLM not found in transformers.")
    print("Please ensure your 'transformers' installation (e.g., from GitHub) includes these classes.")
    exit("Required ModernBERT classes not found. Exiting.")


# --- Configuration ---
extracted_dataset_dir = "/home/navyat/nt/genome_subset_1000"  
output_dir = "/home/navyat/nt/modernbert-pretrain_out"      

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Model and training parameters
tokenizer_name = "answerdotai/ModernBERT-base" # Tokenizer for pre-training
mlm_probability = 0.15                         # Masked Language Model probability
max_sequence_length = 512                      # Max input sequence length for the model
learning_rate = 5e-5
batch_size = 8                                 
num_epochs = 3                                

# --- Load and Split Dataset ---
print(f"Loading dataset from: {extracted_dataset_dir}")
try:
    raw_dataset = load_from_disk(extracted_dataset_dir)
except Exception as e:
    print(f"Error loading dataset from {extracted_dataset_dir}: {e}")
    print("Please ensure the path is correct and the dataset is properly formatted (e.g., contains .arrow files).")
    exit(1)

# Confirm it's a Dataset object 
if not isinstance(raw_dataset, Dataset):
    raise TypeError(f"Expected a single Dataset, but loaded type was: {type(raw_dataset)}. Please check your dataset format.")

print("Loaded dataset is a single Dataset. Creating a 90/10 train/test split.")
train_test_split = raw_dataset.train_test_split(test_size=0.1, seed=42)
split_datasets = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})

print(f"Train dataset size: {len(split_datasets['train'])}")
print(f"Test dataset size: {len(split_datasets['test'])}")


# --- Tokenizer Loading ---
print(f"Loading tokenizer: {tokenizer_name}")
hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

special_tokens_to_add = {}
if hf_tokenizer.pad_token is None: special_tokens_to_add['pad_token'] = '[PAD]'
if hf_tokenizer.unk_token is None: special_tokens_to_add['unk_token'] = '[UNK]'
if hf_tokenizer.cls_token is None: special_tokens_to_add['cls_token'] = '[CLS]'
if hf_tokenizer.sep_token is None: special_tokens_to_add['sep_token'] = '[SEP]'
if hf_tokenizer.mask_token is None: special_tokens_to_add['mask_token'] = '[MASK]'

if special_tokens_to_add:
    hf_tokenizer.add_special_tokens(special_tokens_to_add)
    print(f"Added missing special tokens: {list(special_tokens_to_add.keys())}")


# --- Tokenization ---
sequence_column_name = "sequence" 

if sequence_column_name not in split_datasets["train"].column_names:
    raise ValueError(
        f"Column '{sequence_column_name}' not found in the dataset. "
        f"Available columns are: {split_datasets['train'].column_names}. "
        "Please check your dataset and adjust 'sequence_column_name' accordingly."
    )

def tokenize_function(examples):
    return hf_tokenizer(examples[sequence_column_name], return_attention_mask=False)

print("Tokenizing dataset...")

num_workers = os.cpu_count() if os.cpu_count() else 1
tokenized_dataset = split_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=num_workers,
    remove_columns=split_datasets["train"].column_names, 
    load_from_cache_file=False
)
print("Dataset tokenization complete.")


# --- Grouping texts into fixed-size chunks ---
def group_texts(examples):
    # Concatenate all texts from a batch into a single list
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Calculate the total length, rounding down to a multiple of max_sequence_length
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_sequence_length) * max_sequence_length
    # Split the concatenated texts into chunks of max_sequence_length
    result = {
        k: [t[i : i + max_sequence_length] for i in range(0, total_length, max_sequence_length)]
        for k, t in concatenated_examples.items()
    }
    # For Masked Language Modeling, the labels are just the input_ids
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
print(f"Train dataset size after chunking: {len(lm_dataset['train'])}")
print(f"Test dataset size after chunking: {len(lm_dataset['test'])}")


# --- Initialize ModernBERT Model ---
print("Initializing ModernBERT model...")
bert_config = ModernBertConfig(
    vocab_size=hf_tokenizer.vocab_size,
    hidden_size=768,          # Base model size
    num_hidden_layers=12,     # Base model layers
    num_attention_heads=12,   # Base model attention heads
    intermediate_size=3072,   # 4 * hidden_size for base model
    max_position_embeddings=max_sequence_length,
    global_rope_theta=10000,  # Specific to ModernBERT's RoPE
    type_vocab_size=1,        # Typically 1 for single-sequence tasks like MLM on DNA
    pad_token_id=hf_tokenizer.pad_token_id,
    bos_token_id=hf_tokenizer.bos_token_id,
    eos_token_id=hf_tokenizer.eos_token_id,
    cls_token_id=hf_tokenizer.cls_token_id,
    sep_token_id=hf_tokenizer.sep_token_id,
)
model = ModernBertForMaskedLM(bert_config)
print(f"ModernBERT model initialized with {model.num_parameters():.2e} parameters.")


# --- Data Collator ---
# This collator dynamically masks tokens for MLM during training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer,
    mlm=True,
    mlm_probability=mlm_probability
)


# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",      # Evaluate at the end of each epoch
    save_strategy="epoch",            # Save checkpoint at the end of each epoch
    load_best_model_at_end=True,      # Load the best model found during training
    metric_for_best_model="eval_loss",# Metric to determine the best model
    greater_is_better=False,          # Smaller loss is better
    push_to_hub=False,                # Do not push to Hugging Face Hub
    logging_dir=os.path.join(output_dir, "logs"), # TensorBoard logs
    logging_steps=500,                # Log every N steps
    report_to="none",                 # No external reporting (e.g., wandb, tensorboard)
    fp16=torch.cuda.is_available(),   # Enable mixed precision if GPU is available
    remove_unused_columns=False,      # Required for DataCollatorForLanguageModeling
    save_total_limit=1,               # Only keep the best model checkpoint
)


# --- Trainer Initialization and Training ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"], # Used for internal evaluation to determine best model
    data_collator=data_collator,
    tokenizer=hf_tokenizer,          # Pass tokenizer for saving along with the model
)

print("Starting pre-training...")
try:
    trainer.train()
    print("Pre-training complete.")
except Exception as e:
    print(f"An error occurred during training: {e}")
    exit(1)

# --- Save Final Model and Tokenizer ---
print("Saving final pre-trained model and tokenizer...")
trainer.save_model(output_dir)
hf_tokenizer.save_pretrained(output_dir)
print(f"Pre-trained model and tokenizer saved to {output_dir}")

print("\nPre-training script finished successfully.")

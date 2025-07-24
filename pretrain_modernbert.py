# train_modernbert_genome.py

import os
import math
import torch
# import pandas as pd # Not needed for direct HF dataset loading
# import numpy as np # Not needed for direct HF dataset loading
# import wandb # Commented out wandb import

from datasets import load_dataset, Dataset
from transformers import (
    PreTrainedTokenizerFast,
    ModernBertConfig,
    ModernBertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EvalPrediction
)

# --- Updated load_data function to load from Hugging Face Hub ---
def load_data(dataset_name: str, dataset_split: str = "train", dataset_config: str = None):
    """
    Loads a dataset from the Hugging Face Hub using datasets.load_dataset().

    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub (e.g., "InstaDeepAI/multi_species_genomes").
        dataset_split (str): The split to load (e.g., "train", "validation", "test"). Defaults to "train".
        dataset_config (str, optional): The name of a specific dataset configuration if available
                                        (e.g., "6kbp" for multi_species_genomes). Defaults to None.

    Returns:
        Dataset: The loaded Hugging Face dataset object.

    Raises:
        Exception: If there's an error loading the dataset, usually due to network
                   issues, incorrect dataset name/config, or insufficient disk space
                   in the cache directory (controlled by HF_HOME).
    """
    print(f"Attempting to load dataset '{dataset_name}' with split '{dataset_split}' from Hugging Face Hub...")
    if dataset_config:
        print(f"Using dataset configuration: {dataset_config}")

    try:
        # This is the correct call for loading from the Hugging Face Hub
        # It will automatically manage downloading and caching to the path set by HF_HOME.
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
        print(f"Successfully loaded dataset '{dataset_name}' with {len(dataset)} examples.")
        return dataset
    except Exception as e:
        print(f"ERROR: Failed to load dataset '{dataset_name}': {e}")
        hf_home_env = os.environ.get("HF_HOME")
        if hf_home_env:
            print(f"Hugging Face cache directory is set to: {hf_home_env}")
            print(f"Please ensure directory '{hf_home_env}' exists and has write permissions and sufficient disk space.")
        else:
            print("HF_HOME environment variable is NOT set. Hugging Face cache defaults to ~/.cache/huggingface.")
            print("Consider setting HF_HOME to a directory with more space (e.g., in /projects/lz25/navyat/).")
        raise # Re-raise the exception so the job fails clearly


def load_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        special_tokens=["<unk>", "<pad>", "<cls>", "<sep>", "<mask>"],
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
    )
    # The ModernBert tokenizer expects <s> and </s> as BOS/EOS tokens
    if "<s>" not in tokenizer.all_special_tokens:
        tokenizer.add_special_tokens({'bos_token': '<s>'})
    if "</s>" not in tokenizer.all_special_tokens:
        tokenizer.add_special_tokens({'eos_token': '</s>'})
    
    required_tokens = ["<pad>", "<cls>", "<sep>", "<mask>", "<unk>", "<s>", "</s>"]

    print("Checking tokenizer special token IDs:")
    for token in required_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            print(f"Missing token: {token} (This might indicate an issue with your tokenizer file or logic)")
        else:
            print(f"{token} ID: {token_id}")
    print('\n')
    return tokenizer


def tokenize_dataset(dataset, tokenizer, max_len=512):
    def tokenize_fn(examples):
        # The 'InstaDeepAI/multi_species_genomes' dataset has a 'sequence' column.
        return tokenizer(
            examples["sequence"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_special_tokens_mask=True
        )

    print("Starting dataset tokenization...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=max(1, os.cpu_count() // 2),
        remove_columns=[col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "token_type_ids"]]
    )
    if "text" in tokenized.column_names:
        tokenized = tokenized.remove_columns("text")

    print("Tokenization complete.")
    print(f"Tokenized dataset features: {tokenized.column_names}")
    print('\n')
    return tokenized


# --- compute_metrics function with wandb logging commented out ---
def compute_metrics(eval_preds: EvalPrediction):
    predictions, labels = eval_preds.predictions, eval_preds.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    logits = torch.tensor(predictions)
    labels = torch.tensor(labels)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    perplexity = torch.exp(loss).item() if loss.item() < 100 else float("inf")

    # wandb.log({
    #     "eval_loss": loss.item(),
    #     "eval_perplexity": perplexity
    # })
    
    print(f"[Eval] Loss: {loss.item():.4f}, Perplexity: {perplexity:.4f}")
    print('\n')

    return {"eval_loss": loss.item(), "perplexity": perplexity}


def main():
    # Configuration
    dataset_name = "InstaDeepAI/multi_species_genomes"
    dataset_config = "6kbp"
    tokenizer_path = "/opt/home/e128037/Genome_multi/4k_unigram_tokenizer.json"
    output_dir = "./modernbert_model_checkpoints"
    final_model_save_path = "/opt/home/e128037/Genome_multi/modernbert_final_model"
    
    # Login to Weights & Biases
    # wandb.login(key="e7e01f84083cf5898f0c4607d529b71cb7c89f73")
    # print("Logged into Weights & Biases.")
    print('\n')

    # Load data from Hugging Face Hub
    dataset = load_data(dataset_name, dataset_config=dataset_config, dataset_split="train")
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_len=512)

    # Train/Test Split
    print("Splitting dataset into train and test sets...")
    split = tokenized_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    print('\n')

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Model Config and Initialization
    config = ModernBertConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        global_rope_theta=10000,
    )

    model = ModernBertForMaskedLM(config=config)
    print("Model initialized with ModernBertConfig.")
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings to {len(tokenizer)} for new tokens.")
    print(f"Model number of parameters: {model.num_parameters()}")
    print('\n')

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        weight_decay=1e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.06,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        fp16=True,
        # report_to=["wandb"], # Commented out report_to
        run_name="modernbert-pretrain-full-genome"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"].shuffle(seed=42),
        eval_dataset=split["test"].shuffle(seed=42),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    print('\n')
    trainer.train()

    os.makedirs(final_model_save_path, exist_ok=True)
    trainer.save_model(final_model_save_path)
    tokenizer.save_pretrained(final_model_save_path)
    print(f"Model and tokenizer saved to: {final_model_save_path}")
    print('\n')

    print("\nRunning final evaluation on test set...")
    final_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print("Final Evaluation Results:")
    print(final_metrics)
    # wandb.log(final_metrics) # Commented out wandb logging
    print(f"Final Evaluation Loss: {final_metrics.get('eval_loss', 0):.4f}, "
          f"Perplexity: {final_metrics.get('perplexity', float('nan')):.4f}")

    # wandb.finish() # Commented out wandb finish


if __name__ == "__main__":
    main()

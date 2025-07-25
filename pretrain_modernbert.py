# train_modernbert_genome.py

import os
import math
import torch
import pandas as pd
import numpy as np
import wandb

from datasets import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    ModernBertConfig,
    ModernBertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EvalPrediction
)

def load_data(csv_path, max_samples=None):
    """
    Loads data from a CSV file.
    If max_samples is an integer, it loads up to that many samples.
    If max_samples is None, it loads all samples from the CSV.
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df['length'] = df['sequence'].str.len()
    print("Dataset loaded. Sequence length stats:\n", df['length'].describe())

    dataset = Dataset.from_pandas(df)
    dataset = dataset.rename_column("sequence", "text")
    
    if max_samples is not None:
        # Ensure we don't try to select more rows than exist
        num_to_select = min(max_samples, len(dataset))
        print(f"Selecting {num_to_select} samples from the CSV.")
        return dataset.select(range(num_to_select))
    else:
        print(f"Loading all {len(dataset)} samples from the CSV.")
        return dataset


def load_tokenizer(tokenizer_path):
    print(f"Loading tokenizer from: {tokenizer_path}")
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

    print("Checking tokenizer special token IDs:")
    required_tokens = ["<pad>", "<cls>", "<sep>", "<mask>", "<unk>", "<s>", "</s>"]
    for token in required_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            print(f"Missing token: {token}")
        else:
            print(f"{token} ID: {token_id}")
    return tokenizer


def tokenize_dataset(dataset, tokenizer, max_len=512):
    print(f"Starting dataset tokenization with max_len={max_len}...")
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=max(1, os.cpu_count() // 2), # Use half available CPU cores for mapping
        remove_columns=[col for col in dataset.column_names if col != "text"]
    )
    print("Tokenization complete.")
    print(f"Tokenized dataset features: {tokenized.column_names}")
    return tokenized


def compute_metrics(eval_preds: EvalPrediction):
    predictions, labels = eval_preds.predictions, eval_preds.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    logits = torch.tensor(predictions)
    labels = torch.tensor(labels)

    # Shift for causal language modeling loss calculation (common in BERT pre-training for next token prediction)
    # The original ModernBert pre-training might only do MLM, but if it's also doing NSP or next token, this is relevant.
    # For a pure MLM objective, you might not shift, and compute loss directly on original logits/labels
    # where -100 indicates masked tokens. Assuming your original intent for this part is correct for your task.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) # -100 is typically used for tokens not to be predicted/ignored
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss).item() if loss.item() < 100 else float("inf")

    # Log to Weights & Biases
    wandb.log({
        "eval_loss": loss.item(),
        "eval_perplexity": perplexity
    })
    
    print(f"Evaluation: Loss={loss.item():.4f}, Perplexity={perplexity:.4f}")

    return {"eval_loss": loss.item(), "perplexity": perplexity}


def main():
    # Configuration
    csv_path = "/opt/home/e128037/Genome_multi/multi_species_subset_analysis.csv"
    tokenizer_path = "/opt/home/e128037/Genome_multi/4k_unigram_tokenizer.json"
    output_dir = "./model_files_01"
    save_model_path = "/opt/home/e128037/Genome_multi/model_files_01"
    
    # Login to Weights & Biases
    wandb_api_key = os.environ.get("WANDB_API_KEY", "e7e01f84083cf5898f0c4607d529b71cb7c89f73")
    wandb.login(key=wandb_api_key)
    print("Logged into Weights & Biases.")

    # Load data from the specified CSV file
    # By default, load_data will now load all samples from the CSV.
    # If you want to limit it (e.g., to the first 5000), use: dataset = load_data(csv_path, max_samples=5000)
    dataset = load_data(csv_path) 
    tokenizer = load_tokenizer(tokenizer_path)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Train/Test Split
    # This splits the data loaded from your CSV into a training set and a 10% test set.
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"] # This is the subset of your CSV used for evaluation
    print(f"Train size: {len(train_dataset)}, Test (evaluation) size: {len(eval_dataset)}")

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Model Config and Initialization
    config = ModernBertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
        global_rope_theta=10000,
        # Ensure BOS/EOS token IDs are set if you're using them in your tokenizer and data collation
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = ModernBertForMaskedLM(config=config)
    # Ensure model's embedding size matches tokenizer's vocab size if new tokens were added
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings to {len(tokenizer)} for tokenizer's full vocabulary.")
    print(f"Model initialized with {model.num_parameters()} parameters.")

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
        eval_steps=10, # Evaluate every 10 steps
        fp16=True,
        report_to=["wandb"],
        run_name="modernbert-pretrain"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.shuffle(seed=42),
        eval_dataset=eval_dataset.shuffle(seed=42),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()

    # Save final model and tokenizer
    os.makedirs(save_model_path, exist_ok=True)
    trainer.save_model(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    print(f"Model and tokenizer saved to: {save_model_path}")

    # Final Evaluation
    print("\nRunning final evaluation on test set...")
    final_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print("Final Evaluation Results:")
    print(final_metrics)
    # Note: eval_perplexity might not be directly in final_metrics from trainer.evaluate
    # if compute_metrics only returns eval_loss. Check your compute_metrics function
    # for exact key names being returned. Assuming 'perplexity' is what you want.
    print(f"Final Evaluation Loss: {final_metrics.get('eval_loss', float('nan')):.4f}, "
          f"Perplexity: {final_metrics.get('perplexity', float('nan')):.4f}")

    wandb.finish() # End WandB run


if __name__ == "__main__":
    main()

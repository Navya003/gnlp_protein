# filename: build_modernbert_dna_tokenizer.py (UPDATED)

# Required imports
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors, decoders
from tokenizers import Regex
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
import os

# --- 1. Function to get training corpus from your DNA dataset directory ---
# This path is set based on your 'ls' output, assuming the script is run from the 'nt' directory.
DNA_DATA_DIR = "genome_subset_1000" 

def get_training_corpus(dataset_path):
    """
    Loads a Hugging Face dataset from disk (a directory) and yields text sequences for training.
    Assumes the text content is in a column named 'sequence'.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset directory was not found at: {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded with {len(dataset)} examples. Expected columns: {dataset.column_names}")
    
    # --- IMPORTANT CHANGE HERE: Look for 'sequence' column ---
    if 'sequence' not in dataset.column_names:
        raise ValueError(f"Error: 'sequence' column not found in the dataset. Available columns: {dataset.column_names}")

    for item in dataset:
        yield item['sequence'].strip() # Changed from item['text'] to item['sequence']

# --- 2. Initialize the Tokenizer with a Unigram model ---
print("Step 1: Initializing Tokenizer with Unigram model...")
tokenizer = Tokenizer(models.Unigram())

# --- 3. Define and apply Normalization rules (simplified for DNA) ---
print("Step 2: Defining and applying Normalization rules (simplified for DNA)...")
tokenizer.normalizer = normalizers.Sequence(
    [
        # Add DNA-specific normalizers here if needed, e.g., normalizers.Uppercase()
    ]
)

# --- 4. Define the Pre-tokenizer (Metaspace) ---
print("Step 3: Defining Pre-tokenizer (Metaspace)...")
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

# Test the pre-tokenizer (example for text)
print("\nTesting pre-tokenizer with an example text:")
pre_tokenized_example = tokenizer.pre_tokenizer.pre_tokenize_str("Let's test the pre-tokenizer!")
print(pre_tokenized_example)

# --- 5. Define Special Tokens and Trainer, then Train the Model ---
print("\nStep 4: Defining Special Tokens and Trainer, then Training the Model...")
special_tokens_mb = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>"] 

trainer_mb = trainers.UnigramTrainer(
    vocab_size=25000, # Adjust vocabulary size as needed
    special_tokens=special_tokens_mb,
    unk_token="<unk>",
)

print(f"Starting tokenizer training from dataset in {DNA_DATA_DIR} (this might take a moment)...")
tokenizer.train_from_iterator(get_training_corpus(DNA_DATA_DIR), trainer=trainer_mb)
print("Tokenizer training complete.")

# --- 6. Configure Post-processing for ModernBERT-style formatting ---
print("\nStep 5: Configuring Post-processing for ModernBERT-style special tokens...")
cls_token_id_mb = tokenizer.token_to_id("<cls>")
sep_token_id_mb = tokenizer.token_to_id("<sep>")

if cls_token_id_mb is None or sep_token_id_mb is None:
    print("Warning: <cls> or <sep> token ID not found. Ensure they were in special_tokens during training.")
    cls_token_id_mb = cls_token_id_mb if cls_token_id_mb is not None else 0 
    sep_token_id_mb = sep_token_id_mb if sep_token_id_mb is not None else 1

print(f"CLS token ID: {cls_token_id_mb}, SEP token ID: {sep_token_id_mb}")

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"<cls>:0 $A:0 <sep>:0",
    pair=f"<cls>:0 $A:0 <sep>:0 $B:1 <sep>:1",
    special_tokens=[("<cls>", cls_token_id_mb), ("<sep>", sep_token_id_mb)],
)

# Test with a text example
print("\nTesting post-processor with a text example:")
encoding_test_single = tokenizer.encode("This is a single sentence.")
print(f"Single sequence tokens: {encoding_test_single.tokens}")
print(f"Single sequence type IDs: {encoding_test_single.type_ids}")

encoding_test_pair = tokenizer.encode("First part of sentence.", "Second part here!")
print(f"Paired sequence tokens: {encoding_test_pair.tokens}")
print(f"Paired sequence type IDs: {encoding_test_pair.type_ids}")

# --- 7. Add a Decoder ---
print("\nStep 6: Adding a Metaspace Decoder...")
tokenizer.decoder = decoders.Metaspace()

# Test decoding
test_text_to_decode = "Hello, this is a test sentence for decoding."
encoded_ids = tokenizer.encode(test_text_to_decode).ids
decoded_text = tokenizer.decode(encoded_ids)
print(f"Original text: '{test_text_to_decode}'")
print(f"Decoded text:  '{decoded_text}'")

# --- 8. Save the raw tokenizer file ---
output_tokenizer_path = "modernbert_unigram_dna_tokenizer.json"
print(f"\nStep 7: Saving the raw tokenizer to {output_tokenizer_path}...")
tokenizer.save(output_tokenizer_path)
print("Raw tokenizer saved.")

# --- 9. Wrap the tokenizer for Hugging Face Transformers ---
print("\nStep 8: Wrapping the tokenizer for Hugging Face Transformers (PreTrainedTokenizerFast)...")
wrapped_tokenizer_mb = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
    padding_side="right", # For ModernBERT (like BERT)
)
print("Tokenizer wrapped successfully in PreTrainedTokenizerFast.")

# Example of using the wrapped tokenizer
print("\nExample using the wrapped HF wrapped tokenizer:")
hf_encoding = wrapped_tokenizer_mb("ATCGATCGA") # Example DNA sequence
print(f"HF wrapped tokenizer encoded input IDs: {hf_encoding.input_ids}")
print(f"HF wrapped tokenizer tokens: {wrapped_tokenizer_mb.convert_ids_to_tokens(hf_encoding.input_ids)}")

print("\nScript execution complete. The tokenizer file is saved as 'modernbert_unigram_dna_tokenizer.json'.")

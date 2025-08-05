# extract_subset.py

from datasets import load_dataset
import pandas as pd
import os


DATASET_NAME = "InstaDeepAI/multi_species_genomes"
OUTPUT_CSV_FILE = "full_dataset.csv"
CHUNK_SIZE = 10000  # OOM errors

def convert_full_dataset_to_csv():
    print(f"Loading full dataset from '{DATASET_NAME}'...")
    try:
        # Load the full dataset directly
        dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset has {len(dataset)} examples. Processing in chunks of {CHUNK_SIZE}...")
    

    with open(OUTPUT_CSV_FILE, "w", newline='') as f:
       
        header = list(dataset.features.keys())
        f.write(','.join(header) + '\n')
        
        # Iterate through the dataset in chunks
        for i in range(0, len(dataset), CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, len(dataset))
            
            # Select the current chunk
            chunk = dataset.select(range(i, end))
            
            # Convert to pandas DataFrame
            df_chunk = pd.DataFrame(chunk)
            
            # Append to the CSV file without the header
            df_chunk.to_csv(f, header=False, index=False)
            
            print(f"  - Processed rows {i} to {end-1}")

    print(f"Successfully converted the full dataset to '{OUTPUT_CSV_FILE}'.")


if __name__ == "__main__":
    convert_full_dataset_to_csv()

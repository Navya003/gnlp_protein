import os
from datasets import load_from_disk

def chunk_sequence(sequence, chunk_size=512):
    """Chunks a sequence into smaller pieces of a specified size."""
    return [sequence[i:i+chunk_size] for i in range(0, len(sequence), chunk_size)]

def main():
    dataset_path = "genome_subset_1000"
    output_directory = "dna_text"
    sequences_per_file = 100 # Adjust as needed to control the number of output files

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded with {len(dataset)} sequences.")

    file_count = 0
    current_sequences = []

    for i, item in enumerate(dataset):
        seq = item['text']
        if len(seq) > 512:
            # Chunk long sequences
            chunks = chunk_sequence(seq, 512)
            current_sequences.extend(chunks)
        else:
            current_sequences.append(seq)

        # Write to file if we have enough sequences or it's the last sequence
        if len(current_sequences) >= sequences_per_file or i == len(dataset) - 1:
            output_filepath = os.path.join(output_directory, f"chunk_{file_count:04d}.txt")
            with open(output_filepath, 'w') as f:
                for s in current_sequences:
                    f.write(s + "\n")
            print(f"Saved {len(current_sequences)} sequences to {output_filepath}")
            current_sequences = [] # Reset for next file
            file_count += 1

    print(f"Data preparation complete. Created {file_count} files in '{output_directory}' directory.")

if __name__ == "__main__":
    main()

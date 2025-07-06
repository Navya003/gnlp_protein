# updated
#!/usr/bin/python
import inspect
import os, glob, time, json, datetime
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# tokenise fasta sequences with sentencepiece and export as json file
import argparse
# import gzip # Not used in provided snippet
# import screed # Not used in provided snippet
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast
# from utils import _cite_me # Uncomment if _cite_me is used and defined

# Make sure datasets is imported if it's used elsewhere
import datasets


def sort_filepaths_by_filename(filepaths):
    return sorted(filepaths, key=lambda x: os.path.basename(x))

def file_iterator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

def get_subdirectories(folder_path):
    """
    Get the names of all subdirectories in the specified folder.
    :param folder_path: Path to the folder
    :return: A list of subdirectory names
    """
    subdirectories = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories

def list_files_in_directory(directory):
    files = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                files.append(entry.path)
    files_sorted = sort_filepaths_by_filename(files)
    return files_sorted

def create_dnatokeniser_from_vocab(init_vocab, data_iterator, filesave_index, inputfile_path, vocab_size, special_tokens, output_dir):
    tokeniser = SentencePieceUnigramTokenizer(init_vocab)

    tokeniser.train_from_iterator(
        data_iterator,
        unk_token="<unk>",
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=special_tokens,
    )

    filename = os.path.basename(inputfile_path)
    filename_with_ext = filename + ".json"
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # Save only if the condition is met or if it's the last file
    if filesave_index % 5 == 0 or filesave_index == len(infile_paths) - 1: # infile_paths needs to be passed or accessed globally
        tokeniser_path = os.path.join(output_dir, filename_with_ext)
        print(f"Saving tokenizer to: {tokeniser_path}")
        tokeniser.save(tokeniser_path)
        # It seems the intent was to update the vocabulary for the next iteration
        # This part assumes a specific structure for the vocabulary that needs to be extracted
        # and re-fed. A more direct way to get the tokenizer's current vocab might be better
        # if the SentencePieceUnigramTokenizer object allows it directly.
        # For now, we'll keep the temp file approach, but note its inefficiency.
        tokeniser.save('temp_tokeniser.json')
        with open('temp_tokeniser.json', encoding='utf-8-sig') as f:
            json_file = json.load(f)
        vocab = json_file['model']['vocab']
        for idx, v in enumerate(vocab):
            vocab[idx] = tuple(v)
        return vocab
    else:
        # If not saving, still need to extract the vocabulary for the next iteration
        tokeniser.save('temp_tokeniser.json') # This still creates the temp file
        with open('temp_tokeniser.json', encoding='utf-8-sig') as f:
            json_file = json.load(f)
        vocab = json_file['model']['vocab']
        for idx, v in enumerate(vocab):
            vocab[idx] = tuple(v)
        return vocab


def main():
    parser = argparse.ArgumentParser(
        description='Take gzip fasta file(s), run SentencePiece and export json.'
    )
    parser.add_argument('-i', '--infile_paths', type=str, default=None, nargs="+",
                        help='path to files with biological seqs split by line')
    parser.add_argument('-t', '--tokeniser_path', type=str, default="Tokeniser_4k", # Changed default to 'Tokeniser_4k'
                        help='path to tokeniser.json file to save or load data (also used as output directory)')
    parser.add_argument('-v', '--vocab_size', type=int, default=4096,
                        help='select vocabulary size (DEFAULT: 4096, changed from 32000)') # Corrected default
    parser.add_argument('-b', '--break_size', type=int, default=0,
                        help='split long reads, keep all by default (DEFAULT: 0)')
    parser.add_argument('-c', '--case', type=str, default=None,
                        help='change case, retain original by default (DEFAULT: None)')
    parser.add_argument('-s', '--special_tokens', type=str, nargs="+",
                        default=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                        help='assign special tokens, eg space and pad tokens '
                             '(DEFAULT: ["<s>", "</s>", "<unk>", "<pad>", "<mask>"])')
    parser.add_argument('-e', '--example_seq', type=str, default=None,
                        help='show token to seq map for a sequence (DEFAULT: None)')
    args = parser.parse_args()

    # Capture command line arguments for reproducibility
    command_line_args = " ".join([i for i in sys.argv])
    print(f"COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t {command_line_args}\n")

    # Hardcoded directory for input files. Consider making this an argument if it varies.
    directory = 'dna_text'
    # Use the tokeniser_path argument as the output directory
    output_dir = args.tokeniser_path

    # Check if the primary vocabulary file exists before trying to open it
    primary_vocab_file = 'init_tokenizer.json'
    if not os.path.exists(primary_vocab_file):
        raise FileNotFoundError(f"Initial tokenizer file not found: {primary_vocab_file}")

    with open(primary_vocab_file, encoding='utf-8-sig') as f:
        json_file = json.load(f)
    primary_vocab = json_file['model']['vocab']
    for idx, v in enumerate(primary_vocab):
        primary_vocab[idx] = tuple(v)

    # If infile_paths is not provided via command line, list files from the hardcoded directory
    global infile_paths # Declare global to be accessible in create_dnatokeniser_from_vocab for len()
    if args.infile_paths:
        infile_paths = args.infile_paths
    else:
        infile_paths = list_files_in_directory(directory)
        if not infile_paths:
            raise OSError(f"No input files found in directory: {directory}. Provide either input fasta files or ensure 'dna_text' contains files.")

    init_vocab = None # Initialize init_vocab outside the loop

    if infile_paths:
        for i, inputfile_path in enumerate(infile_paths):
            print(f"Processing ---- {inputfile_path}")
            start_time = time.time()
            print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

            if i == 0:
                init_vocab = primary_vocab
            # The 'init_vocab' for subsequent iterations is the 'vocab' returned from the previous call

            data_iterator = file_iterator(inputfile_path)
            # Pass all necessary arguments explicitly
            vocab = create_dnatokeniser_from_vocab(
                init_vocab,
                data_iterator,
                i,
                inputfile_path,
                args.vocab_size, # Use args.vocab_size
                args.special_tokens, # Use args.special_tokens
                output_dir # Pass output_dir
            )
            init_vocab = vocab # Update init_vocab for the next iteration

            time_taken = time.time() - start_time
            total_time = str(datetime.timedelta(seconds=time_taken))
            print(f"--- Total {total_time} seconds ---")


if __name__ == "__main__":
    start_time_main = time.time() # Renamed to avoid conflict if `start_time` used internally in `main`
    main()
    # _cite_me() # Uncomment if _cite_me is used
    print(f"--- Script finished in {str(datetime.timedelta(seconds=time.time() - start_time_main))} ---")

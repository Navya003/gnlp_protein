#!/usr/bin/python
# make sure utils can be imported
import inspect
import os,glob,time,json,datetime
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# tokenise fasta sequences with sentencepiece and export as json file
import argparse
import gzip
import os
import sys
from warnings import warn
import screed
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast
#from utils import _cite_me
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

def main():
    parser = argparse.ArgumentParser(
        description='Take gzip fasta file(s), run SentencePiece and export json.'
    )
    parser.add_argument('-i', '--infile_paths', type=str, default=None, nargs="+",
                        help='path to files with biological seqs split by line')
    parser.add_argument('-t', '--tokeniser_path', type=str, default="",
                        help='path to tokeniser.json file to save or load data')
    parser.add_argument('-v', '--vocab_size', type=int, default=4096,
                        help='select vocabulary size (DEFAULT: 32000)')
    parser.add_argument('-b', '--break_size', type=int, default=0,
                        help='split long reads, keep all by default (DEFAULT: 0)')
    parser.add_argument('-c', '--case', type=str, default=None,
                        help='change case, retain original by default (DEFAULT: None)')
    parser.add_argument('-s', '--special_tokens', type=str, nargs="+",
            default=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                        help='assign special tokens, eg space and pad tokens \
                        (DEFAULT: ["<s>", "</s>", "<unk>", "<pad>", "<mask>"])')
    parser.add_argument('-e', '--example_seq', type=str, default=None,
                        help='show token to seq map for a sequence \
                        (DEFAULT: None)')
    args = parser.parse_args()
    directory = 'dna_text'
    
    #loading datasets
    files = list_files_in_directory(directory)
   # print(files[4])
    infile_paths = files#[3:]#args.infile_paths
#    tokeniser_path ="genomenlp_tokeniser1.json"# args.tokeniser_path
    vocab_size = args.vocab_size
    break_size = args.break_size
    case = args.case
    special_tokens = args.special_tokens
    example_seq = args.example_seq

    if infile_paths == None:
        raise OSError("Provide either input fasta file!")
    
    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")


    def create_dnatokeniser_from_vocab(init_vocab,data_iterator,filesave_index):
        
        tokeniser = SentencePieceUnigramTokenizer(init_vocab)
         
        tokeniser.train_from_iterator(
            data_iterator,
            unk_token="<unk>",
            vocab_size=vocab_size,
            show_progress=True,
            special_tokens=special_tokens,
            )
        
        filename = os.path.basename(inputfile_path)

        filename=filename+".json"
        if filesave_index%5==0 or filesave_index==len(infile_paths)-1:
           tokeniser_path = "Tokeniser_4k/"+filename
           print(filename)
           tokeniser.save(tokeniser_path)
        tokeniser.save('temp_tokeniser.json')
        with open('temp_tokeniser.json', encoding='utf-8-sig') as f:
            json_file = json.load(f)
            vocab = json_file['model']
            vocab=vocab['vocab']
            for idx, v in enumerate(vocab):
                vocab[idx] = tuple(v)
        
       # print(vocab)
        return vocab


    primary_vocab_file = 'init_tokenizer.json'#chunk_146.json'#dnatokeniser.json'

    with open(primary_vocab_file, encoding='utf-8-sig') as f:
     json_file = json.load(f)
     primary_vocab = json_file['model']
     primary_vocab=primary_vocab['vocab']
     for idx, v in enumerate(primary_vocab):
        primary_vocab[idx] = tuple(v)
 #   print(infile_paths)
    if infile_paths:
        for i,inputfile_path in enumerate(infile_paths):
            print("Processing ---- ",inputfile_path)
            start_time = time.time()
            print("Start time :", start_time)
            if i==0:
                init_vocab=primary_vocab
            print(inputfile_path)
            data_iterator = file_iterator(inputfile_path)
            vocab = create_dnatokeniser_from_vocab(init_vocab,data_iterator,i)
            init_vocab = vocab
            time_taken = time.time() - start_time
            total_time = str(datetime.timedelta(seconds=time_taken))
            print("--- Total %s seconds ---" % (total_time))

if __name__ == "__main__":
    start_time = time.time()
    main()
   # _cite_me()
    print("--- Total %s seconds ---" % (time.time() - start_time))

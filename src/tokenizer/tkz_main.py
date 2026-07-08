import cProfile
import pstats
import json
from src.tokenizer.bpe import BPETrainer
from src.tokenizer import Tokenizer


def train_bpe(input_path, special_tokens, vocab_size):
    with cProfile.Profile() as pr:
        vocab, merges = BPETrainer(vocab_size, special_tokens).train(input_path)
    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    with open("vocab.json", "w") as f:
        json.dump({str(k): list(v) for k, v in vocab.items()}, f)
    with open("merges.txt", "w") as f:
        json.dump([(list(a), list(b)) for a, b in merges], f)


def tokenize_to_bin(input_path, dataset_bin):
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_path="vocab.json",
        merges_path="merges.txt",
        special_tokens=special_tokens)
    arr = tokenizer.encode_file_parallel(input_path) # arr is np.int32
    arr.tofile(dataset_bin)


if __name__ == "__main__":
    #-------------------------------- train bpe -------------------------------
    input_path = "data/raw/tinystories/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 500
    special_tokens = ['<|endoftext|>']
    train_bpe(input_path, special_tokens, vocab_size)
    
    #--------------------------- tokenize + create bin -------------------------
    dataset_bin = "data/processed/tinystories/TinyStoriesV2-GPT4-valid.bin"
    tokenize_to_bin(input_path, dataset_bin)
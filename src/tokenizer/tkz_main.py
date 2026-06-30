import cProfile
import pstats

import json

from src.tokenizer.bpe import BPETrainer

if __name__ == "__main__":

    input_path = "data/raw/tinystories/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 500
    special_tokens = ['<|endoftext|>']
    
    with cProfile.Profile() as pr:
        vocab, merges = BPETrainer(vocab_size, special_tokens).train(input_path)

    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    with open("vocab.json", "w") as f:
        json.dump({str(k): list(v) for k, v in vocab.items()}, f)
    
    with open("merges.txt", "w") as f:
        json.dump([(list(a), list(b)) for a, b in merges], f)

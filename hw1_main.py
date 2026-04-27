from hw1 import *

if __name__ == "__main__":
    
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10
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

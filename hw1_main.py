import cProfile
import pstats

from hw1 import *

if __name__ == "__main__":

    # input_path = "data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 10
    # special_tokens = ['<|endoftext|>']
    
    # with cProfile.Profile() as pr:
    #     vocab, merges = BPETrainer(vocab_size, special_tokens).train(input_path)

    # stats = pstats.Stats(pr)
    # stats.sort_stats("cumulative")
    # stats.print_stats(20)

    # with open("vocab.json", "w") as f:
    #     json.dump({str(k): list(v) for k, v in vocab.items()}, f)
    
    # with open("merges.txt", "w") as f:
    #     json.dump([(list(a), list(b)) for a, b in merges], f)

    # ── Hyperparameters ──────────────────────────────────────────────────────
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    context_length = 256
    batch_size   = 32
    num_iters    = 10    # 5000
    log_interval = 1   # 100
    d_model      = 512
    num_heads    = 16
    d_ff         = 1344  # d_model * 4
    num_layers   = 4
    theta        = 10_000.0
    max_lr           = 3e-4
    min_lr       = 3e-5
    betas = (0.9, 0.95)
    warmup_iters = max(1, num_iters // 10)
    cosine_cycle_iters = num_iters

    # ── Data ──────────────────────────────────────────────────────────────────
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath="vocab.json",
        merges_filepath="merges.txt",
        special_tokens=special_tokens,
    )
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    dataset_bin = "data/TinyStoriesV2-GPT4-valid.bin"
    if not os.path.exists(dataset_bin):
        with open(input_path) as f:
            text = f.read()
        token_ids = tokenizer.encode(text)
        arr = np.array(token_ids, dtype=np.int32)
        arr.tofile(dataset_bin)
        n_tokens = len(arr)
        del arr, token_ids, text
    else:
        n_tokens = os.path.getsize(dataset_bin) // np.dtype(np.int32).itemsize
    dataset = np.memmap(dataset_bin, dtype=np.int32, mode="r", shape=(n_tokens,))

    # ── Model + optimizer ────────────────────────────────────────────────────
    actual_vocab_size = max(tokenizer.vocab.keys()) + 1
    model = TransformerLM(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
        context_length=context_length,
        vocab_size=actual_vocab_size,
        num_layers=num_layers,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=max_lr,
        betas=betas,
        weight_decay=0.1,
        eps=1e-8,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    for step in range(1, num_iters + 1):
        lr = get_cosine_lr(step, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        x, y = data_loader(dataset, batch_size, context_length, device)
        logits = model(x)                          # (B, T, vocab_size)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % log_interval == 0:
            print(f"step {step:>6} | loss {loss.item():.4f} | lr {lr:.2e}")

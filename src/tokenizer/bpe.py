import os, re, regex
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm

from .utils import find_chunk_boundaries

class BPETrainer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab_size: int, special_tokens: list[str], num_processes: int = os.cpu_count() or 1):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = num_processes


    @staticmethod
    def _pretokenize_chunk(args: tuple) -> dict:
        """Worker: read one file chunk and return its word_counts dict."""
        input_path, start, end, special_tokens, pat = args
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

        if special_tokens:
            split_pat = '|'.join(re.escape(st) for st in sorted(special_tokens, key=len, reverse=True))
            parts = re.split(split_pat, chunk)
        else:
            parts = [chunk]

        word_counts: dict = defaultdict(int)
        for part in parts:
            for token in regex.findall(pat, part):
                word = tuple(bytes([b]) for b in token.encode('utf-8'))
                word_counts[word] += 1
        return dict(word_counts)

    def train(self, input_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        split_token = self.special_tokens[0].encode('utf-8') if self.special_tokens else b"\n"

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, split_token)

        chunk_args = [
            (input_path, start, end, self.special_tokens, self.PAT)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(processes=len(chunk_args)) as pool:
            chunk_counts = list(tqdm(
                pool.imap(self._pretokenize_chunk, chunk_args),
                total=len(chunk_args),
                desc="Pretokenizing chunks",
            ))

        # Merge word counts from all chunks
        word_counts: dict = defaultdict(int)
        for counts in chunk_counts:
            for word, count in counts.items():
                word_counts[word] += count

        # Initialize vocab: indices 0-255 are single bytes, then special tokens
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, st in enumerate(self.special_tokens):
            vocab[256 + i] = st.encode('utf-8')

        # Build initial pair counts
        pair_counts: dict = defaultdict(int)
        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += count

        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        merges: list[tuple[bytes, bytes]] = []
        next_id = 256 + len(self.special_tokens)

        for _ in tqdm(range(num_merges), desc="Computing BPE merges"):
            if not pair_counts:
                break

            # Pick best pair: highest count, break ties lexicographically
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

            new_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[next_id] = new_token
            next_id += 1

            # Merge best_pair in every word, updating pair counts incrementally
            new_words: dict = {}
            for word, count in word_counts.items():
                if best_pair[0] not in word:
                    new_words[word] = new_words.get(word, 0) + count
                    continue

                new_word: list = []
                i = 0
                while i < len(word):
                    if (i + 1 < len(word)
                            and word[i] == best_pair[0]
                            and word[i + 1] == best_pair[1]):
                        # Update left-neighbor pair
                        if new_word:
                            pair_counts[(new_word[-1], best_pair[0])] -= count
                            pair_counts[(new_word[-1], new_token)] += count
                        # Update right-neighbor pair (crosses the merge boundary)
                        if i + 2 < len(word):
                            pair_counts[(best_pair[1], word[i + 2])] -= count
                            pair_counts[(new_token, word[i + 2])] += count
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                t = tuple(new_word)
                new_words[t] = new_words.get(t, 0) + count

            del pair_counts[best_pair]
            word_counts = new_words

        return vocab, merges
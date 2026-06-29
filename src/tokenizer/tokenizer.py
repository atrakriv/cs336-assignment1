import os, re, regex, json
from functools import lru_cache
import heapq
from collections.abc import Iterable, Iterator
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from .utils import find_chunk_boundaries

class Tokenizer:
    PAT = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| """
        r"""?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    PAT_RE = regex.compile(PAT)

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges

        self.special_tokens = special_tokens or []
        self.special_token_set = set(self.special_tokens)

        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.inf = float("inf")

        # Ensure special tokens exist
        for st in self.special_tokens:
            b = st.encode("utf-8")
            if b not in self.inverse_vocab:
                nid = max(self.vocab.keys()) + 1
                self.vocab[nid] = b
                self.inverse_vocab[b] = nid

        # precompute special token ids
        self.special_token_ids = {
            st: self.inverse_vocab[st.encode("utf-8")]
            for st in self.special_tokens
        }

        # precompile special-token splitter
        self.special_pattern = None
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile(
                "(" + "|".join(re.escape(s) for s in sorted_special) + ")"
            )

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path) as f:
            vocab_data = json.load(f)
        vocab = {int(k): bytes(v) for k, v in vocab_data.items()}

        with open(merges_path) as f:
            merges = [(bytes(a), bytes(b)) for a, b in json.load(f)]

        return cls(vocab, merges, special_tokens)

    # -----------------------------
    # FAST GPT-2 STYLE BPE
    # -----------------------------
    @lru_cache(maxsize=200000)
    def _bpe_encode(self, token_bytes: bytes) -> tuple[int, ...]:
        merge_ranks = self.merge_ranks
        inverse_vocab = self.inverse_vocab
        inf = self.inf

        n = len(token_bytes)
        if n == 1:
            return (inverse_vocab[token_bytes],)

        symbols = [bytes([b]) for b in token_bytes]

        prev = [-1] + list(range(n - 1))
        nxt = list(range(1, n)) + [-1]
        alive = [True] * n

        heap = []

        def push(i):
            j = nxt[i]
            if j != -1 and alive[i] and alive[j]:
                r = merge_ranks.get((symbols[i], symbols[j]), inf)
                if r != inf:
                    heapq.heappush(heap, (r, i))

        for i in range(n - 1):
            push(i)

        while heap:
            rank, i = heapq.heappop(heap)

            if not alive[i]:
                continue

            j = nxt[i]
            if j == -1 or not alive[j]:
                continue

            if merge_ranks.get((symbols[i], symbols[j]), inf) != rank:
                continue

            # merge i + j
            symbols[i] = symbols[i] + symbols[j]
            alive[j] = False

            nxt[i] = nxt[j]
            if nxt[j] != -1:
                prev[nxt[j]] = i

            # update neighbors
            if prev[i] != -1:
                push(prev[i])

            push(i)

        ids = []
        i = 0
        while prev[i] != -1:
            i = prev[i]

        while i != -1:
            if alive[i]:
                ids.append(inverse_vocab[symbols[i]])
            i = nxt[i]

        return tuple(ids)
    
    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        ids = []
        append = ids.append
        extend = ids.extend

        parts = (
            self.special_pattern.split(text)
            if self.special_pattern
            else [text]
        )

        for part in parts:
            if not part:
                continue

            if part in self.special_token_set:
                append(self.special_token_ids[part])
            else:
                for tok in self.PAT_RE.findall(part):
                    extend(self._bpe_encode(tok.encode("utf-8")))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for t in iterable:
            yield from self.encode(t)


    @staticmethod
    def _encode_chunk(args):
        tokenizer, path, start, end = args
        with open(path, "rb") as f:
            f.seek(start)
            text = f.read(end - start).decode("utf-8", errors="ignore")
        return np.array(tokenizer.encode(text), dtype=np.int32)

    def encode_file_parallel(self, input_path, num_processes=None):
        num_processes = num_processes or os.cpu_count() or 1

        split_token = (
            self.special_tokens[0].encode("utf-8")
            if self.special_tokens
            else b"\n"
        )

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, split_token
            )

        args = [
            (self, input_path, a, b)
            for a, b in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(processes=len(args)) as pool:
            chunks = list(
                tqdm(
                    pool.imap(self._encode_chunk, args),
                    total=len(args),
                    desc="Encoding",
                )
            )

        return np.concatenate(chunks)


    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode(
            "utf-8", errors="replace"
        )
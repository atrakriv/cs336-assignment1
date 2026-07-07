from __future__ import annotations

import json, time
from functools import lru_cache
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download


class WallclockTimer:
    """Tracks cumulative wallclock seconds elapsed, carrying over an offset from prior (resumed) runs."""

    def __init__(self, offset_sec: float = 0.0):
        self.offset_sec = offset_sec
        self._start = time.time()

    def elapsed(self) -> float:
        return self.offset_sec + (time.time() - self._start)


def convert_to_hf(vocab_in, merges_in):
    @lru_cache
    def gpt2_bytes_to_unicode() -> dict[int, str]:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def convert_merges_to_hf(merges_path: str | Path, out_path: str | Path) -> None:
        """Convert [[bytes1], [bytes2]] merges JSON to HF BPE format 'token1 token2' per line."""
        b2u = gpt2_bytes_to_unicode()
        with open(merges_path, encoding="utf-8") as f:
            raw = json.load(f)
        lines = [
            "".join(b2u[b] for b in left) + " " + "".join(b2u[b] for b in right)
            for left, right in raw
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def convert_vocab_to_hf(vocab_path: str | Path, out_path: str | Path) -> None:
        """Convert {str_id: [byte_values]} vocab JSON to HF BPE format {token_str: int_id}."""
        b2u = gpt2_bytes_to_unicode()
        with open(vocab_path, encoding="utf-8") as f:
            raw = json.load(f)
        hf_vocab = {
            "".join(b2u[b] for b in byte_list): int(str_id)
            for str_id, byte_list in raw.items()
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hf_vocab, f, ensure_ascii=False)

    vocab_out  = Path(vocab_in).stem  + "-hf.json"
    merges_out = Path(merges_in).stem + "-hf.txt"
    convert_vocab_to_hf(vocab_in, vocab_out)
    convert_merges_to_hf(merges_in, merges_out)
    print(f"Written {vocab_out} and {merges_out}")


# bin_path = 'data/tiny-stories/TinyStoriesV2-GPT4-valid-tkz.bin'
# path_in_repo = 'TinyStoriesV2-GPT4-valid.bin'
# repo_id = 'atakrv/tiny-stories-v2-gpt4'
# token = 'hf_...'
def upload_bin_to_hf(
    bin_path: str | Path,
    repo_id: str,
    path_in_repo: str | None = None,
    repo_type: str = "dataset",
    token: str | None = None,
) -> None:

    bin_path = Path(bin_path)
    if path_in_repo is None:
        path_in_repo = bin_path.name

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    url = api.upload_file(
        path_or_fileobj=str(bin_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded {bin_path} → {url}")


# repo_id = "atakrv/tiny-stories-v2-gpt4"
# filename = "TinyStoriesV2-GPT4-train.bin"
# local_dir = "data/tinystories/"
def download_bin_from_hf(
    repo_id: str,
    filename: str, 
    local_dir: str | Path,
    token: str | None = None,
) -> None:
    local_file_path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        repo_type="dataset",
        local_dir=local_dir
    )
    print(f"File downloaded and cached at: {local_file_path}")

"""Upload one model's saved activations (lm_probe/acts/*.npz) to a Hugging Face dataset repo (created private; the free private quota is ~100 GB, so the full set is public), check every
file's sha256 on the Hub, then delete the local copy and leave a shard_XXXX.uploaded marker (lm_probe.py treats
the shard as done). Nothing is deleted unless its checksum on the Hub matches.

    HF_ACTS_REPO=user/llm-idiom-probe-activations python experiments/upload_acts.py qwen3-0.6b

Token: HF_TOKEN, or the file in HF_TOKEN_FILE (default /root/.hf_token). Repo layout: <model>/shard_XXXX.npz,
<model>/meta.json, README.md.
Read back:  z = np.load("shard_0000.npz"); span = common.bf16_to_float32(z["span"])   # [N, L+1, D]
"""
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
from common import out_dir, resolve  # noqa: E402

README = """---
license: other
---
# llm-idiom-probe: pooled hidden states

Per model `<model>/shard_XXXX.npz` (from `experiments/lm_probe.py --save-acts`), one row per variant of
`substitutions_v1` without grammar flags (8 593 variants):

- `span`, `right`, `sent`: `[N, L+1, D]` mean hidden states over the span tokens, the tokens after the span and the
  whole sentence (without the start token); layer 0 = embeddings. Stored as bfloat16 bit patterns (`uint16`),
  convert with `(x.astype(np.uint32) << 16).view(np.float32)`. NaN = empty pool.
- `variant_id`, `condition`: join with the dataset / the E2-E3 parquet files.

`<model>/meta.json`: HF id, number of layers, hidden size, GPU.
"""


def token() -> str:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    return Path(os.environ.get("HF_TOKEN_FILE", "/root/.hf_token")).read_text().strip()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(name: str) -> None:
    short, _ = resolve(name)
    api = HfApi(token=token())
    repo = os.environ.get("HF_ACTS_REPO") or f"{api.whoami()['name']}/llm-idiom-probe-activations"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.file_exists(repo, "README.md", repo_type="dataset"):
        api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset")
    d = out_dir(short) / "lm_probe" / "acts"
    files = sorted(d.glob("shard_[0-9][0-9][0-9][0-9].npz"))
    if not files:
        print(f"[upload] {short}: nothing to upload")
        return
    local = {f.name: sha256(f) for f in files}
    meta = out_dir(short) / "meta.json"
    if meta.exists():
        api.upload_file(path_or_fileobj=str(meta), path_in_repo=f"{short}/meta.json", repo_id=repo, repo_type="dataset")
    api.upload_folder(folder_path=str(d), path_in_repo=short, repo_id=repo, repo_type="dataset",
                      allow_patterns=list(local), commit_message=f"Add {short} activations")
    info = {Path(i.path).name: i for i in api.get_paths_info(repo, [f"{short}/{n}" for n in local],
                                                            repo_type="dataset", expand=True)}
    bad = [n for n, h in local.items() if n not in info or not info[n].lfs or info[n].lfs.sha256 != h]
    if bad:
        sys.exit(f"[upload] {short}: checksum mismatch / missing on the Hub: {bad} -> local files kept")
    for f in files:
        f.unlink()
        f.with_suffix(".uploaded").write_text(local[f.name] + "\n")
    print(f"[upload] {short}: {len(files)} files verified on {repo} (sha256), local copies deleted")


if __name__ == "__main__":
    main(sys.argv[1])

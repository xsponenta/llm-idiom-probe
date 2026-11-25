"""Pinned, checksum-verified model downloads (for hosts whose network intercepts HTTPS to Hugging Face).

    python experiments/hf_manifest.py build                 # on a TRUSTED network: write hf_manifest.json
    python experiments/hf_manifest.py fetch qwen3-1.7b      # on the GPU host: download + verify, exit 1 on mismatch

`build` records, for every model in common.MODELS, the current commit of the repo and the checksum of each file
we need (sha256 for LFS files such as *.safetensors, the git blob sha1 for small files such as config.json).
`fetch` downloads exactly that commit and recomputes every checksum locally, so a proxy that tampers with the
weights or the tokenizer is detected. Only after `fetch` succeeds does run_all.sh run the model (offline).
"""
import fnmatch
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "hf_manifest.json"
ALLOW = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"]
IGNORE = ["consolidated*", "original/*", "*.gguf", "*.pth", "*.bin", "*.pt", "*.onnx", "*.msgpack", "*.h5"]


def wanted(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in ALLOW) and not any(fnmatch.fnmatch(name, p) for p in IGNORE)


def build() -> None:
    import requests

    sys.path.insert(0, str(HERE))
    from common import MODELS

    out = {}
    for short, hf_id in MODELS.items():
        d = requests.get(f"https://huggingface.co/api/models/{hf_id}", params={"blobs": "true"}, timeout=60).json()
        files = {}
        for s in d["siblings"]:
            if not wanted(s["rfilename"]):
                continue
            lfs = s.get("lfs")
            files[s["rfilename"]] = {"sha256": lfs["sha256"]} if lfs else {"git_sha1": s["blobId"]}
        out[short] = {"hf_id": hf_id, "revision": d["sha"], "files": files}
        print(f"{short:14s} {hf_id:48s} {d['sha'][:10]}  {len(files)} files")
    MANIFEST.write_text(json.dumps(out, indent=1))
    print(f"-> {MANIFEST}")


def digest(path: Path, kind: str) -> str:
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
    if kind == "git_sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(short: str) -> None:
    from huggingface_hub import snapshot_download

    m = json.loads(MANIFEST.read_text())[short]
    path = Path(snapshot_download(m["hf_id"], revision=m["revision"], allow_patterns=list(m["files"])))
    bad = []
    for name, want in m["files"].items():
        kind, value = next(iter(want.items()))
        p = path / name
        if not p.exists():
            bad.append(f"{name}: missing")
        elif digest(p, kind) != value:
            bad.append(f"{name}: {kind} mismatch")
    if bad:
        print(f"VERIFY FAILED for {short}:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print(f"verified {short}: {len(m['files'])} files match the manifest (revision {m['revision'][:10]})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["fetch"] and len(sys.argv) == 3:
        fetch(sys.argv[2])
    else:
        sys.exit(__doc__)

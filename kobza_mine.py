"""Read more Kobza documents and keep only sentences that may contain a UK SimIdioms idiom.

    .venv/bin/python kobza_mine.py            # docs 10_000 .. 310_000 of the first shard

Input: data/raw/kobza/000_ukr_corpus_00000.parquet (first shard, 1.7 GB; HF streaming kept timing out).
Not kept on disk; download it with:
    .venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Goader/kobza', \
        '000_ukr_corpus_00000.parquet', repo_type='dataset', revision='5ec24e168e06ed0e7e0ef70fd734f610ac9ebf93', \
        local_dir='data/raw/kobza')"

Loose prefilter (recall first): every word of an idiom variant (≥3 letters) must appear in the sentence
by its 4-char prefix. Exact / lemma matching happens later in build_dataset.py.
Output: data/raw/kobza/kobza_mined_sentences.parquet (doc_id, source, sentence)
"""
import json
import re
import time
from itertools import product

import pandas as pd
import pyarrow.parquet as pq

SHARD = "data/raw/kobza/000_ukr_corpus_00000.parquet"   # Goader/kobza @ 5ec24e16, first shard
START, N_DOCS = 10_000, 300_000          # first 10k docs are already in kobza_first_10000.parquet
PER_VARIANT = 400                         # stop collecting for a variant after this many sentences
OUT = "data/raw/kobza/kobza_mined_sentences.parquet"

WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*")
SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[«\"„“—–-]?[А-ЯІЇЄҐA-Z0-9])")


def norm(w):
    w = w.lower().replace("’", "'").replace("ʼ", "'")
    w = {"у": "в", "й": "і", "із": "з", "зі": "з"}.get(w, w)
    return "в" + w[1:] if len(w) >= 3 and w.startswith("у") else w


def prefix(w):
    return norm(w)[:4]


# idiom variants -> required prefixes
variants = set()
for line in open("data/raw/simidioms/sim_idioms.jsonl", encoding="utf-8"):
    r = json.loads(line)
    for x in r["uk"].get("idiom", []):
        if x.get("field") != "idiom":
            continue
        s = re.sub(r"\([^)]*\)|\[[^\]]*\]|\.\.\.|…", " ", x["text"]).lower()
        s = re.sub(r"\s*/\s*", "/", s)
        for combo in product(*[w.split("/") for w in s.split()]):
            ws = [w for w in (re.sub(r"[^\w'-]", "", w) for w in combo) if len(w) >= 3 and not w.endswith(("сь", "небудь"))]
            if len(ws) >= 2:
                variants.add(tuple(sorted({prefix(w) for w in ws})))
by_key = {}
for v in variants:
    by_key.setdefault(v[0], []).append(v)
count = dict.fromkeys(variants, 0)
print(f"{len(variants)} multi-word variants to look for", flush=True)

def docs():
    """Docs START .. START + N_DOCS of the shard, read batch by batch (the file does not fit in memory)."""
    i = 0
    for batch in pq.ParquetFile(SHARD).iter_batches(batch_size=5_000, columns=["id", "source", "text"]):
        for doc in batch.to_pylist():
            if i >= START + N_DOCS:
                return
            if i >= START:
                yield doc
            i += 1


rows, seen, t0 = [], set(), time.time()
for k, doc in enumerate(docs(), 1):
    for para in doc["text"].split("\n"):
        for s in SPLIT_RE.split(para.strip()):
            s = s.strip()
            low = s.lower()
            if (not 5 <= len(s.split()) <= 60 or len(s) > 500
                    or re.search(r"[ыэъё]", low) or not re.search(r"[іїєґ]", low)):
                continue
            P = {prefix(w) for w in WORD_RE.findall(low)}
            hit = [v for key in P & by_key.keys() for v in by_key[key]
                   if count[v] < PER_VARIANT and all(p in P for p in v)]
            if hit and low not in seen:
                seen.add(low)
                rows.append((doc["id"], doc["source"], s))
                for v in hit:
                    count[v] += 1
    if k % 20_000 == 0:
        print(f"{k:7} docs | {len(rows):7} sentences | "
              f"{sum(c > 0 for c in count.values())} variants seen | {time.time() - t0:.0f}s", flush=True)

pd.DataFrame(rows, columns=["doc_id", "source", "sentence"]).to_parquet(OUT, index=False)
print(f"saved {len(rows)} sentences -> {OUT}")

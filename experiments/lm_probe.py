"""E2-E4 in one forward pass over the raw sentences (no chat template).

    python experiments/lm_probe.py --models qwen3-1.7b --items-per-batch 4
    python experiments/lm_probe.py --models qwen3-0.6b --limit-items 20      # quick test

E2 surprisal: per variant, -log p (nats) of the span tokens, of the 3 tokens right after the span
   (does the model "expect" the continuation once the idiom is broken?), and of the whole sentence.
E3 similarity: per layer, cosine between the variant and ORIG of the same item (all variants), and NEUTRAL
   (neutral-frame variants). Three poolings: mean over the span tokens ("span"), mean over the tokens after
   the span ("right" context), mean over all tokens of the sentence ("sent").
E4 features: span-mean hidden states (all layers, float16; float32 if a value exceeds the float16 range) of ORIG, NEUTRAL and LIT_CONTEXT, for the
   figurative-vs-literal probe in analyze.py (train/test split by idiom there).
Output: data/results/gpu/<model>/lm_probe[_limitN]/shard_XXXX.{parquet,npz}; finished shards are skipped.
--save-acts (or SAVE_ACTS=1): also every variant's pooled hidden states, all layers, all three pools, in
   lm_probe/acts/shard_XXXX.npz: span / right / sent [N, L+1, D] as bfloat16 bits (uint16; read with
   common.bf16_to_float32), variant_id, condition. bfloat16 = the models' own precision, no float16 overflow.
   NaN = empty pool (no tokens after a sentence-final span). A shard counts as done once its acts file exists
   or acts/shard_XXXX.uploaded marks it as uploaded to the Hub (upload_acts.py) and deleted locally.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from common import (device, encode_with_offsets, free, limit_items, load_full, load_model, out_dir, resolve,
                    span_token_positions, working, write_meta)

E4_CONDITIONS = ("ORIG", "NEUTRAL", "LIT_CONTEXT")
POST = 3
POOLS = ("span", "right", "sent")


def mean_pool(h: torch.Tensor, pos: list[int]) -> torch.Tensor:
    """h [L+1, T, D] -> [L+1, D]; NaN when the pool is empty (e.g. no tokens after a sentence-final span)."""
    return h[:, pos].mean(1) if pos else torch.full_like(h[:, 0], float("nan"))


@torch.no_grad()
def forward(tok, model, rows: pd.DataFrame):
    ids, offs = encode_with_offsets(tok, rows.sentence.tolist())
    n = max(map(len, ids))
    pad = tok.pad_token_id
    dev = device(model)
    input_ids = torch.tensor([i + [pad] * (n - len(i)) for i in ids], device=dev)
    mask = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i in ids], device=dev)   # right padding
    out = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
    logp = torch.log_softmax(out.logits.float(), dim=-1)
    hs = torch.stack(out.hidden_states, dim=1)                # [B, L+1, T, D]
    recs, pooled = [], {p: [] for p in POOLS}
    for b, r in enumerate(rows.itertuples()):
        seq = ids[b]
        pos = span_token_positions(offs[b], int(r.span_start), int(r.span_end))
        right = list(range(pos[-1] + 1, len(seq))) if pos else []
        tok_lp = [logp[b, p - 1, seq[p]].item() for p in range(1, len(seq))]      # lp of token p given < p
        surpr = lambda ps: [-tok_lp[p - 1] for p in ps if 1 <= p < len(seq)]       # noqa: E731
        sp = surpr(pos)
        post = surpr(right[:POST])
        recs.append({"variant_id": r.variant_id, "item_id": r.item_id, "idiom_id": r.idiom_id,
                     "condition": r.condition, "frame": r.frame, "source": r.source,
                     "n_tok_span": len(pos), "n_tok_right": len(right), "n_tok_sent": len(seq) - 1,
                     "span_surprisal_sum": float(np.sum(sp)) if sp else np.nan,
                     "span_surprisal_mean": float(np.mean(sp)) if sp else np.nan,
                     "span_first_surprisal": sp[0] if sp else np.nan,
                     "post_surprisal_sum": float(np.sum(post)) if post else np.nan,
                     "post_surprisal_mean": float(np.mean(post)) if post else np.nan,
                     "sent_surprisal_mean": float(np.mean([-x for x in tok_lp]))})
        h = hs[b].float()                                     # [L+1, T, D]
        pooled["span"].append(mean_pool(h, pos))
        pooled["right"].append(mean_pool(h, right))
        pooled["sent"].append(mean_pool(h, list(range(1, len(seq)))))   # without the start token
    return recs, {p: torch.stack(v) for p, v in pooled.items()}       # each [B, L+1, D]


def cos_layers(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist()


def bf16_bits(x: torch.Tensor) -> np.ndarray:
    """float tensor -> its bfloat16 bit pattern as uint16 (numpy has no bfloat16)."""
    return x.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)


def process_shard(tok, model, df: pd.DataFrame, items_per_batch: int, save_acts: bool = False):
    recs, feats, feat_ids = [], [], []
    acts = {p: [] for p in POOLS}
    items = df.item_id.drop_duplicates().tolist()
    for i in range(0, len(items), items_per_batch):
        rows = df[df.item_id.isin(items[i:i + items_per_batch])].reset_index(drop=True)
        r, pooled = forward(tok, model, rows)
        if save_acts:
            for p in POOLS:
                acts[p].append(bf16_bits(pooled[p]))          # rows in the same order as r
        for _, g in rows.groupby("item_id"):
            idx = g.index.tolist()
            by_cond = dict(zip(g.condition, idx))
            for k in idx:
                rec = r[k]
                for ref in ("ORIG", "NEUTRAL"):
                    j = by_cond.get(ref)
                    if j is None or (ref == "NEUTRAL" and rec["frame"] != "neutral"):
                        continue
                    for p in POOLS:
                        rec[f"cos_{p}_vs_{ref}"] = cos_layers(pooled[p][k], pooled[p][j])
                if rec["condition"] in E4_CONDITIONS and rec["n_tok_span"]:
                    feats.append(pooled["span"][k].float().cpu().numpy())
                    feat_ids.append(rec["variant_id"])
        recs += r
    return recs, feats, feat_ids, ({p: np.concatenate(v) for p, v in acts.items()} if save_acts else None)


def shard_done(d, k: int, save_acts: bool) -> bool:
    if not (d / f"shard_{k:04d}.parquet").exists():
        return False
    a = d / "acts"
    return not save_acts or (a / f"shard_{k:04d}.npz").exists() or (a / f"shard_{k:04d}.uploaded").exists()


def run_model(name: str, df: pd.DataFrame, items_per_batch: int, shard_items: int, tag: str,
              save_acts: bool = False) -> None:
    short, hf_id = resolve(name)
    d = out_dir(short) / f"lm_probe{tag}"
    d.mkdir(parents=True, exist_ok=True)
    if save_acts:
        (d / "acts").mkdir(exist_ok=True)
    items = df.item_id.drop_duplicates().sort_values().tolist()
    shards = [items[i:i + shard_items] for i in range(0, len(items), shard_items)]
    todo = [k for k in range(len(shards)) if not shard_done(d, k, save_acts)]
    print(f"[E2-E4] {short}: {len(todo)} of {len(shards)} shards to run", flush=True)
    if not todo:
        return
    tok, model = load_model(hf_id)
    write_meta(short, hf_id, model, {"lm_probe_post_tokens": POST, "lm_probe_pools": POOLS})
    t0 = time.time()
    for k in todo:
        part = df[df.item_id.isin(shards[k])]
        ipb = items_per_batch
        while True:
            try:
                recs, feats, feat_ids, acts = process_shard(tok, model, part, ipb, save_acts)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if ipb == 1:
                    raise
                ipb = max(1, ipb // 2)
                print(f"  OOM -> {ipb} items per batch", flush=True)
        if acts is not None:
            tmp = d / "acts" / f"shard_{k:04d}.tmp.npz"
            np.savez(tmp, **acts, variant_id=np.array([r["variant_id"] for r in recs]),
                     condition=np.array([r["condition"] for r in recs]))
            tmp.rename(d / "acts" / f"shard_{k:04d}.npz")          # complete file or none
        if feats:
            # float16 halves the disk, but Gemma-3 models have "massive activations" above the float16 range
            # (65 504) that would turn into inf -> such shards are kept in float32
            X = np.stack(feats)
            X16 = X.astype(np.float16)
            np.savez(d / f"shard_{k:04d}.npz", X=X16 if np.isfinite(X16).all() else X, variant_id=np.array(feat_ids))
        pd.DataFrame(recs).to_parquet(d / f"shard_{k:04d}.parquet", index=False)   # written last = shard done
        print(f"  shard {k + 1}/{len(shards)}  {time.time() - t0:.0f}s", flush=True)
    free(model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--items-per-batch", type=int, default=4, help="sentences per forward pass = items x ~11 variants")
    ap.add_argument("--shard-items", type=int, default=64)
    ap.add_argument("--limit-items", type=int, help="only N random sentences (quick test)")
    ap.add_argument("--save-acts", action="store_true", default=os.environ.get("SAVE_ACTS") == "1",
                    help="also save all pooled hidden states (see the docstring)")
    a = ap.parse_args()
    data = limit_items(working(load_full()), a.limit_items)
    if "idiom_id" not in data:
        data["idiom_id"] = data.item_id
    tag = f"_limit{a.limit_items}" if a.limit_items else ""
    for m in a.models:
        run_model(m, data, a.items_per_batch, a.shard_items, tag, a.save_acts)

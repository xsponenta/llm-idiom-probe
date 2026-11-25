"""Shared helpers for the GPU experiments: model registry, data, questions, model loading, span alignment."""
import gc
import json
import os
import random
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("IDIOM_DATA_DIR", ROOT / "data" / "processed"))
RESULTS = Path(os.environ.get("IDIOM_RESULTS", ROOT / "data" / "results" / "gpu"))
SEED = 42

# short name -> HF id. Gemma and Llama come from the ungated unsloth mirrors (same bf16 weights as
# google/gemma-3-*-it and meta-llama/Llama-3.1-8B-Instruct), so no HF token is needed.
MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",                                  # sizes: bf16 safetensors
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",                                  # 4.1 GB
    "qwen3-4b": "Qwen/Qwen3-4B",                                      # 8.0 GB
    "qwen3-8b": "Qwen/Qwen3-8B",                                      # 16.4 GB
    "qwen3-14b": "Qwen/Qwen3-14B",                                    # 29.5 GB
    "qwen3-32b": "Qwen/Qwen3-32B",                                    # 65.5 GB: needs a >= 72 GB GPU
    "gemma3-4b": "unsloth/gemma-3-4b-it",                             # 8.6 GB
    "mamaylm-4b": "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0",      # Ukrainian adaptation of gemma3-4b
    "gemma3-12b": "unsloth/gemma-3-12b-it",                           # 24.4 GB
    "mamaylm-12b": "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0",    # Ukrainian adaptation of gemma3-12b
    "lapa-12b": "lapa-llm/lapa-v0.1.2-instruct",                      # Ukrainian, Gemma-3-12B based
    "llama3.1-8b": "unsloth/Llama-3.1-8B-Instruct",                   # 16.1 GB
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",           # 24.5 GB (HF-format shards)
}

CONDITIONS = ["ORIG", "SYN_LIT", "SYN_LIT_MATCHED", "SYN_IDIOM", "ANT_LIT", "ANT_IDIOM", "RAND", "PARTIAL",
              "NEUTRAL", "NEUTRAL_SYN_LIT", "NEUTRAL_ANT_LIT", "LIT_CONTEXT"]
NONE = {"uk": "жодне з наведених значень", "en": "none of these meanings"}
LETTERS = "ABCDEF"


def resolve(name: str) -> tuple[str, str]:
    """'qwen3-1.7b' -> ('qwen3-1.7b', 'Qwen/Qwen3-1.7B'); a raw HF id is accepted too."""
    if name in MODELS:
        return name, MODELS[name]
    return name.split("/")[-1].lower(), name


def out_dir(short: str) -> Path:
    d = RESULTS / short
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- data
def load_full() -> pd.DataFrame:
    """The frozen dataset (all rows, incl. flagged ones: question building must see the same rows as evaluate.py)."""
    for name in ("substitutions_v1.parquet", "substitutions.parquet"):
        p = DATA_DIR / name
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"put substitutions_v1.parquet into {DATA_DIR}")


def working(df: pd.DataFrame) -> pd.DataFrame:
    """Working sample: variants without grammar flags (8 593 rows in v1)."""
    if "flag_grammar" in df:
        df = df[~df.flag_grammar.fillna(False).astype(bool)]
    return df.reset_index(drop=True)


def limit_items(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if not n:
        return df
    keep = df.item_id.drop_duplicates().sample(frac=1, random_state=SEED).head(n)
    return df[df.item_id.isin(keep)].reset_index(drop=True)


def questions(df_full: pd.DataFrame) -> list[dict]:
    """T1/T2 questions built exactly as in evaluate.questions() (same seed, same option order), so the GPU results
    are comparable with the OpenRouter ones. Flagged variants are dropped afterwards, not before (keeps the RNG)."""
    df = df_full
    rng = random.Random(SEED)
    meanings = {lang: g.drop_duplicates("item_id").meaning_here.dropna().tolist() for lang, g in df.groupby("language")}
    excluded = set(df_full.variant_id[df_full.flag_grammar.fillna(False).astype(bool)]) if "flag_grammar" in df else set()
    qs = []
    for r in df.itertuples():
        marked = f"{r.sentence[:r.span_start]}[[{r.replacement}]]{r.sentence[r.span_end:]}"
        other = rng.choice([m for m in meanings[r.language] if m != r.meaning_here] or ["-"])
        opts = [("same", r.meaning_here), ("opposite", r.opposite_meaning), ("literal", r.literal_reading),
                ("near", getattr(r, "near_meaning", None)), ("other", other)]
        opts = [o for o in opts if isinstance(o[1], str) and o[1]]
        rng.shuffle(opts)
        opts.append(("none", NONE[r.language]))
        gold_kind = {"same": "same", "opposite": "opposite", "none": "none", "literal": "literal",
                     "partial": None}[r.gold]
        gold = next((LETTERS[i] for i, (k, _) in enumerate(opts) if k == gold_kind), None)
        base = {"item_id": r.item_id, "idiom_id": getattr(r, "idiom_id", r.item_id), "lang": r.language,
                "condition": r.condition, "source": getattr(r, "source", None), "variant_id": r.variant_id}
        if r.variant_id in excluded:
            continue
        if gold or r.gold == "partial":
            qs.append({**base, "qid": f"T1|{r.item_id}|{r.condition}", "task": "T1", "gold": gold,
                       "kinds": {LETTERS[i]: k for i, (k, _) in enumerate(opts)},
                       "sentence": marked, "options": {LETTERS[i]: t for i, (_, t) in enumerate(opts)}})
        if r.condition not in ("ORIG", "NEUTRAL", "LIT_CONTEXT"):
            qs.append({**base, "qid": f"T2|{r.item_id}|{r.condition}", "task": "T2",
                       "gold": None if r.gold == "partial" else "yes" if r.gold == "same" else "no",
                       "A": r.orig_sentence, "B": r.sentence})
    return qs


# ---------------------------------------------------------------- model
def pinned_revision(hf_id: str) -> str | None:
    """Commit recorded in hf_manifest.json (verified download), so an offline load finds exactly those files."""
    p = Path(__file__).with_name("hf_manifest.json")
    if not p.exists():
        return None
    return next((m["revision"] for m in json.loads(p.read_text()).values() if m["hf_id"] == hf_id), None)


def load_model(hf_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = pinned_revision(hf_id)
    tk = {"revision": rev}
    if "mistral" in hf_id.lower():
        tk["fix_mistral_regex"] = True      # the hub tokenizer ships a wrong pre-tokenizer regex (transformers warns)
    tok = AutoTokenizer.from_pretrained(hf_id, **tk)
    # "auto" = all visible GPUs (one queue per GPU via CUDA_VISIBLE_DEVICES; qwen3-32b spreads over both)
    kw = {"device_map": "auto" if torch.cuda.is_available() else "cpu", "revision": rev}
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, **kw)        # transformers >= 4.56
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, **kw)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


def free(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def device(model) -> torch.device:
    return next(model.parameters()).device


def chat_prompt(tok, user: str) -> str:
    """Single-turn chat prompt ending where the assistant's answer starts. Qwen3: thinking switched off."""
    msgs = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def start_token_id(tok) -> int:
    """Document-start token for raw text: BOS, or <|endoftext|> for models without BOS (Qwen)."""
    if tok.bos_token_id is not None:
        return tok.bos_token_id
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    return eot if isinstance(eot, int) and eot != tok.unk_token_id else tok.eos_token_id


def encode_with_offsets(tok, texts: list[str]) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    """Tokenize raw sentences (no chat template) with char offsets; exactly one start token first."""
    enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
    start = start_token_id(tok)
    ids = [[start] + list(i) for i in enc["input_ids"]]
    offs = [[(0, 0)] + [tuple(o) for o in off] for off in enc["offset_mapping"]]
    return ids, offs


def span_token_positions(offsets: list[tuple[int, int]], a: int, b: int) -> list[int]:
    """Token positions whose characters overlap the span [a, b)."""
    return [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]


def answer_token_ids(tok, surface_forms: list[str]) -> list[int]:
    """First-token ids of the surface forms of one answer ("A", " A", "так", "Так", ...), deduplicated."""
    ids = []
    for s in surface_forms:
        t = tok.encode(s, add_special_tokens=False)
        if t and t[0] not in ids:
            ids.append(t[0])
    return ids


def write_meta(short: str, hf_id: str, model, extra: dict) -> None:
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    meta = {"model": short, "hf_id": hf_id, "n_layers": cfg.num_hidden_layers, "d_model": cfg.hidden_size,
            "dtype": str(next(model.parameters()).dtype), "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu", **extra}
    p = out_dir(short) / "meta.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    p.write_text(json.dumps({**old, **meta}, indent=2))


def bf16_to_float32(bits):
    """uint16 bfloat16 bit patterns (lm_probe.py --save-acts) -> float32 numpy array (exact)."""
    import numpy as np
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)

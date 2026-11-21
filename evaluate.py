"""Label-free evaluation on the substitution dataset: the gold answer follows from the condition.

    .venv/bin/python evaluate.py run --models openai/gpt-6-luna google/gemma-3-4b-it
    .venv/bin/python evaluate.py report

T1 meaning (multiple choice). "What does the [[marked]] part mean in this sentence?"
   options: meaning_here / opposite_meaning / literal_reading / meaning of another idiom / none of these
   gold:    same -> meaning_here, opposite -> opposite_meaning, none (RAND) -> none of these
T2 paraphrase. "Do sentences A (original) and B (variant) mean the same?"   gold: same -> yes, else no

Output: data/results/eval_answers.jsonl (one line per model x question), report printed per condition.
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from agent_label import API, SPENT_LOG, headers

DATA = Path("data/processed")
RES = Path("data/results")
ANSWERS = RES / "eval_answers.jsonl"
SEED = 42
BATCH = 10
WORKERS = 4
DEFAULT_MODELS = ["google/gemma-3-4b-it", "mistralai/ministral-3b-2512", "meta-llama/llama-3.1-8b-instruct",
                  "mistralai/mistral-nemo", "google/gemma-3-12b-it", "openai/gpt-6-luna"]   # small ... strong
NONE = {"uk": "жодне з наведених значень", "en": "none of these meanings"}

SYSTEM = {
    "T1": """For each item, read the sentence and choose what the part marked with [[ ]] means in this sentence.
Pick exactly one option letter. Answer with JSON only: {"answers": [{"id": "...", "answer": "B"}, ...]}""",
    "T2": """For each item, decide whether sentence A and sentence B convey the same meaning
(small wording differences are fine; a changed, opposite or nonsensical meaning is not the same).
Answer with JSON only: {"answers": [{"id": "...", "answer": "yes"}, ...]}""",
}


def questions():
    df = pd.read_parquet(DATA / "substitutions.parquet")
    rng = random.Random(SEED)
    meanings = {lang: g.drop_duplicates("item_id").meaning_here.dropna().tolist() for lang, g in df.groupby("language")}
    qs = []
    for r in df.itertuples():
        marked = f"{r.sentence[:r.span_start]}[[{r.replacement}]]{r.sentence[r.span_end:]}"
        other = rng.choice([m for m in meanings[r.language] if m != r.meaning_here] or ["-"])
        opts = [("same", r.meaning_here), ("opposite", r.opposite_meaning), ("literal", r.literal_reading),
                ("near", getattr(r, "near_meaning", None)), ("other", other)]   # near = close but wrong
        opts = [o for o in opts if isinstance(o[1], str) and o[1]]
        rng.shuffle(opts)
        opts.append(("none", NONE[r.language]))
        letters = "ABCDEF"
        gold_kind = {"same": "same", "opposite": "opposite", "none": "none", "literal": "literal",
                     "partial": None}[r.gold]                 # PARTIAL: no gold, we look at what is chosen
        gold = next((letters[i] for i, (k, _) in enumerate(opts) if k == gold_kind), None)
        if gold or r.gold == "partial":
            qs.append({"qid": f"T1|{r.item_id}|{r.condition}", "task": "T1", "lang": r.language,
                       "condition": r.condition, "gold": gold,
                       "kinds": {letters[i]: k for i, (k, _) in enumerate(opts)},
                       "prompt": {"sentence": marked,
                                  "options": {letters[i]: t for i, (_, t) in enumerate(opts)}}})
        if r.condition not in ("ORIG", "NEUTRAL", "LIT_CONTEXT"):   # T2 compares within the same frame
            qs.append({"qid": f"T2|{r.item_id}|{r.condition}", "task": "T2", "lang": r.language,
                       "condition": r.condition,
                       "gold": None if r.gold == "partial" else "yes" if r.gold == "same" else "no",
                       "prompt": {"A": r.orig_sentence, "B": r.sentence}})
    return qs


def call(model, task, batch):
    payload = [{"id": q["qid"], **q["prompt"]} for q in batch]
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "reasoning": {"effort": "low", "exclude": True}, "usage": {"include": True}, "max_tokens": 6000,
            "messages": [{"role": "system", "content": SYSTEM[task]},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
    for _ in range(3):
        try:
            r = requests.post(f"{API}/chat/completions", headers=headers(), json=body, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                continue
            r.raise_for_status()
            d = r.json()
            text = d["choices"][0]["message"]["content"] or ""
            # small models often return broken / several JSON objects: read the id-answer pairs directly
            ans = dict(re.findall(r'"id"\s*:\s*"([^"]+)"\s*,\s*"answer"\s*:\s*"([^"]*)"', text))
            return ans, (d.get("usage") or {}).get("cost", 0) or 0
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            print("  ! retry:", repr(e)[:120])
    return {}, 0


def run(models, max_usd=1.0, limit=None):
    RES.mkdir(parents=True, exist_ok=True)
    qs = questions()
    done = set()
    if ANSWERS.exists():
        done = {(a["model"], a["qid"]) for a in map(json.loads, open(ANSWERS, encoding="utf-8"))}
    spent = 0.0
    for model in models:
        todo = [q for q in qs if (model, q["qid"]) not in done][:limit]
        batches = [b for task in ["T1", "T2"] for t in [[q for q in todo if q["task"] == task]]
                   for b in (t[i:i + BATCH] for i in range(0, len(t), BATCH))]
        print(f"{model}: {len(todo)} questions in {len(batches)} requests")
        with ThreadPoolExecutor(WORKERS) as pool, open(ANSWERS, "a", encoding="utf-8") as out:
            futs = {pool.submit(call, model, b[0]["task"], b): b for b in batches}
            for k, f in enumerate(as_completed(futs), 1):
                ans, cost = f.result()
                spent += cost
                with open(SPENT_LOG, "a") as log:
                    log.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "task": "evaluate",
                                          "model": model, "cost": cost}) + "\n")
                for q in futs[f]:
                    if q["qid"] in ans:
                        a = ans[q["qid"]].upper()[:1] if q["task"] == "T1" else ans[q["qid"]].lower()[:3].strip()
                        out.write(json.dumps({"model": model, "qid": q["qid"], "task": q["task"], "lang": q["lang"],
                                              "condition": q["condition"], "gold": q["gold"], "answer": a,
                                              "chosen_kind": q.get("kinds", {}).get(a)}, ensure_ascii=False) + "\n")
                out.flush()
                if k % 20 == 0 or k == len(futs):
                    print(f"  {k}/{len(futs)} requests, total ${spent:.4f}")
                if spent > max_usd:
                    print("  stop: budget reached")
                    return


def report():
    a = pd.DataFrame(map(json.loads, open(ANSWERS, encoding="utf-8")))
    a["model"] = a.model.str.split("/").str[-1]
    desc = a[a.gold.isna()]                                  # PARTIAL: descriptive only
    a = a[a.gold.notna()].copy()
    a["correct"] = a.answer == a.gold
    order = ["ORIG", "SYN_LIT", "SYN_LIT_MATCHED", "SYN_IDIOM", "ANT_LIT", "ANT_IDIOM", "RAND", "LIT_CONTEXT",
             "NEUTRAL", "NEUTRAL_SYN_LIT", "NEUTRAL_ANT_LIT"]
    for task, title in [("T1", "T1 meaning choice: accuracy"), ("T2", "T2 same meaning as original: accuracy")]:
        t = a[a.task == task]
        if t.empty:
            continue
        tab = t.pivot_table(index=["lang", "model"], columns="condition", values="correct", aggfunc="mean")
        print(f"\n=== {title}  (n per cell ≈ {int(t.groupby(['lang', 'model', 'condition']).size().median())})")
        print((tab[[c for c in order if c in tab]] * 100).round(0).to_string())
    t1 = a[(a.task == "T1") & ~a.correct]
    if not t1.empty:
        print("\n=== T1 errors: what was chosen instead (share of errors)")
        print(pd.crosstab([t1.lang, t1.condition], t1.chosen_kind, normalize="index").round(2).to_string())
    if not desc.empty:
        print("\n=== PARTIAL (one idiom word changed): what models think it means, share")
        print(pd.crosstab(desc[desc.task == "T1"].model, desc[desc.task == "T1"].chosen_kind,
                          normalize="index").round(2).to_string())
        t2 = desc[desc.task == "T2"]
        if not t2.empty:
            print("\nPARTIAL, T2: share 'same meaning as original'")
            print(t2.groupby("model").answer.apply(lambda s: (s == "yes").mean()).round(2).to_string())
    pd.concat([a, desc]).to_parquet(RES / "eval_answers.parquet", index=False)
    heatmap(a, order)


def heatmap(a, order):
    """Accuracy per model x condition, one panel per task (single-hue sequential scale, values printed)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#eaf2fc", "#86b6ef", "#2a78d6", "#104281"])
    models = (a.groupby("model").correct.mean().sort_values().index.tolist())   # weakest on top
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(models) + 1.8),
                             gridspec_kw={"width_ratios": [6, 5]})
    for ax, (task, title) in zip(axes, [("T1", "T1: what does the span mean?"),
                                        ("T2", "T2: same meaning as the original?")]):
        t = a[(a.task == task) & (a.lang == "uk")]
        cols = [c for c in order if c in t.condition.unique()]
        tab = t.pivot_table(index="model", columns="condition", values="correct", aggfunc="mean").reindex(
            index=models, columns=cols) * 100
        ax.imshow(tab.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v = tab.values[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10,
                            color="white" if v > 60 else "#1a1a1a")
        ax.set_xticks(range(len(cols)), cols, fontsize=9)
        ax.set_yticks(range(len(models)), models if ax is axes[0] else [""] * len(models), fontsize=9)
        ax.set_title(title, fontsize=11, loc="left")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Ukrainian idioms: accuracy, % (gold answer follows from the substitution)", x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(RES / "accuracy_uk.png", dpi=160)
    print(f"\nchart -> {RES / 'accuracy_uk.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.models, a.max_usd, a.limit)
    else:
        report()

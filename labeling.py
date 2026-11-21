"""Labels live separately from the dataset, in data/labels/<task>/:

    items.jsonl         what to label (made by `export`)
    INSTRUCTIONS.md     the task for agents (same wording a human sees)
    human.jsonl         your labels (made by `label`)
    <model>.jsonl       one file per LLM annotator, e.g. gpt-6-luna.jsonl, gemini-3.1-flash-lite.jsonl
    consensus.parquet   final label per item (made by `consensus`)

Every label file is JSONL: {"id": ..., "label": ..., "note": ...}; the file name is the annotator.

    .venv/bin/python labeling.py export                  # all tasks -> items.jsonl + INSTRUCTIONS.md
    .venv/bin/python labeling.py label idiomatic         # label by hand in the terminal
    .venv/bin/python labeling.py label usage --disputed  # only items the agents disagree on
    (labeling is blind: agent labels are hidden unless --show-agents; keep it blind for the audit sample)
    .venv/bin/python labeling.py consensus               # merge annotators, print agreement

Consensus rule: human label wins; otherwise ≥2 agents that all give the same (non-"?") label;
otherwise the item is "disputed" and waits for a human.
"""
import json
import random
import sys
from datetime import datetime
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

DATA = Path("data/processed")
LABELS = Path("data/labels")
SEED = 42
MIN_AGENTS = 2
N_KOBZA = 500          # idioms in the manual Kobza set (random sample, one sentence each)

TASKS = {
    "idiomatic": {
        "question": "Is this expression an idiom (meaning is figurative / not the sum of its words)?",
        "labels": {"1": "idiom", "0": "not an idiom: free word combination, term, collocation or plain word",
                   "?": "unsure"},
    },
    "usage": {
        "question": "In this sentence, is the marked [[span]] the idiom, and is it used figuratively or literally?",
        "labels": {"f": "figurative: idiomatic meaning", "l": "literal: the words mean what they say",
                   "x": "wrong match: the span is not this idiom / boundaries badly wrong", "?": "unsure"},
    },
    "kobza": {
        "question": "UK corpus sentence: is the marked [[span]] an idiom here, and how is it used?",
        "labels": {"f": "figurative: an idiom used with its idiomatic meaning",
                   "l": "literal: the same words, but meaning what they say",
                   "n": "not an idiom: the expression itself is a plain phrase / term / collocation",
                   "x": "wrong match: the span is not this expression / boundaries badly wrong",
                   "m": "mention: the idiom is listed / quoted / explained, not used in a sentence",
                   "?": "unsure"},
    },
    "subst_check": {
        "question": "Is the variant correct for its condition (grammatical in context, and the meaning matches what is expected)?",
        "labels": {"1": "ok", "0": "problem: ungrammatical, wrong meaning for the condition, invented idiom, "
                   "idiomatic reading in LIT_CONTEXT, or (ORIG) wrong idiom span", "?": "unsure"},
    },
    "pair": {
        "question": "Do the UK idiom and the EN idiom have the same figurative meaning (good translation pair)?",
        "labels": {"1": "same meaning", "0": "different meaning (or only partly overlapping)", "?": "unsure"},
    },
}


# ------------------------------------------------------------------ items
def make_items(tasks=None):
    data = pd.read_parquet(DATA / "idiom_sentences.parquet")
    found = data[data.span_start.notna()]
    rng = random.Random(SEED)

    def examples(df, n=2):
        rows = df.sort_values("ok_design_a", ascending=False).head(n)
        return [m[:400] for m in rows.marked]

    # idiomatic: every SimIdioms idiom with ≥1 sentence (MAGPIE idioms are curated, skip them)
    idiomatic = []
    for iid, g in found[found.cluster_id.notna() & (found.source != "magpie")].groupby("idiom_id"):
        r = g.iloc[0]
        idiomatic.append({"id": iid, "language": r.language, "idiom": g.idiom.mode().iat[0],
                          "meaning": r.meaning, "examples": examples(g)})

    # usage: sentences where usage is unknown (Kobza) or the span came from lemma matching
    need = found[(found.source == "kobza") | ((found.source == "simidioms") & (found.method == "lemma"))]
    usage = [{"id": r.sentence_id, "language": r.language, "idiom": r.idiom, "meaning": r.meaning,
              "source": r.source, "sentence": r.marked} for r in need.itertuples()]

    # pair: SimIdioms clusters with design-A sentences in both languages
    ok = found[found.ok_design_a & found.cluster_id.notna() & (found.source == "simidioms")]
    pair = []
    for cid, g in ok.groupby("cluster_id"):
        side = {lang: g[g.language == lang] for lang in ["uk", "en"]}
        if any(s.empty for s in side.values()):
            continue
        item = {"id": cid}
        for lang, s in side.items():
            r = s.sort_values(["method", "n_words"]).iloc[0]   # exact first, then shortest; idiom from the same row
            item[f"{lang}_idiom"] = r.idiom
            item[f"{lang}_meaning"] = r.meaning
            item[f"{lang}_sentence"] = r.marked[:400]
        pair.append(item)

    # kobza: one corpus sentence per UK idiom, for manual labeling (idiom-ness + usage in one pass).
    # Idioms the agents unanimously called "not an idiom" are left out to save time.
    not_idiom = set()
    cons = LABELS / "idiomatic" / "consensus.parquet"
    if cons.exists():
        c = pd.read_parquet(cons)
        not_idiom = set(c[(c.decided_by == "agents_unanimous") & (c.label == "0")].id)
    k = found[(found.source == "kobza") & found.ok_design_a & ~found.idiom_id.isin(not_idiom)]
    k = k.assign(_exact=(k.method != "exact").astype(int), _r=[rng.random() for _ in range(len(k))])
    best = k.sort_values(["_exact", "_r"]).groupby("idiom_id").head(1)
    kobza = [{"id": r.sentence_id, "idiom_id": r.idiom_id, "language": "uk", "idiom": r.idiom,
              "meaning": r.meaning, "method": r.method, "doc_id": r.kobza_doc_id,
              "sentence_plain": r.sentence, "sentence": r.marked} for r in best.itertuples()]

    items = {"idiomatic": idiomatic, "usage": usage, "pair": pair, "kobza": kobza,
             "subst_check": subst_check_items() if (DATA / "substitutions.parquet").exists() else []}
    for task, rows in items.items():
        if tasks and task not in tasks:
            continue
        if task != "subst_check":                         # subst_check keeps its own order (sample first)
            rng.shuffle(rows)
        if task == "kobza":
            rows = rows[:N_KOBZA]
            add_context(rows)
        d = LABELS / task
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "items.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (d / "INSTRUCTIONS.md").write_text(instructions(task), encoding="utf-8")
        print(f"{task:10} {len(rows):5} items -> {d / 'items.jsonl'}")


EXPECT = {
    "ORIG": "оригінал: чи правильно виділено ідіому (межі спану, це справді вона)?",
    "SYN_LIT": "те саме значення, звичайними словами (не ідіома)",
    "SYN_LIT_MATCHED": "те саме значення, звичайними словами (не ідіома)",
    "SYN_IDIOM": "інша СПРАВЖНЯ ідіома з тим самим значенням",
    "ANT_LIT": "протилежне значення, звичайними словами",
    "ANT_IDIOM": "справжня ідіома з протилежним значенням",
    "PARTIAL": "та сама ідіома з ОДНИМ заміненим словом; граматично, і це НЕ відома ідіома",
    "LIT_CONTEXT": "нове речення: ті самі слова в ПРЯМОМУ значенні, ідіоматичне прочитання неможливе",
}
N_SUBST_SAMPLE = 50     # sentences the human checks (all their variants); the judge checks everything


def subst_check_items():
    """Every variant except RAND and exact duplicates; the 50 sampled sentences come first, grouped."""
    df = pd.read_parquet(DATA / "substitutions.parquet")
    df = df[(df.condition != "RAND") & ~df.flag_duplicate]
    orig = df[df.condition == "ORIG"].set_index("item_id")
    ids = sorted(df.item_id.unique())
    sample = set(random.Random(SEED).sample(ids, N_SUBST_SAMPLE))
    order = list(EXPECT)
    df = df.assign(_s=~df.item_id.isin(sample), _c=df.condition.map(order.index)).sort_values(["_s", "item_id", "_c"])
    rows = []
    for r in df.itertuples():
        o = orig.loc[r.item_id]
        rows.append({"id": f"{r.item_id}|{r.condition}", "condition": r.condition, "expect": EXPECT[r.condition],
                     "idiom": r.idiom, "meaning_here": r.meaning_here, "in_human_sample": r.item_id in sample,
                     "original": f"{o.sentence[:o.span_start]}[[{o.replacement}]]{o.sentence[o.span_end:]}",
                     "variant": f"{r.sentence[:r.span_start]}[[{r.replacement}]]{r.sentence[r.span_end:]}"})
    return rows


def add_context(rows, chars=300):
    """Text around each sentence from its Kobza document (for the human; agents see the sentence only)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want = {r["doc_id"] for r in rows}
    text_of = {}
    for path in [Path("data/raw/kobza/kobza_first_10000.parquet"), Path("data/raw/kobza/000_ukr_corpus_00000.parquet")]:
        if not path.exists():
            continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=20_000, columns=["id", "text"]):
            hit = batch.filter(pc.is_in(batch.column("id"), value_set=pa_array(want)))
            text_of.update(zip(hit.column("id").to_pylist(), hit.column("text").to_pylist()))
            if want <= text_of.keys():
                break
    for r in rows:
        text = text_of.get(r["doc_id"], "")
        i = text.find(r["sentence_plain"])
        if i < 0:
            r["context_before"], r["context_after"] = "", ""
            continue
        before = text[max(0, i - chars):i]
        after = text[i + len(r["sentence_plain"]):i + len(r["sentence_plain"]) + chars]
        r["context_before"] = " ".join(before.split()[1:] if i > chars else before.split())
        r["context_after"] = " ".join(after.split()[:-1] if len(after) == chars else after.split())
    print(f"context found for {sum(bool(r['context_before'] or r['context_after']) for r in rows)}/{len(rows)}")


def pa_array(values):
    import pyarrow as pa
    return pa.array(sorted(values))


def instructions(task):
    t = TASKS[task]
    labels = "\n".join(f'- `"{k}"`: {v}' for k, v in t["labels"].items())
    return f"""# Labeling task: {task}

{t["question"]}

Read `items.jsonl` in this folder (one JSON object per line). `[[...]]` marks the idiom span.
For every item write one line to `<your-name>.jsonl` in this folder:

    {{"id": "<item id>", "label": "<label>", "note": "<short reason, optional>"}}

Allowed labels:
{labels}

Judge each item on its own. Use `"?"` rather than guessing. Do not skip items; keep the ids exactly.
"""


# ------------------------------------------------------------------ io
def load_items(task):
    with open(LABELS / task / "items.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_labels(task):
    """{annotator: {id: label}}; later lines override earlier ones (re-labeling)."""
    out = {}
    for p in sorted((LABELS / task).glob("*.jsonl")):
        if p.stem == "items":
            continue
        labs = {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if str(r.get("label")) in TASKS[task]["labels"]:
                    labs[r["id"]] = str(r["label"])
        out[p.stem] = labs
    return out


# ------------------------------------------------------------------ consensus
def decide(labels_of_item):
    """labels_of_item: {annotator: label} -> (final_label, decided_by)."""
    if "human" in labels_of_item and labels_of_item["human"] != "?":
        return labels_of_item["human"], "human"
    agents = {a: l for a, l in labels_of_item.items() if a != "human"}
    votes = set(agents.values())
    if len(agents) >= MIN_AGENTS and len(votes) == 1 and "?" not in votes:
        return votes.pop(), "agents_unanimous"
    return None, "disputed" if agents else "unlabeled"


def consensus(task, verbose=True):
    items = load_items(task)
    labels = load_labels(task)
    rows = []
    for it in items:
        per = {a: labs[it["id"]] for a, labs in labels.items() if it["id"] in labs}
        final, by = decide(per)
        rows.append({"id": it["id"], "label": final, "decided_by": by, **{f"by_{a}": l for a, l in per.items()}})
    df = pd.DataFrame(rows)
    df.to_parquet(LABELS / task / "consensus.parquet", index=False)
    if verbose:
        print(f"\n=== {task}: {len(items)} items, annotators: {', '.join(labels) or 'none'}")
        print(df.decided_by.value_counts().to_string())
        print("final labels:", df.label.value_counts().to_dict())
        for a, b in combinations(labels, 2):        # pairwise raw agreement on shared items
            shared = set(labels[a]) & set(labels[b])
            if shared:
                agree = sum(labels[a][i] == labels[b][i] for i in shared) / len(shared)
                print(f"agreement {a} vs {b}: {agree:.0%} on {len(shared)} items")
        audit(labels)
    return df


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return c - h, c + h


def kappa(pairs):
    """Cohen's kappa for a list of (a, b) labels."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    cats = {x for p in pairs for x in p}
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def audit(labels):
    """Agents vs human on the items the human labeled (blind). Shows if agents share errors."""
    human = {i: l for i, l in labels.get("human", {}).items() if l != "?"}
    if not human:
        return
    agents = [a for a in labels if a != "human"]
    print(f"--- audit vs human ({len(human)} items)")
    for a in agents:
        pairs = [(human[i], labels[a][i]) for i in human if i in labels[a]]
        if pairs:
            acc = sum(h == x for h, x in pairs) / len(pairs)
            print(f"  {a:24} accuracy {acc:.0%}  kappa {kappa(pairs):.2f}  (n={len(pairs)})")
    # what the agents-only rule would have decided on these items
    rule = {i: decide({a: labels[a][i] for a in agents if i in labels[a]}) for i in human}
    unan = [(human[i], l) for i, (l, by) in rule.items() if by == "agents_unanimous"]
    if unan:
        k = sum(h == l for h, l in unan)
        lo, hi = wilson(k, len(unan))
        print(f"  unanimous agents correct: {k}/{len(unan)} = {k / len(unan):.0%}  (95% CI {lo:.0%}-{hi:.0%})")
        wrong = Counter((l, h) for h, l in unan if h != l)
        if wrong:
            print("  unanimous errors (agents said -> human said):", dict(wrong))
    disp = sum(by == "disputed" for _, by in rule.values())
    print(f"  disputed among audited: {disp}/{len(rule)}")


# ------------------------------------------------------------------ manual labeling
HL, DIM, BOLD, RESET = "\033[1;30;43m", "\033[2m", "\033[1m", "\033[0m"   # idiom on yellow, dim context


def highlight(marked):
    return marked.replace("[[", HL).replace("]]", RESET)


def show(task, it):
    print("\033[2J\033[H", end="")                     # clear screen
    if task == "pair":
        print(f"{it['id']}\n")
        print(f"{BOLD}UK: {it['uk_idiom']}{RESET}  —  {it['uk_meaning']}\n   {highlight(it['uk_sentence'])}\n")
        print(f"{BOLD}EN: {it['en_idiom']}{RESET}  —  {it['en_meaning']}\n   {highlight(it['en_sentence'])}")
        return
    if task == "subst_check":
        print(f"{BOLD}{it['idiom']}{RESET}  {DIM}— {it.get('meaning_here')}{RESET}\n")
        print(f"{BOLD}{it['condition']}{RESET}: {it['expect']}\n")
        if it["condition"] != "ORIG":
            print(f"   {DIM}оригінал:{RESET} {highlight(it['original'])}")
        print(f"   {DIM}варіант: {RESET} {highlight(it['variant'])}")
        return
    print(f"{BOLD}{it['idiom']}{RESET}")
    print(f"{DIM}значення: {it.get('meaning')}{RESET}\n")
    if it.get("context_before") is not None:
        print(f"   {DIM}…{it['context_before']}{RESET} {highlight(it['sentence'])} {DIM}{it['context_after']}…{RESET}")
    else:
        for s in it.get("examples", [it.get("sentence")]):
            print(f"   {highlight(s)}")


# Ukrainian ЙЦУКЕН layout -> the Latin key at the same place, so labels work in any keyboard layout
UK_LAYOUT = dict(zip("йцукенгшщзфівапролджєячсмитьбю№.", "qwertyuiopasdfghjkl;'zxcvbnm,.#/"))
DIGITS = {"1": "f", "2": "l", "3": "n", "4": "x", "5": "m", "0": "?", "/": "?"}


def normalize(key):
    key = key.lower()
    key = UK_LAYOUT.get(key, key)
    return DIGITS.get(key, key)


def getkey():
    """One key press, no Enter (falls back to input() outside a terminal)."""
    if not sys.stdin.isatty():
        return (sys.stdin.readline().strip() or "q")[0]
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def label(task, disputed_only=False, show_agents=False):
    items = load_items(task)
    labels = load_labels(task)
    done = labels.get("human", {})
    todo = [it for it in items if it["id"] not in done]
    if disputed_only:
        cons = consensus(task, verbose=False).set_index("id")
        todo = [it for it in todo if cons.at[it["id"], "decided_by"] == "disputed"]
    keys = TASKS[task]["labels"]
    digit_of = {v: d for d, v in DIGITS.items() if d != "/"}
    legend = "   ".join(f"{BOLD}{k}{RESET}{DIM}/{digit_of[k]}{RESET} {v.split(':')[0]}" if k in digit_of
                        else f"{BOLD}{k}{RESET} {v.split(':')[0]}" for k, v in keys.items())
    path = LABELS / task / "human.jsonl"
    history = []                                          # (index in todo) for undo

    if not sys.stdin.isatty():
        print("not a terminal: type the key and press Enter")
    k = 0
    while k < len(todo):
        it = todo[k]
        show(task, it)
        agents = {a: labs[it["id"]] for a, labs in labels.items() if a != "human" and it["id"] in labs}
        if show_agents and agents:
            print(f"\n   {DIM}agents: {agents}{RESET}")
        counts = Counter(done.values())
        print(f"\n{DIM}{'─' * 80}{RESET}")
        print(f"{TASKS[task]['question']}\n{legend}")
        print(f"{DIM}#  нотатка   b  назад (переробити попередню)   q  вийти        "
              f"[{len(done)}/{len(items)} готово | {dict(counts)}]{RESET}")
        note = ""
        raw = getkey()
        a = raw if raw in keys else normalize(raw)
        if a == "#":
            note = input("\nнотатка > ").strip()
            print("мітка: ", end="", flush=True)
            raw = getkey()
            a = raw if raw in keys else normalize(raw)
        if a == "q":
            break
        if a == "b" and history:
            k = history.pop()
            done.pop(todo[k]["id"], None)                 # re-label; the new line overrides the old one
            continue
        if a not in keys:
            print(f"\n  невідома клавіша {a!r}: натисни одну з {'/'.join(keys)} або цифру 1-5 (0 = ?)")
            getkey()
            continue
        with open(path, "a", encoding="utf-8") as out:
            out.write(json.dumps({"id": it["id"], "label": a, "note": note,
                                  "ts": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False) + "\n")
        done[it["id"]] = a
        history.append(k)
        k += 1
    print(f"\nзбережено в {path}: {len(done)} міток")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "consensus"
    if cmd == "export":
        make_items(sys.argv[2:] or None)
    elif cmd == "label":
        label(sys.argv[2], disputed_only="--disputed" in sys.argv, show_agents="--show-agents" in sys.argv)
    elif cmd == "consensus":
        for t in sys.argv[2:] or TASKS:
            if (LABELS / t / "items.jsonl").exists():
                consensus(t)
    else:
        print(__doc__)

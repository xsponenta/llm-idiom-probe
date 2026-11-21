"""Minimal-pair substitution dataset (design A): the idiom span is replaced, the context stays fixed.

    .venv/bin/python substitutions.py select --n 250                  # sentences per language
    .venv/bin/python substitutions.py generate --limit 10             # pilot
    .venv/bin/python substitutions.py generate                        # all (resumes)
    .venv/bin/python substitutions.py build                           # -> data/processed/substitutions.parquet
    .venv/bin/python substitutions.py show --n 5

Conditions (gold relation of the new sentence to the original meaning):
    ORIG        original idiom                         same
    SYN_LIT     plain words with the same meaning      same
    SYN_IDIOM   another idiom with the same meaning    same
    ANT_LIT     plain words with the opposite meaning  opposite
    ANT_IDIOM   an idiom with the opposite meaning     opposite
    RAND        random words, same number of words     none   (made here, no LLM)

The LLM only writes the replacement phrases (inflected to fit the unchanged context) plus two distractor
meanings for evaluation; the sentences are spliced here: sentence[:start] + replacement + sentence[end:].
"""
import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from agent_label import API, SPENT_LOG, headers

DATA = Path("data/processed")
WORK = Path("data/interim")
SEED = 42
MODEL = "anthropic/claude-sonnet-5.5"
BATCH = 10
N_PARSEME = 120
WORKERS = 4
GEN_FILE = WORK / "substitutions_llm.jsonl"
WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*")

CONDITIONS = {"ORIG": "same", "SYN_LIT": "same", "SYN_LIT_MATCHED": "same", "SYN_IDIOM": "same",
              "ANT_LIT": "opposite", "ANT_IDIOM": "opposite", "RAND": "none",
              "PARTIAL": "partial",        # one word of the idiom changed: no fixed gold, analysed descriptively
              "LIT_CONTEXT": "literal",    # NEW sentence, same words used literally (design B)
              "NEUTRAL": "same", "NEUTRAL_SYN_LIT": "same",   # NEW short sentence, context does not reveal meaning
              "NEUTRAL_ANT_LIT": "opposite"}
EXTRA_FILE = WORK / "substitutions_extra.jsonl"    # partial / lit_sentence / syn_lit_candidates (second LLM pass)
TOKENIZER = "data/raw/tokenizers/qwen3/tokenizer.json"   # proxy until the target models are fixed

SYSTEM = """You help build a minimal-pair dataset for a study of how language models understand idioms.

Each item has a sentence (Ukrainian or English) split into "left", "span" (the idiom) and "right",
plus the idiom's meaning (may be noisy or missing; trust the sentence). Your replacement goes EXACTLY between
"left" and "right": left + replacement + right must be a grammatical sentence. So never repeat words that
are already in "left" or "right" (e.g. if left ends with "the" or "as", do not start with "the"/"as"),
match case, tense, person, number and gender, and use the sentence's language. No extra punctuation.

- "syn_lit": 1-4 plain, non-idiomatic words with the same meaning as the idiom in this sentence.
  Must not reuse the idiom's content words.
- "syn_idiom": a different idiom of the same language with (nearly) the same meaning, inflected to fit; null if none exists.
- "ant_lit": 1-4 plain words with the opposite meaning in this sentence (the sentence may become odd but must stay grammatical).
- "ant_idiom": an idiom of the same language with the opposite meaning, inflected to fit; null if none exists.
- "is_idiom": false if the marked span is NOT used as an idiom here (plain collocation, term, literal use,
  grammatical phrase); then still fill the other fields as well as you can. Otherwise true.
- "meaning_here": in the sentence's language, one short phrase: what the idiom means in this sentence.
- "literal_reading": in the sentence's language, one short phrase: what the idiom's words would mean if read literally.
- "opposite_meaning": in the sentence's language, one short phrase: the opposite of the idiom's meaning here.

Answer with JSON only:
{"items": [{"id": "...", "is_idiom": true, "syn_lit": "...", "syn_idiom": "...", "ant_lit": "...", "ant_idiom": null,
            "meaning_here": "...", "literal_reading": "...", "opposite_meaning": "..."}]}"""


def wiktionary_examples():
    """uk.wiktionary examples where the idiom is highlighted with {{виділ|...}} -> exact span."""
    from idiom_sources import parse_uk_wiktionary, plain, split_args, templates
    rows = []
    for page in map(json.loads, open("data/raw/wiktionary/uk_wiktionary_frazeologizmy.jsonl", encoding="utf-8")):
        info = parse_uk_wiktionary(page["title"], page["wikitext"]) or {}
        for line in page["wikitext"].splitlines():
            for a, b in templates(line):
                args = split_args(line[a:b])
                if args[0] not in ("приклад", "приклад2") or len(args) < 2 or "{{виділ" not in args[1]:
                    continue
                ex = re.sub(r"\{\{виділ\|([^{}|]*)\}\}", r"⟦\1⟧", args[1])
                ex = re.sub(r"\(\s*\[\[[^\]]*\]\][^)]*\)\s*$", "", ex)
                ex = plain(ex).strip(" «»\"")
                if ex.count("⟦") == 0 or ex.count("⟦") != ex.count("⟧"):
                    continue
                i, j = ex.index("⟦"), ex.rindex("⟧")                 # first..last highlight = the span
                sent = ex.replace("⟦", "").replace("⟧", "")
                start, end = i, j - (2 * ex.count("⟦") - 1)
                rows.append({"idiom": page["title"], "meaning": info.get("meaning_uk"), "sentence": sent,
                             "span_start": start, "span_end": end, "source": "uk_wiktionary", "method": "annotated"})
    return pd.DataFrame(rows)


def select(n_target=750):
    """UK only: SimIdioms (agent-confirmed idioms) + uk.wiktionary examples + PARSEME, one sentence per idiom."""
    from idiom_sources import key
    d = pd.read_parquet(DATA / "idiom_sentences.parquet")
    c = pd.read_parquet("data/labels/idiomatic/consensus.parquet")
    confirmed = set(c[(c.decided_by == "agents_unanimous") & (c.label == "1")].id)
    sim = d[(d.source == "simidioms") & (d.language == "uk") & d.ok_design_a & d.idiom_id.isin(confirmed)]
    sim = sim.assign(source="simidioms")[["idiom", "meaning", "sentence", "span_start", "span_end", "source", "method"]]

    wik = wiktionary_examples()
    pm = pd.read_parquet("data/raw/parseme/uk_parseme_idioms.parquet")
    pm = pm[pm.category.isin(["VID", "NID", "AdjID"])].assign(meaning=None, source="parseme", method="annotated")
    pm["idiom"] = pm.surface.str.lower()

    allr = pd.concat([sim, wik, pm[["idiom", "meaning", "sentence", "span_start", "span_end", "source", "method"]]],
                     ignore_index=True)
    allr["span_start"], allr["span_end"] = allr.span_start.astype(int), allr.span_end.astype(int)
    allr["span_text"] = [s[a:b] for s, a, b in zip(allr.sentence, allr.span_start, allr.span_end)]
    allr["n_words"] = allr.sentence.map(lambda x: len(WORD_RE.findall(x)))
    span_words = allr.span_text.map(lambda x: len(WORD_RE.findall(x)))
    allr = allr[allr.n_words.between(6, 40) & (span_words >= 2) & (span_words / allr.n_words <= 0.7)
                & (allr.span_text.str.strip().str.len() > 0)]
    allr["key"] = allr.idiom.map(key)
    prio = {"simidioms": 0, "uk_wiktionary": 1, "parseme": 2}
    allr = (allr.assign(_p=allr.source.map(prio), _exact=(allr.method == "lemma").astype(int),
                        _r=allr.sentence.map(lambda x: hashlib.sha1(x.encode()).hexdigest()))
            .sort_values(["_p", "_exact", "_r"]).groupby("key").head(1))      # one sentence per idiom
    allr = allr.drop_duplicates("sentence")
    # all wiktionary + N_PARSEME news sentences + SimIdioms up to n_target (the generator drops non-idioms later)
    wik = allr[allr.source == "uk_wiktionary"]
    pmx = allr[allr.source == "parseme"].sample(frac=1, random_state=SEED).head(N_PARSEME)
    sim = allr[allr.source == "simidioms"].sample(frac=1, random_state=SEED).head(n_target - len(wik) - len(pmx))
    sel = pd.concat([sim, wik, pmx], ignore_index=True)
    sel["language"] = "uk"
    sel["sentence_id"] = [hashlib.sha1(f"{a}|{b}".encode()).hexdigest()[:12] for a, b in zip(sel.source, sel.sentence)]
    sel["idiom_id"] = "uk:" + sel.key
    sel["marked"] = [f"{s[:a]}[[{s[a:b]}]]{s[b:]}" for s, a, b in zip(sel.sentence, sel.span_start, sel.span_end)]
    cols = ["sentence_id", "idiom_id", "language", "idiom", "meaning", "sentence", "span_start", "span_end",
            "span_text", "marked", "method", "n_words", "source"]
    sel = sel.drop_duplicates("sentence_id")[cols].reset_index(drop=True)
    sel.to_parquet(WORK / "substitution_sources.parquet", index=False)
    print(f"{len(sel)} UK sentences, one per idiom"); print(sel.source.value_counts().to_string())


def call(batch):
    body = {"model": MODEL, "temperature": 0.3, "response_format": {"type": "json_object"},
            "reasoning": {"effort": "low", "exclude": True}, "usage": {"include": True}, "max_tokens": 6000,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps(batch, ensure_ascii=False)}]}
    for _ in range(3):
        try:
            r = requests.post(f"{API}/chat/completions", headers=headers(), json=body, timeout=240)
            if r.status_code == 429 or r.status_code >= 500:
                continue
            r.raise_for_status()
            d = r.json()
            m = re.search(r"\{.*\}", d["choices"][0]["message"]["content"] or "", re.S)
            return (json.loads(m.group(0))["items"] if m else []), (d.get("usage") or {}).get("cost", 0) or 0
        except (requests.RequestException, ValueError, KeyError) as e:
            print("  ! retry:", repr(e)[:120])
    return [], 0


def generate(limit=None, max_usd=1.5):
    src = pd.read_parquet(WORK / "substitution_sources.parquet")
    done = {json.loads(l)["id"] for l in open(GEN_FILE)} if GEN_FILE.exists() else set()
    todo = [{"id": r.sentence_id, "left": r.sentence[:r.span_start], "span": r.span_text,
             "right": r.sentence[r.span_end:], "meaning": r.meaning}
            for r in src.itertuples() if r.sentence_id not in done][:limit]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    ids = {t["id"] for t in todo}
    print(f"{MODEL}: {len(todo)} sentences in {len(batches)} requests")
    spent, n = 0.0, 0
    with ThreadPoolExecutor(WORKERS) as pool, open(GEN_FILE, "a", encoding="utf-8") as out:
        futs = [pool.submit(call, b) for b in batches]
        for k, f in enumerate(as_completed(futs), 1):
            items, cost = f.result()
            spent += cost
            with open(SPENT_LOG, "a") as log:
                log.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "task": "substitutions",
                                      "model": MODEL, "cost": cost}) + "\n")
            for it in items:
                if it.get("id") in ids and it["id"] not in done:
                    out.write(json.dumps({**it, "model": MODEL}, ensure_ascii=False) + "\n")
                    done.add(it["id"])
                    n += 1
            out.flush()
            if k % 10 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)} requests, {n} sentences, ${spent:.4f}")
            if spent > max_usd:
                print("  stop: budget reached")
                for x in futs:
                    x.cancel()
                break


def trim_overlap(rep, left, right):
    """Drop a word the model repeated from the context: 'The' + 'the event begins' -> 'event begins'."""
    lw, rw, words = WORD_RE.findall(left)[-1:], WORD_RE.findall(right)[:1], rep.split()
    if lw and len(words) > 1 and words[0].lower() == lw[0].lower():
        words = words[1:]
    if rw and len(words) > 1 and words[-1].strip(",.").lower() == rw[0].lower():
        words = words[:-1]
    return " ".join(words)


def overlap(a, b):
    wa, wb = {w.lower() for w in WORD_RE.findall(a)}, {w.lower() for w in WORD_RE.findall(b)}
    return len(wa & wb) / max(1, min(len(wa), len(wb)))


def ok_phrase(p, span):
    return isinstance(p, str) and 0 < len(p.split()) <= 8 and p.strip().lower() != span.strip().lower() \
        and not re.search(r"[\[\]\n]", p)


def prefixes(text):
    return {w.lower()[:4] for w in WORD_RE.findall(text) if len(w) > 3}


def word_diff(a, b):
    a, b = WORD_RE.findall(a.lower()), WORD_RE.findall(b.lower())
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def add_flags(df, src):
    """Automatic quality flags (no labels needed). Soft flags are often harmless; see README of the dataset."""
    span = df.item_id.map(df[df.condition == "ORIG"].set_index("item_id").replacement)
    cond = df.condition
    doubtful = set(open(WORK / "doubtful_ids.txt").read().split()) if (WORK / "doubtful_ids.txt").exists() else set()
    df["flag_span_by_lemma_match"] = (cond == "ORIG") & df.item_id.map(src.method).eq("lemma").to_numpy()
    df["flag_reuses_idiom_word"] = cond.isin(["SYN_LIT", "SYN_LIT_MATCHED", "ANT_LIT"]) & np.array([
        bool(prefixes(r) & prefixes(s)) for r, s in zip(df.replacement, span)])
    df["flag_partial_not_one_word"] = (cond == "PARTIAL") & np.array([word_diff(r, s) != 1 for r, s in zip(df.replacement, span)])
    df["flag_lit_words_differ"] = (cond == "LIT_CONTEXT") & np.array([bool(prefixes(s) - prefixes(r))
                                                                     for r, s in zip(df.replacement, span)])
    df["flag_token_gap_ge3"] = (cond == "SYN_LIT_MATCHED") & ((df.n_tok_span - df.n_tok_idiom).abs() >= 3)
    df["flag_duplicate"] = df.duplicated(["item_id", "sentence"], keep="first") & (cond != "ORIG")
    df["flag_doubtful"] = df.item_id.isin(doubtful)            # items the LLM pass itself marked as doubtful
    return df


NEUTRAL_FILE = WORK / "substitutions_neutral.jsonl"   # neutral-context sentence + near_meaning per item (LLM pass)
RARE_FILE = WORK / "substitutions_rare.jsonl"         # rare dictionary idioms: full record incl. sentence (LLM pass)
GRAMMAR_FILE = WORK / "grammar_review.jsonl"          # {"id": "<item>|<cond>", "ok", "fix", "note"} (LLM review pass)
LLM_PASS = "llm-agent"                              # generator label for the second LLM pass
NO_FIX = {"ORIG", "NEUTRAL", "LIT_CONTEXT", "RAND"}  # whole sentences / originals: flag only, never rewrite


def starts_sentence(text, i):
    """Is position i the start of a sentence (so a replacement there needs a capital letter)?"""
    before = text[:i].rstrip(" «\"„“'(—–-")
    return not before or before[-1] in ".!?…:"


def unmark(text):
    """'a [[b]] c' -> ('a b c', start, end) or None."""
    if not isinstance(text, str) or text.count("[[") != 1 or text.count("]]") != 1 or text.index("[[") > text.index("]]"):
        return None
    i = text.index("[[")
    j = text.index("]]") - 2
    plain = text.replace("[[", "").replace("]]", "")
    return (plain, i, j) if plain[i:j].strip() else None


def jsonl(path):
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))} if path.exists() else {}


def build():
    src = pd.read_parquet(WORK / "substitution_sources.parquet").set_index("sentence_id")
    src["sentence_origin"] = "natural"
    gen, extra, neutral = jsonl(GEN_FILE), jsonl(EXTRA_FILE), jsonl(NEUTRAL_FILE)
    # rare dictionary idioms: the LLM pass wrote the sentence too
    rare_rows = []
    for rid, r in jsonl(RARE_FILE).items():
        u = unmark(r.get("sentence"))
        if u is None:
            continue
        plain, i, j = u
        rare_rows.append({"sentence_id": rid, "idiom_id": "uk:" + r["key"], "language": "uk", "idiom": r["idiom"],
                          "meaning": r.get("meaning"), "sentence": plain, "span_start": i, "span_end": j,
                          "span_text": plain[i:j], "marked": r["sentence"], "method": "generated",
                          "n_words": len(WORD_RE.findall(plain)), "source": "uk_wiktionary_rare",
                          "sentence_origin": "generated"})
        gen[rid] = {**r, "model": LLM_PASS}
        extra[rid] = r
        neutral[rid] = r
    if rare_rows:
        src = pd.concat([src, pd.DataFrame(rare_rows).set_index("sentence_id")])
    review = jsonl(GRAMMAR_FILE)
    rng = random.Random(SEED)
    vocab = {"uk": [w for x in src.sentence for w in WORD_RE.findall(x) if w.isalpha() and w.islower() and len(w) > 2]}
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER)

    def n_tok(text):
        return len(tok.encode(" " + text, add_special_tokens=False).ids)

    rows = []
    for sid, g in gen.items():
        if sid not in src.index or g.get("is_idiom") is False:
            continue                                              # generator says: not an idiom here
        s = src.loc[sid]
        x, nx = extra.get(sid, {}), neutral.get(sid, {})
        a, b, span = int(s.span_start), int(s.span_end), s.span_text
        if (a > 0 and s.sentence[a - 1].isalpha()) or (b < len(s.sentence) and s.sentence[b].isalpha()):
            continue                                              # span cut in the middle of a word
        target = n_tok(span)
        cands = [c for c in x.get("syn_lit_candidates") or [] if ok_phrase(c, span)]
        matched = min(cands, key=lambda c: (abs(n_tok(c) - target), abs(len(c) - len(span))), default=None)
        base = {"item_id": sid, "language": s.language, "source": s.source, "idiom_id": s.idiom_id,
                "idiom": s.idiom, "meaning": s.meaning, "meaning_here": g.get("meaning_here"),
                "literal_reading": g.get("literal_reading"), "opposite_meaning": g.get("opposite_meaning"),
                "near_meaning": nx.get("near_meaning"), "n_words_idiom": len(WORD_RE.findall(span)),
                "n_tok_idiom": target}

        def add(cond, rep, sentence, start, end, frame, orig, origin, generator):
            rows.append({**base, "condition": cond, "gold": CONDITIONS[cond], "frame": frame, "replacement": rep,
                         "sentence": sentence, "span_start": start, "span_end": end, "orig_sentence": orig,
                         "sentence_origin": origin, "n_words_span": len(WORD_RE.findall(rep)),
                         "n_tok_span": n_tok(rep), "generator": generator})

        # frame "original": the source sentence, only the span changes
        cap = span[:1].isupper() and starts_sentence(s.sentence, a)
        reps = {"ORIG": span, "SYN_LIT": g.get("syn_lit"), "SYN_LIT_MATCHED": matched,
                "SYN_IDIOM": g.get("syn_idiom"), "ANT_LIT": g.get("ant_lit"), "ANT_IDIOM": g.get("ant_idiom"),
                "RAND": " ".join(rng.choice(vocab["uk"]).lower() for _ in WORD_RE.findall(span)),
                "PARTIAL": x.get("partial")}
        for cond, rep in reps.items():
            if cond != "ORIG" and not ok_phrase(rep, span):
                continue                                          # null / broken replacement: condition missing
            rep = rep.strip()
            if cond not in ("ORIG", "RAND"):
                rep = trim_overlap(rep, s.sentence[:a], s.sentence[b:])
            if cond == "PARTIAL" and overlap(rep, span) < 0.3:
                continue                                          # partial must keep most of the idiom
            if cond == "SYN_IDIOM" and overlap(rep, span) > 0.5:
                continue                                          # same idiom in another form, not a synonym
            if cap:
                rep = rep[:1].upper() + rep[1:]
            gen_by = LLM_PASS if cond in ("PARTIAL", "SYN_LIT_MATCHED") else g.get("model")
            add(cond, rep, s.sentence[:a] + rep + s.sentence[b:], a, a + len(rep), "original", s.sentence,
                s.sentence_origin, gen_by)

        # frame "neutral": new short sentence whose context does not reveal the meaning
        u = unmark(nx.get("neutral_sentence"))
        if u:
            plain, i, j = u
            nspan = plain[i:j]
            add("NEUTRAL", nspan, plain, i, j, "neutral", plain, "generated", LLM_PASS)
            for cond, key in [("NEUTRAL_SYN_LIT", "neutral_syn_lit"), ("NEUTRAL_ANT_LIT", "neutral_ant_lit")]:
                rep = nx.get(key)
                if ok_phrase(rep, nspan):
                    rep = trim_overlap(rep.strip(), plain[:i], plain[j:])
                    if nspan[:1].isupper() and starts_sentence(plain, i):
                        rep = rep[:1].upper() + rep[1:]
                    add(cond, rep, plain[:i] + rep + plain[j:], i, i + len(rep), "neutral", plain, "generated",
                        LLM_PASS)

        # frame "literal" (design B): the same idiom words in a new, literal sentence
        u = unmark(x.get("lit_sentence")) if x.get("literal_plausible") else None
        if u:
            plain, i, j = u
            add("LIT_CONTEXT", plain[i:j], plain, i, j, "literal", plain, "generated", LLM_PASS)

    df = pd.DataFrame(rows)
    # grammar review: minimal fixes of the replacement phrase; whole sentences are only flagged
    df["variant_id"] = df.item_id + "|" + df.condition
    df["grammar_ok"] = df.variant_id.map(lambda v: review.get(v, {}).get("ok"))
    df["grammar_note"] = df.variant_id.map(lambda v: review.get(v, {}).get("note"))
    n_fixed = 0
    for k in df.index:
        rv = review.get(df.at[k, "variant_id"], {})
        fix = rv.get("fix")
        if rv.get("ok") is False and isinstance(fix, str) and fix.strip() and df.at[k, "condition"] not in NO_FIX:
            a, b = int(df.at[k, "span_start"]), int(df.at[k, "span_end"])
            sent = df.at[k, "sentence"]
            df.at[k, "sentence"] = sent[:a] + fix.strip() + sent[b:]
            df.at[k, "replacement"] = fix.strip()
            df.at[k, "span_end"] = a + len(fix.strip())
            df.at[k, "grammar_ok"] = True
            df.at[k, "grammar_note"] = f"fixed: {rv.get('note', '')}"
            n_fixed += 1
    df["flag_grammar"] = df.grammar_ok.eq(False)
    df = add_flags(df, src)
    df["syn_lit_candidates"] = df.item_id.map(lambda i: extra.get(i, {}).get("syn_lit_candidates"))
    df["literal_plausible"] = df.item_id.map(lambda i: extra.get(i, {}).get("literal_plausible"))
    df.to_parquet(DATA / "substitutions.parquet", index=False)
    df.to_csv(DATA / "substitutions.csv", index=False, encoding="utf-8-sig")
    print(f"{df.item_id.nunique()} sentences -> {len(df)} variants; grammar fixes applied: {n_fixed}, "
          f"still flagged: {int(df.flag_grammar.sum())}")
    print(df.groupby("source").item_id.nunique().to_string())
    print(df.condition.value_counts().to_string())


def show(n):
    df = pd.read_parquet(DATA / "substitutions.parquet")
    HL, RESET = "\033[1;30;43m", "\033[0m"
    for sid in df.item_id.drop_duplicates().head(n):
        g = df[df.item_id == sid]
        print(f"\n\033[1m{g.idiom.iat[0]}\033[0m — {g.meaning.iat[0]}")
        for r in g.itertuples():
            print(f"  {r.condition:15} {r.sentence[:r.span_start]}{HL}{r.replacement}{RESET}{r.sentence[r.span_end:]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "generate", "build", "show"])
    ap.add_argument("--n", type=int, default=750)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-usd", type=float, default=1.5)
    a = ap.parse_args()
    {"select": lambda: select(a.n), "generate": lambda: generate(a.limit, a.max_usd),
     "build": build, "show": lambda: show(a.n)}[a.cmd]()

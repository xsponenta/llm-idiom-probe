"""Idiom sentence dataset: every idiom we can find, every sentence it occurs in, with its span.

    .venv/bin/python build_dataset.py

Sources
- SimIdioms: dictionary examples for all clusters (UK + EN). Usage assumed figurative.
- MAGPIE:    EN corpus sentences annotated as figurative / literal (span from annotation).
- Kobza:     UK web corpus sample, searched for the UK SimIdioms idioms. Usage unknown.

Output (data/processed/)
- idiom_sentences.parquet / .csv : one row per (sentence, idiom) with span and quality flags
- idioms.parquet / .csv          : one row per idiom with sentence counts per source
"""
import hashlib
import json
import random
import re
import time
from functools import lru_cache
from itertools import product
from pathlib import Path

import pandas as pd
import pymorphy3
import spacy

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
random.seed(SEED)

MAX_WORDS = 60            # longer sentences are not used for design A
MAX_COVERAGE = 0.7        # span may cover at most 70% of the sentence words (design A)
WINDOW_EXTRA = 3          # lemma match: idiom words must fit in len(idiom) + 3 tokens
KOBZA_CANDIDATES = 150    # max candidate sentences checked per idiom variant
KOBZA_PER_IDIOM = 20      # max Kobza sentences kept per idiom
KOBZA_WINDOW_EXTRA = 1    # stricter lemma window for corpus search (more false positives there)

WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*")
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- SimIdioms clusters
sim = [json.loads(line) for line in open(RAW / "simidioms" / "sim_idioms.jsonl", encoding="utf-8")]


def texts(items):
    return [x["text"].strip() for x in items or [] if x.get("text", "").strip()]


def idiom_variants(side):
    """Only real idioms of this language; field == "translation" entries are calques of the other side."""
    return texts([x for x in side.get("idiom", []) if x.get("field") == "idiom"])


def meaning(r, lang):
    other = "en" if lang == "uk" else "uk"
    for side, key in [(lang, "figurative_meaning"), (lang, "figurative_meaning_translated"),
                      (other, "figurative_meaning")]:
        m = texts(r[side].get(key))
        if m:
            return m[0], f"{side}.{key}"
    return None, None


# Only unambiguous placeholders. Bare "що", "чим", "кого", "it", "one" are often real idiom words
# ("поки що", "ні з чим", "make it"); in UPPERCASE entries lowercase government words are dropped anyway.
PLACEHOLDERS = {
    "uk": {"когось", "кому-сь", "комусь", "кимсь", "чогось", "чомусь", "чимсь", "щось", "хтось",
           "кого-небудь", "кому-небудь", "ким-небудь", "чого-небудь", "чому-небудь", "чим-небудь",
           "що-небудь", "хто-небудь", "який-небудь", "кого-чого", "кому-чому", "ким-чим"},
    "en": {"one's", "someone's", "somebody's", "someone", "somebody", "something", "sth", "sb",
           "oneself", "yourself"},
}


def clean_entry(entry, lang):
    s = entry.replace("’", "'").replace("ʼ", "'")
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", s)
    s = re.sub(r"\.\.\.|…", " ", s)
    words = s.split()
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) > 0.6 * len(letters):
        words = [w for w in words if not any(c.islower() for c in w)]
    s = re.sub(r"\s*/\s*", "/", " ".join(words).lower())
    out = []
    for combo in product(*[w.split("/") for w in s.split()]):
        ws = [re.sub(r"[^\w'-]", "", w).strip("-'") for w in combo]
        ws = [w for w in ws if w and w not in PLACEHOLDERS[lang]]
        if ws:
            out.append(" ".join(ws))
    return list(dict.fromkeys(out))


clusters = []
for r in sim:
    for lang in ["uk", "en"]:
        variants = idiom_variants(r[lang])
        if not variants:
            continue
        m, m_src = meaning(r, lang)
        clusters.append({
            "cluster_id": r["id_"], "language": lang, "variants": variants,
            "entry_of": {c: e for e in variants for c in clean_entry(e, lang)},   # cleaned -> original entry
            "meaning": m, "meaning_source": m_src,
            "examples": texts(r[lang].get("example")),
        })
clusters = pd.DataFrame(clusters)
log(f"SimIdioms: {len(sim)} clusters; idioms per language {clusters.groupby('language').size().to_dict()}")

# ---------------------------------------------------------------- span matching
nlp = {"uk": spacy.load("uk_core_news_md", exclude=["ner", "parser"]),
       "en": spacy.load("en_core_web_sm", exclude=["ner", "parser"])}
morph = pymorphy3.MorphAnalyzer(lang="uk")
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}


def norm_uk(w):
    w = w.lower().replace("’", "'").replace("ʼ", "'")
    w = {"у": "в", "й": "і", "із": "з", "зі": "з"}.get(w, w)
    return "в" + w[1:] if len(w) >= 3 and w.startswith("у") else w


@lru_cache(maxsize=500_000)
def uk_forms(word):
    return frozenset(norm_uk(p.normal_form) for p in morph.parse(word.lower())[:3])


def lemmas(tok, lang):
    if lang == "en":
        return {tok.lemma_.lower(), tok.lower_}
    return {norm_uk(tok.lemma_), norm_uk(tok.text)} | uk_forms(tok.text)


@lru_cache(maxsize=None)
def idiom_words(variant, lang):
    """[(lemmas, is_content)] for each word of a cleaned idiom variant."""
    doc = nlp[lang](variant)
    words = [(frozenset(lemmas(t, lang)),
              t.pos_ in {"NOUN", "PROPN", "VERB"} or (t.pos_ in CONTENT_POS and not t.is_stop))
             for t in doc if not t.is_punct]
    if not any(c for _, c in words):
        words = [(l, True) for l, _ in words]
    return tuple(words)


def exact_span(sentence, variant):
    m = re.search(r"(?<!\w)" + r"[\s,]+".join(map(re.escape, variant.split())) + r"(?!\w)", sentence, re.I)
    return (m.start(), m.end()) if m else None


def lemma_span(doc, words, lang, extra=WINDOW_EXTRA, ordered=False):
    toks = [t for t in doc if not t.is_punct]
    tok_lem = [lemmas(t, lang) for t in toks]
    content = [l for l, c in words if c]
    func = [l for l, c in words if not c]
    cands = [[i for i, tl in enumerate(tok_lem) if tl & l] for l in content]
    if not all(cands):
        return None
    best = None
    for combo in product(*cands):
        if len(set(combo)) < len(combo) or (ordered and list(combo) != sorted(combo)):
            continue
        lo, hi = min(combo), max(combo)
        if hi - lo + 1 > len(words) + extra:
            continue
        if best is None or hi - lo < best[1] - best[0]:
            best = (lo, hi)
    if best is None:
        return None
    lo, hi = best

    def is_func(i):
        return 0 <= i < len(toks) and any(tok_lem[i] & f for f in func)

    def skippable(i):
        return 0 <= i < len(toks) and toks[i].pos_ in {"DET", "PRON"}

    for _ in range(2):
        if is_func(lo - 1):
            lo -= 1
        elif skippable(lo - 1) and is_func(lo - 2):
            lo -= 2
        if is_func(hi + 1):
            hi += 1
        elif skippable(hi + 1) and is_func(hi + 2):
            hi += 2
    return toks[lo].idx, toks[hi].idx + len(toks[hi])


def missing_words(doc, start, end, variant, lang):
    """Words of the idiom variant with no lemma match inside the span (lemma matches only)."""
    span_lem = [lemmas(t, lang) for t in doc if start <= t.idx < end and not t.is_punct]
    return [i for i, (l, _) in enumerate(idiom_words(variant, lang)) if not any(l & sl for sl in span_lem)]


def find_span(sentence, variants, lang, doc=None, allow_lemma=lambda v: True, extra=WINDOW_EXTRA,
              ordered=False):
    """Returns dict(span_start, span_end, matched_variant, method, span_complete) or None."""
    for v in variants:
        sp = exact_span(sentence, v)
        if sp:
            return {"span_start": sp[0], "span_end": sp[1], "matched_variant": v, "method": "exact",
                    "span_complete": True}
    doc = doc if doc is not None else nlp[lang](sentence)
    for v in variants:
        if not allow_lemma(v):
            continue
        sp = lemma_span(doc, idiom_words(v, lang), lang, extra, ordered)
        if sp:
            return {"span_start": sp[0], "span_end": sp[1], "matched_variant": v, "method": "lemma",
                    "span_complete": not missing_words(doc, *sp, v, lang)}
    return None


# ---------------------------------------------------------------- 1. SimIdioms examples
rows = []
for c in clusters.itertuples():
    for sent in c.examples:
        sp = find_span(sent, list(c.entry_of), c.language) if c.entry_of else None
        rows.append({
            "source": "simidioms", "language": c.language, "cluster_id": c.cluster_id,
            "idiom": c.entry_of[sp["matched_variant"]] if sp else c.variants[0],
            "sentence": sent, "usage": "figurative", "usage_source": "dictionary_example",
            **(sp or {"method": "miss"}),
        })
sim_rows = pd.DataFrame(rows)
log(f"SimIdioms examples: {len(sim_rows)}; method counts\n"
    f"{pd.crosstab(sim_rows.language, sim_rows.method)}")

# ---------------------------------------------------------------- 2. MAGPIE
en_cluster_of = {}                         # cleaned EN variant -> cluster_id (for linking)
for c in clusters[clusters.language == "en"].itertuples():
    for v in c.entry_of:
        en_cluster_of.setdefault(v, c.cluster_id)

rows = []
for line in open(RAW / "magpie" / "MAGPIE_filtered_split_random.jsonl", encoding="utf-8"):
    m = json.loads(line)
    sent = m["context"][2]
    start, end = min(a for a, _ in m["offsets"]), max(b for _, b in m["offsets"])
    keys = clean_entry(m["idiom"], "en")
    rows.append({
        "source": "magpie", "language": "en",
        "cluster_id": next((en_cluster_of[k] for k in keys if k in en_cluster_of), None),
        "idiom": m["idiom"], "sentence": sent,
        "usage": {"i": "figurative", "l": "literal"}[m["label"]], "usage_source": "magpie_annotators",
        "label_confidence": m["confidence"], "magpie_id": m["id"], "genre": m["genre"],
        "context_before": " ".join(m["context"][:2]), "context_after": " ".join(m["context"][3:]),
        "span_start": start, "span_end": end, "matched_variant": m["idiom"], "method": "annotated",
        "span_complete": True,
    })
magpie_rows = pd.DataFrame(rows)
log(f"MAGPIE: {len(magpie_rows)} sentences, {magpie_rows.idiom.nunique()} idioms, "
    f"{magpie_rows.cluster_id.notna().sum()} rows linked to SimIdioms")

# ---------------------------------------------------------------- 3. Kobza (UK corpus search)
kobza = pd.read_parquet(RAW / "kobza" / "kobza_first_10000.parquet", columns=["id", "text", "source"])
SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[«\"„“—–-]?[А-ЯІЇЄҐA-Z0-9])")
sents, seen = [], set()
for doc_id, text, src in kobza.itertuples(index=False):
    for para in text.split("\n"):
        for s in SPLIT_RE.split(para.strip()):
            s = s.strip()
            low = s.lower()
            if (not 5 <= len(s.split()) <= MAX_WORDS or len(s) > 500 or low in seen
                    or re.search(r"[ыэъё]", low) or not re.search(r"[іїєґ]", low)):
                continue
            seen.add(low)
            sents.append((doc_id, src, s))
log(f"Kobza: {len(kobza)} docs -> {len(sents)} unique UK sentences")

# more docs, already prefiltered to sentences that may hold an idiom (kobza_mine.py)
MINED = RAW / "kobza" / "kobza_mined_sentences.parquet"
if MINED.exists():
    for doc_id, src, s in pd.read_parquet(MINED).itertuples(index=False):
        if s.lower() not in seen:
            seen.add(s.lower())
            sents.append((doc_id, src, s))
    log(f"Kobza: + mined sentences -> {len(sents)} sentences")


def prefix(w):
    return norm_uk(w)[:4]


index = {}                                  # 4-char prefix -> set of sentence ids
for i, (_, _, s) in enumerate(sents):
    for w in set(WORD_RE.findall(s.lower())):
        index.setdefault(prefix(w), set()).add(i)


def n_content(v):
    return sum(c for _, c in idiom_words(v, "uk"))


def kobza_variants(c):
    """Corpus search uses multi-word variants only: one word ("зрозуміти") matches anything."""
    return [v for v in c.entry_of if len(v.split()) >= 2]


uk_clusters = clusters[clusters.language == "uk"]
cand_of = {}                                # (cluster_id) -> candidate sentence ids
for c in uk_clusters.itertuples():
    ids = set()
    for v in kobza_variants(c):
        content = [t.text for t in nlp["uk"](v) if not t.is_punct and len(t.text) >= 3] or v.split()
        sets = [index.get(prefix(w), set()) for w in content]
        hit = set.intersection(*sets) if sets else set()
        ids |= set(random.sample(sorted(hit), min(len(hit), KOBZA_CANDIDATES)))
    if ids:
        cand_of[c.cluster_id] = ids
need = sorted(set().union(*cand_of.values()))
log(f"Kobza: {len(cand_of)} UK idioms have candidates; parsing {len(need)} candidate sentences")

docs = dict(zip(need, nlp["uk"].pipe((sents[i][2] for i in need), batch_size=256)))
log("Kobza: parsed")

rows = []
for c in uk_clusters.itertuples():
    found = []
    for i in sorted(cand_of.get(c.cluster_id, ())):
        doc_id, src, s = sents[i]
        # single-content-word idioms ("з голки") only by exact match: lemma match is too loose for them
        sp = find_span(s, kobza_variants(c), "uk", doc=docs[i], allow_lemma=lambda v: n_content(v) >= 2,
                       extra=KOBZA_WINDOW_EXTRA, ordered=True)
        # corpus lemma matches: keep only complete spans that do not run across quotes / dashes / brackets
        if sp and sp["method"] == "lemma" and (
                not sp["span_complete"] or re.search(r"[\"«»„“”()\[\];:—–]| - ", s[sp["span_start"]:sp["span_end"]])):
            sp = None
        if sp:
            found.append({"source": "kobza", "language": "uk", "cluster_id": c.cluster_id,
                          "idiom": c.entry_of[sp["matched_variant"]], "sentence": s,
                          "usage": "unknown", "usage_source": None, "kobza_doc_id": doc_id,
                          "kobza_source": src, **sp})
    found.sort(key=lambda r: (r["method"] != "exact", not r["span_complete"]))
    rows += found[:KOBZA_PER_IDIOM]
kobza_rows = pd.DataFrame(rows)
log(f"Kobza: {len(kobza_rows)} sentences for {kobza_rows.cluster_id.nunique()} UK idioms\n"
    f"{kobza_rows.method.value_counts().to_dict()}")

# ---------------------------------------------------------------- combine + quality flags
data = pd.concat([sim_rows, magpie_rows, kobza_rows], ignore_index=True)
data = data.astype({"span_start": "Int64", "span_end": "Int64"})
meta = clusters.set_index(["cluster_id", "language"])[["meaning", "meaning_source"]]
data = data.join(meta, on=["cluster_id", "language"])

has = data.span_start.notna()
data["span_text"] = [s[a:b] if h else None for s, a, b, h in zip(data.sentence, data.span_start, data.span_end, has)]
data["marked"] = [f"{s[:a]}[[{s[a:b]}]]{s[b:]}" if h else s
                  for s, a, b, h in zip(data.sentence, data.span_start, data.span_end, has)]
data["n_words"] = data.sentence.map(lambda s: len(WORD_RE.findall(s)))
data["span_coverage"] = (data.span_text.map(lambda t: len(WORD_RE.findall(t)) if isinstance(t, str) else None)
                         .astype(float) / data.n_words).round(2)
data["ok_design_a"] = (has & data.span_complete.fillna(False).astype(bool)
                       & (data.span_coverage <= MAX_COVERAGE) & (data.n_words <= MAX_WORDS))
data["idiom_key"] = data.cluster_id.where(data.source != "magpie",
                                          data.cluster_id.fillna("MAGPIE:" + data.idiom))
data["idiom_id"] = data.language + ":" + data.idiom_key
# stable ids: labels in data/labels/ refer to these, so they must survive a rebuild
data["sentence_id"] = [hashlib.sha1(f"{a}|{b}|{c}".encode()).hexdigest()[:12]
                       for a, b, c in zip(data.source, data.idiom_id, data.sentence)]
data = data.drop_duplicates("sentence_id")
data.insert(0, "row_id", range(len(data)))

cols = ["row_id", "sentence_id", "idiom_id", "source", "language", "idiom_key", "cluster_id", "idiom", "meaning", "meaning_source",
        "sentence", "marked", "span_start", "span_end", "span_text", "matched_variant", "method",
        "span_complete", "span_coverage", "n_words", "ok_design_a", "usage", "usage_source",
        "label_confidence", "magpie_id", "genre", "context_before", "context_after",
        "kobza_doc_id", "kobza_source"]
data = data[cols]
data.to_parquet(OUT / "idiom_sentences.parquet", index=False)
data.to_csv(OUT / "idiom_sentences.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- idiom inventory
found = data[data.span_start.notna()]
idioms = (found.groupby(["idiom_id", "language", "idiom_key"])
          .agg(idiom=("idiom", lambda s: s.mode().iat[0]),
               cluster_id=("cluster_id", "first"),
               meaning=("meaning", "first"),
               n_sentences=("row_id", "size"),
               n_ok_design_a=("ok_design_a", "sum"),
               n_simidioms=("source", lambda s: (s == "simidioms").sum()),
               n_magpie=("source", lambda s: (s == "magpie").sum()),
               n_kobza=("source", lambda s: (s == "kobza").sum()),
               n_figurative=("usage", lambda s: (s == "figurative").sum()),
               n_literal=("usage", lambda s: (s == "literal").sum()))
          .reset_index())
both = (set(idioms[(idioms.language == "uk") & (idioms.n_ok_design_a > 0)].cluster_id.dropna())
        & set(idioms[(idioms.language == "en") & (idioms.n_ok_design_a > 0)].cluster_id.dropna()))
idioms["cluster_in_both_languages"] = idioms.cluster_id.isin(both)
idioms.to_parquet(OUT / "idioms.parquet", index=False)
idioms.to_csv(OUT / "idioms.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- summary
log("done")
print("\nSentences with a span, by source:")
print(pd.crosstab([found.language, found.source], found.ok_design_a.map({True: "ok_A", False: "not_A"}),
                  margins=True))
print("\nIdioms with ≥1 sentence:", idioms.groupby("language").size().to_dict())
print("Idioms with ≥1 design-A sentence:", idioms[idioms.n_ok_design_a > 0].groupby("language").size().to_dict())
print("SimIdioms clusters with design-A sentences in both UK and EN:", len(both))
print("EN idioms with ≥3 figurative and ≥3 literal (design B):",
      ((idioms.n_figurative >= 3) & (idioms.n_literal >= 3)).sum())

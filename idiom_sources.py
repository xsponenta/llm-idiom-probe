"""Extra Ukrainian idiom sources, merged with SimIdioms into one inventory.

    .venv/bin/python idiom_sources.py

Sources (raw copies go to data/raw/<source>/):
- uk.wiktionary  Категорія:Фразеологізми/uk     ~417 entries, meaning + literary examples   CC BY-SA 4.0
- en.wiktionary  Category:Ukrainian idioms        ~104 entries, English gloss + examples      CC BY-SA 4.0
- PARSEME 2.0 UK (news, GRAC) idioms annotated in real sentences                              CC BY-NC-SA 4.0
- SimIdioms UK idioms (field == "idiom")                                                       academic only

Output: data/interim/uk_idiom_inventory.parquet, one row per idiom (deduplicated by lemma key):
    idiom, key, sources, meaning_uk, meaning_en, examples (list of sentences), simidioms_cluster, n_examples
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import pymorphy3
import requests

RAW = Path("data/raw")
INTERIM = Path("data/interim")
UA = {"User-Agent": "idiom-course-project/0.1 (research script)"}
morph = pymorphy3.MorphAnalyzer(lang="uk")


def get_json(url, params):
    for attempt in range(5):
        r = requests.get(url, headers=UA, timeout=60, params=params)
        try:
            return r.json()
        except ValueError:
            time.sleep(3 * (attempt + 1))         # rate limit / transient error page
    r.raise_for_status()
    raise RuntimeError(f"no JSON from {url}")


# ------------------------------------------------------------------ wikitext helpers
def templates(text):
    """Top-level {{...}} templates with their (start, end) positions; handles nesting."""
    out, depth, start, i = [], 0, None, 0
    while i < len(text) - 1:
        if text[i:i + 2] == "{{":
            if depth == 0:
                start = i
            depth += 1
            i += 2
        elif text[i:i + 2] == "}}" and depth:
            depth -= 1
            i += 2
            if depth == 0:
                out.append((start, i))
        else:
            i += 1
    return out


def split_args(tpl):
    """'{{name|a|b|k=v}}' -> ['name', 'a', 'b', 'k=v'] (top-level pipes only)."""
    body, parts, depth, cur = tpl[2:-2], [], 0, ""
    for j, ch in enumerate(body):
        if body[j:j + 2] in ("{{", "[[") :
            depth += 1
        elif body[j:j + 2] in ("}}", "]]") and depth:
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts]


def plain(text):
    """Wiki markup -> plain text (keeps the text of highlight / link templates)."""
    text = re.sub(r"\{\{(?:те саме|те же)\|([^|{}]*)\}\}", r"те саме, що «\1»", text)
    for _ in range(3):
        text = re.sub(r"\{\{(?:виділ|выдел|w|l|m)\|(?:[^|{}]*\|)?([^|{}]*)\}\}", r"\1", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"'''?|<[^>]+>|­", "", text)
    return re.sub(r"\s+", " ", text).strip(" ;,")


def wiki_pages(host, titles):
    """{title: wikitext}, 50 titles per request, redirects followed."""
    out = {}
    for i in range(0, len(titles), 50):
        r = get_json(f"https://{host}/w/api.php", {
            "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
            "redirects": 1, "format": "json", "formatversion": 2, "titles": "|".join(titles[i:i + 50])})
        for p in r["query"].get("pages", []):
            if "revisions" in p:
                out[p["title"]] = p["revisions"][0]["slots"]["main"]["content"]
        time.sleep(0.5)
    return out


def category(host, cat):
    titles, cont = [], {}
    while True:
        r = get_json(f"https://{host}/w/api.php", {
            "action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": 500,
            "cmnamespace": 0, "format": "json", **cont})
        titles += [m["title"] for m in r["query"]["categorymembers"]]
        if "continue" not in r:
            return titles
        cont = r["continue"]


# ------------------------------------------------------------------ uk.wiktionary
def parse_uk_wiktionary(title, wt):
    m = re.search(r"=+\s*Значення\s*=+\n(.*?)(?=\n=+[^=\n]+=+\s*\n|\Z)", wt, re.S)
    if not m:
        return None
    meanings, examples = [], []
    for line in m.group(1).splitlines():
        if not line.startswith("#") or line.startswith("#:"):
            continue
        line = line.lstrip("#").strip()
        for a, b in reversed(templates(line)):
            args = split_args(line[a:b])
            if args[0] in ("приклад", "приклад2") and len(args) > 1 and args[1]:
                ex = re.sub(r"\(\s*\[\[[^\]]*\]\][^)]*\)\s*$", "", args[1])      # trailing (author)
                ex = plain(ex)
                if len(ex.split()) >= 3:
                    examples.append(ex)
        meaning = plain(re.sub(r"\{\{приклад2?\|.*", "", line)).strip(" .")
        if re.search(r"\w", meaning):
            meanings.append(meaning)
    return {"idiom": title, "meaning_uk": "; ".join(meanings[:3]) or None, "examples": examples[::-1]}


# ------------------------------------------------------------------ en.wiktionary
def parse_en_wiktionary(title, wt):
    m = re.search(r"==Ukrainian==\n(.*?)(?=\n==[^=]|\Z)", wt, re.S)
    if not m:
        return None
    meanings, examples = [], []
    for line in m.group(1).splitlines():
        if re.match(r"#[^:*]", line):
            meanings.append(plain(line.lstrip("#")))
        elif line.startswith("#:"):
            for a, b in templates(line):
                args = split_args(line[a:b])
                if args[0] in ("ux", "uxi", "quote", "usex") and len(args) > 2 and args[1] == "uk":
                    examples.append(plain(args[2]))
    meanings = [x for x in meanings if x]
    return {"idiom": title, "meaning_en": "; ".join(meanings[:3]) or None,
            "examples": [e for e in examples if len(e.split()) >= 3]}


# ------------------------------------------------------------------ PARSEME
PARSEME_URL = "https://gitlab.com/parseme/sharedtask-data/-/raw/master/2.0/subtask1/UK/{}.cupt"
IDIOM_CATS = {"VID", "NID", "AdjID", "AdvID", "PronID", "IntjID"}   # no AdpID/ConjID: functional words


def parse_cupt(path):
    rows, toks, text = [], [], None
    for line in list(open(path, encoding="utf-8")) + ["\n"]:
        line = line.rstrip("\n")
        if line.startswith("# text = "):
            text = line[9:]
        elif line.startswith("#"):
            continue
        elif not line:
            if toks and text:
                rows += sentence_mwes(text, toks)
            toks, text = [], None
        else:
            c = line.split("\t")
            if c[0].isdigit():
                toks.append((c[1], c[2], c[10] if len(c) > 10 else "*"))
    return rows


def sentence_mwes(text, toks):
    # char offsets of tokens in the sentence text
    pos, offs = 0, []
    for form, _, _ in toks:
        i = text.find(form, pos)
        offs.append((i, i + len(form)) if i >= 0 else None)
        if i >= 0:
            pos = i + len(form)
    mwes = {}
    for k, (_, _, tag) in enumerate(toks):
        for part in tag.split(";") if tag not in ("*", "_") else []:
            mid, _, cat = part.partition(":")
            mwes.setdefault(mid, {"cat": None, "idx": []})
            mwes[mid]["idx"].append(k)
            if cat:
                mwes[mid]["cat"] = cat
    out = []
    for m in mwes.values():
        if m["cat"] not in IDIOM_CATS or any(offs[k] is None for k in m["idx"]):
            continue
        idx = m["idx"]
        out.append({"idiom": " ".join(toks[k][1].lower() for k in idx), "category": m["cat"], "sentence": text,
                    "span_start": offs[idx[0]][0], "span_end": offs[idx[-1]][1],
                    "surface": " ".join(toks[k][0] for k in idx)})
    return out


# ------------------------------------------------------------------ merge key
def key(idiom):
    """Order-free lemma key, so 'бити байдики' == 'байдики бити' == 'б'є байдики'."""
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]|[^\w\s'’-]", " ", idiom.lower()).replace("’", "'")
    words = [w for w in s.split() if w not in {"що-небудь", "кого-небудь", "когось", "щось", "чогось"}]
    return " ".join(sorted(morph.parse(w)[0].normal_form for w in words))


def main():
    for d in ["wiktionary", "parseme"]:
        (RAW / d).mkdir(parents=True, exist_ok=True)

    # uk.wiktionary
    path = RAW / "wiktionary" / "uk_wiktionary_frazeologizmy.jsonl"
    if not path.exists():
        titles = category("uk.wiktionary.org", "Категорія:Фразеологізми/uk")
        pages = wiki_pages("uk.wiktionary.org", titles)
        with open(path, "w", encoding="utf-8") as f:
            for t, wt in pages.items():
                f.write(json.dumps({"title": t, "wikitext": wt}, ensure_ascii=False) + "\n")
    ukw = [parse_uk_wiktionary(r["title"], r["wikitext"]) for r in map(json.loads, open(path, encoding="utf-8"))]
    ukw = pd.DataFrame([r for r in ukw if r]).assign(source="uk_wiktionary")
    print(f"uk.wiktionary: {len(ukw)} idioms, {ukw.meaning_uk.notna().sum()} with meaning, "
          f"{(ukw.examples.str.len() > 0).sum()} with examples")

    # en.wiktionary
    path = RAW / "wiktionary" / "en_wiktionary_ukrainian_idioms.jsonl"
    if not path.exists():
        titles = category("en.wiktionary.org", "Category:Ukrainian idioms")
        pages = wiki_pages("en.wiktionary.org", titles)
        with open(path, "w", encoding="utf-8") as f:
            for t, wt in pages.items():
                f.write(json.dumps({"title": t, "wikitext": wt}, ensure_ascii=False) + "\n")
    enw = [parse_en_wiktionary(r["title"], r["wikitext"]) for r in map(json.loads, open(path, encoding="utf-8"))]
    enw = pd.DataFrame([r for r in enw if r]).assign(source="en_wiktionary")
    print(f"en.wiktionary: {len(enw)} idioms, {enw.meaning_en.notna().sum()} with gloss, "
          f"{(enw.examples.str.len() > 0).sum()} with examples")

    # PARSEME
    pm = []
    for split in ["train", "dev"]:
        p = RAW / "parseme" / f"UK_{split}.cupt"
        if not p.exists():
            r = requests.get(PARSEME_URL.format(split), headers=UA, timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)
        pm += parse_cupt(p)
    pm = pd.DataFrame(pm)
    pm.to_parquet(RAW / "parseme" / "uk_parseme_idioms.parquet", index=False)
    print(f"PARSEME: {len(pm)} idiom occurrences, {pm.idiom.nunique()} idioms; {pm.category.value_counts().to_dict()}")
    pm_g = (pm.assign(marked=[f"{s[:a]}[[{s[a:b]}]]{s[b:]}" for s, a, b in zip(pm.sentence, pm.span_start, pm.span_end)])
            .groupby("idiom").agg(examples=("marked", list), category=("category", "first"),
                                  surface=("surface", lambda s: s.str.lower().mode().iat[0])).reset_index()
            .assign(source="parseme"))
    pm_g["idiom"] = pm_g.surface        # display the most frequent surface form, not the lemma string

    # SimIdioms (UK idioms only)
    sim = []
    for r in map(json.loads, open(RAW / "simidioms" / "sim_idioms.jsonl", encoding="utf-8")):
        mu = [x["text"] for x in r["uk"].get("figurative_meaning", []) or []]
        mt = [x["text"] for x in r["uk"].get("figurative_meaning_translated", []) or []]
        me = [x["text"] for x in r["en"].get("figurative_meaning", []) or []]
        for x in r["uk"].get("idiom", []):
            if x.get("field") == "idiom":
                sim.append({"idiom": x["text"], "meaning_uk": (mu or mt or [None])[0], "meaning_en": (me or [None])[0],
                            "examples": [e["text"] for e in r["uk"].get("example", []) or []],
                            "simidioms_cluster": r["id_"], "source": "simidioms"})
    sim = pd.DataFrame(sim)

    # merge by lemma key
    allr = pd.concat([sim, ukw, enw, pm_g], ignore_index=True)
    allr["key"] = allr.idiom.map(key)
    allr = allr[allr.key.str.len() > 0]
    order = {"simidioms": 0, "uk_wiktionary": 1, "en_wiktionary": 2, "parseme": 3}
    allr = allr.sort_values("source", key=lambda s: s.map(order))
    inv = allr.groupby("key").agg(
        idiom=("idiom", "first"),
        sources=("source", lambda s: sorted(set(s), key=order.get)),
        meaning_uk=("meaning_uk", lambda s: next((x for x in s if isinstance(x, str) and x), None)),
        meaning_en=("meaning_en", lambda s: next((x for x in s if isinstance(x, str) and x), None)),
        examples=("examples", lambda s: [e for lst in s for e in (lst if isinstance(lst, list) else [])]),
        simidioms_cluster=("simidioms_cluster", "first"),
        parseme_category=("category", "first"),
    ).reset_index()
    inv["n_words"] = inv.idiom.str.split().str.len()
    inv["n_examples"] = inv.examples.str.len()
    inv.to_parquet(INTERIM / "uk_idiom_inventory.parquet", index=False)

    print(f"\ninventory: {len(inv)} unique UK idioms")
    print(inv.sources.map(lambda s: "+".join(s)).value_counts().to_string())
    print("with any meaning:", (inv.meaning_uk.notna() | inv.meaning_en.notna()).sum(),
          "| with ≥1 example:", (inv.n_examples > 0).sum())


if __name__ == "__main__":
    main()

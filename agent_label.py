"""LLM annotators via OpenRouter. Each model writes data/labels/<task>/<name>.jsonl, like a human would.

    .venv/bin/python agent_label.py credits                          # key check + balance
    .venv/bin/python agent_label.py estimate pair                    # cost estimate, no calls
    .venv/bin/python agent_label.py run pair --limit 40              # pilot with default models
    .venv/bin/python agent_label.py run pair                         # everything (resumes)
    .venv/bin/python agent_label.py run usage --models deepseek/deepseek-v4.1-flash

Already labeled ids are skipped, so a run can be stopped and continued.
Spending is tracked from OpenRouter's own cost field; a run stops at --max-usd.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

from labeling import LABELS, TASKS, load_items, load_labels

API = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "google/gemini-3.1-flash-lite"]
BATCH = 20                 # items per request
WORKERS = 4                # parallel requests per model
SPENT_LOG = LABELS / "spending.jsonl"

GUIDE = {
    "idiomatic": """An idiom = a fixed expression whose meaning is figurative / not derivable from its words
(e.g. "kick the bucket", "бити байдики", "накивати п'ятами"). Count metaphorical set phrases too.
NOT idioms (label "0"): terms and names ("прожитковий мінімум", "державна зрада"), plain collocations and
discourse phrases with literal meaning ("дуже рідко", "подати скаргу", "у зв'язку з"), single plain words.
The examples show how the expression is used; the meaning field may be noisy.""",
    "usage": """The marked [[span]] was found automatically for the given idiom.
"f": the span is this idiom and it is used with its idiomatic (figurative) meaning.
"l": the same words are used literally ("he kicked the bucket of water over").
"x": the span is not this idiom (other words, different expression, broken boundaries, a mere coincidence).""",
    "subst_check": """Each item is one variant of a Ukrainian sentence in an idiom minimal-pair dataset. "original" has the
idiom marked [[ ]]; "variant" has the replacement marked [[ ]]; "condition" and "expect" say what the replacement
should be. Label "0" if ANY of these holds: the variant is ungrammatical in its sentence (case, agreement, word order);
its meaning does not match the expectation (a "synonym" that changes the meaning, an "antonym" that is not opposite);
SYN_IDIOM / ANT_IDIOM is not a real, established Ukrainian idiom (invented or calqued); PARTIAL changes more than one
word or is itself a known idiom (PARTIAL is SUPPOSED to be a broken, non-existent idiom: never penalize it for that,
only for grammar or for changing more than one word); LIT_CONTEXT can still be read idiomatically; ORIG span does not cover the idiom
correctly. Otherwise "1". Be strict about grammar, tolerant about style. Put the reason in "note" when "0".""",
    "pair": """Compare the figurative meanings of the UK idiom and the EN idiom as used in their sentences.
"1" only if a translator could use one for the other in most contexts. Partial overlap, different
nuance of main meaning, or different idioms → "0". The meaning fields may be copied from one side, so
judge the idioms themselves.""",
}


def api_key():
    for line in open(".env", encoding="utf-8"):
        if line.startswith("OPENROUTER_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key:
                return key
    raise SystemExit("Put your key into .env: OPENROUTER_API_KEY=sk-or-...")


def headers():
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}


def credits():
    r = requests.get(f"{API}/key", headers=headers(), timeout=30)
    r.raise_for_status()
    d = r.json()["data"]
    print(f"key ok | used ${d.get('usage', 0):.4f} | limit {d.get('limit')} | remaining {d.get('limit_remaining')}")
    if SPENT_LOG.exists():
        spent = sum(json.loads(l)["cost"] for l in open(SPENT_LOG) if l.strip())
        print(f"spent by agent_label.py so far: ${spent:.4f}")


def system_prompt(task):
    t = TASKS[task]
    labels = "\n".join(f'"{k}": {v}' for k, v in t["labels"].items())
    return f"""You are an expert linguist annotating a Ukrainian/English idiom dataset.

Question: {t["question"]}

{GUIDE[task]}

Labels:
{labels}

You get a JSON list of items. Answer with JSON only:
{{"labels": [{{"id": "<item id>", "label": "<label>", "note": "<max 10 words>"}}, ...]}}
One entry per item, ids copied exactly, same order. Use "?" rather than guessing."""


def todo_items(task, name, limit=None):
    done = set(load_labels(task).get(name, {}))
    items = [it for it in load_items(task) if it["id"] not in done]
    return items[:limit] if limit else items


def model_prices():
    d = requests.get(f"{API}/models", timeout=30).json()["data"]
    return {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in d}


def estimate(task, models, limit=None):
    prices = model_prices()
    sp = len(system_prompt(task)) / 3.5
    total = 0
    for m in models:
        items = todo_items(task, annotator_name(m), limit)
        n_req = -(-len(items) // BATCH)
        tok_in = sum(len(json.dumps(it, ensure_ascii=False)) for it in items) / 3 + n_req * sp
        tok_out = len(items) * 25 + n_req * 300          # labels + some reasoning slack
        pin, pout = prices[m]
        cost = tok_in * pin + tok_out * pout
        total += cost
        print(f"{m:35} {len(items):5} items  {n_req:4} requests  ~{tok_in / 1e3:6.0f}k in  ~${cost:.3f}")
    print(f"{'total':35} ~${total:.3f}")
    return total


def annotator_name(model):
    return model.split("/")[-1].replace(":", "_")


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))["labels"] if m else []


def call(model, task, batch):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt(task)},
                     {"role": "user", "content": json.dumps(batch, ensure_ascii=False)}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "low", "exclude": True},
        "usage": {"include": True}, "max_tokens": 6000,
    }
    for attempt in range(3):
        try:
            r = requests.post(f"{API}/chat/completions", headers=headers(), json=body, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()
            cost = (d.get("usage") or {}).get("cost", 0) or 0
            return parse(d["choices"][0]["message"]["content"] or ""), cost
        except (requests.RequestException, ValueError, KeyError) as e:
            err = e
            time.sleep(3)
    print(f"  ! {model}: batch failed ({err!r}); will be retried next run")
    return [], 0


def run(task, models, limit=None, max_usd=1.0):
    allowed = set(TASKS[task]["labels"])
    spent = 0.0
    for model in models:
        name = annotator_name(model)
        items = todo_items(task, name, limit)
        if not items:
            print(f"{model}: nothing to do")
            continue
        batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
        print(f"{model}: {len(items)} items in {len(batches)} requests -> {name}.jsonl")
        out = open(LABELS / task / f"{name}.jsonl", "a", encoding="utf-8")
        n_ok = 0
        with ThreadPoolExecutor(WORKERS) as pool:
            futs = [pool.submit(call, model, task, b) for b in batches]
            for k, f in enumerate(as_completed(futs), 1):
                labels, cost = f.result()
                spent += cost
                with open(SPENT_LOG, "a") as log:
                    log.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "task": task,
                                          "model": model, "cost": cost}) + "\n")
                ids = {it["id"] for b in batches for it in b}
                for lab in labels:
                    l = str(lab.get("label", "")).strip()
                    if lab.get("id") in ids and l in allowed:
                        out.write(json.dumps({"id": lab["id"], "label": l, "note": lab.get("note", ""),
                                              "model": model}, ensure_ascii=False) + "\n")
                        n_ok += 1
                out.flush()
                if k % 10 == 0 or k == len(futs):
                    print(f"  {k}/{len(futs)} requests, {n_ok} labels, run cost ${spent:.4f}")
                if spent > max_usd:
                    print(f"  stop: spent ${spent:.4f} > --max-usd {max_usd}")
                    for x in futs:
                        x.cancel()
                    break
        out.close()
        if spent > max_usd:
            break
    print(f"run cost: ${spent:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["credits", "estimate", "run"])
    ap.add_argument("task", nargs="?", choices=list(TASKS))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--limit", type=int, help="only the first N unlabeled items (pilot)")
    ap.add_argument("--max-usd", type=float, default=1.0, help="stop this run after spending this much")
    a = ap.parse_args()
    if a.cmd == "credits":
        credits()
    elif a.cmd == "estimate":
        estimate(a.task, a.models, a.limit)
    else:
        run(a.task, a.models, a.limit, a.max_usd)

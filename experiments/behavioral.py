"""E1. Behavioral eval without generation: the next-token log-probability of each allowed answer.

    python experiments/behavioral.py --models qwen3-1.7b gemma3-12b --batch 16
    python experiments/behavioral.py --models qwen3-0.6b --limit-items 20      # quick test

T1: which option (letters A-F, only those present) the [[marked]] part means here.  T2: "так" vs "ні".
The answer turn is pre-filled with "Відповідь:"; plain and bold ("**C**") answers are both counted (see PREFIX).
Output: data/results/gpu/<model>/e1.jsonl, one line per question (resumes: answered qids are skipped).
Fields: pred (argmax over allowed answers), probs (renormalised over allowed answers), mass (probability the
model puts on any allowed answer at all = format compliance), gold, chosen_kind (T1: what the chosen option is).
"""
import argparse
import json
import time

import torch

from common import (LETTERS, answer_token_ids, chat_prompt, device, free, limit_items, load_full, load_model,
                    out_dir, questions, resolve, write_meta)

T1_PROMPT = """Прочитай речення і визнач, що означає частина, позначена [[ ]], саме в цьому реченні.

Речення: {sentence}

Варіанти:
{options}

Відповідай лише однією літерою варіанта."""

T2_PROMPT = """Чи передають речення A і B однакове значення? Невеликі відмінності у формулюванні не важливі; \
змінене, протилежне або безглузде значення — це не те саме.

A: {A}
B: {B}

Відповідай лише одним словом: так або ні."""

# The assistant turn is pre-filled with "Відповідь:" so that the next token is the answer. Some models still
# write the answer in bold (" **C**"), so both formats are scored and summed (marginalised):
#   T1: P(letter) = P(" C" | prefix) + P(" **" | prefix) * P("C" | prefix + " **")
#   T2: P(yes) = sum over the forms below of P(form | prefix), each form scored token by token (teacher forcing)
METHOD = "v2-prefix-bold"           # v1 (first run): no prefix, plain letters only
PREFIX = "Відповідь:"
BOLD = " **"
T2_FORMS = {"yes": [" так", " Так", " **так", " **Так"], "no": [" ні", " Ні", " **ні", " **Ні"]}


def render(tok, q: dict) -> str:
    if q["task"] == "T1":
        opts = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
        return chat_prompt(tok, T1_PROMPT.format(sentence=q["sentence"], options=opts)) + PREFIX
    return chat_prompt(tok, T2_PROMPT.format(A=q["A"], B=q["B"])) + PREFIX


def ids_of(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False).input_ids


def left_pad(tok, seqs: list[list[int]], dev) -> dict:
    n = max(map(len, seqs))
    pad = tok.pad_token_id
    return {"input_ids": torch.tensor([[pad] * (n - len(s)) + s for s in seqs], device=dev),
            "attention_mask": torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in seqs], device=dev)}


@torch.no_grad()
def letter_logprobs(tok, model, qs: list[dict], cand: dict) -> list[dict]:
    """T1: two forward passes per question (prompt, prompt + bold marker); letters are single tokens."""
    dev = device(model)
    prompts = [ids_of(tok, render(tok, q)) for q in qs]
    bold = ids_of(tok, BOLD)
    nb = len(bold)
    seqs = prompts + [p + bold for p in prompts]
    logits = model(**left_pad(tok, seqs, dev)).logits[:, -(nb + 1):, :].float()   # last nb+1 positions
    logp = torch.log_softmax(logits, dim=-1)
    out = []
    for k, q in enumerate(qs):
        lp = logp[k, -1]                                        # after the prefix
        lb = logp[len(qs) + k]                                  # positions: prefix end, then each bold token
        p_bold = sum(lb[t, bold[t]] for t in range(nb))         # exact log P(" **" | prefix), all its tokens
        out.append({a: torch.logaddexp(torch.logsumexp(lp[cand[a]], 0),
                                       p_bold + torch.logsumexp(lb[-1][cand[a]], 0)).item() for a in q["options"]})
    return out


@torch.no_grad()
def word_logprobs(tok, model, qs: list[dict]) -> list[dict]:
    """T2: log p of each answer form after the prompt (teacher forcing on token ids), summed over the forms."""
    dev = device(model)
    prompts = [ids_of(tok, render(tok, q)) for q in qs]
    forms = {f: ids_of(tok, f) for fs in T2_FORMS.values() for f in fs}
    jobs = [(i, a, p + forms[f], len(forms[f])) for i, p in enumerate(prompts)
            for a, fs in T2_FORMS.items() for f in fs]
    enc = left_pad(tok, [j[2] for j in jobs], dev)
    logp = torch.log_softmax(model(**enc).logits.float(), dim=-1)
    T = enc["input_ids"].shape[1]                               # left padding: the form = the last n tokens
    per: list[dict] = [{a: [] for a in T2_FORMS} for _ in qs]
    for row, (i, a, seq, n) in enumerate(jobs):
        ids = enc["input_ids"][row]
        per[i][a].append(sum(logp[row, t - 1, ids[t]].item() for t in range(T - n, T)))
    return [{a: torch.logsumexp(torch.tensor(v), 0).item() for a, v in d.items()} for d in per]


def score(tok, model, qs: list[dict], cand: dict) -> list[dict]:
    t1 = [q for q in qs if q["task"] == "T1"]
    t2 = [q for q in qs if q["task"] == "T2"]
    scored = dict(zip([q["qid"] for q in t1], letter_logprobs(tok, model, t1, cand) if t1 else []))
    scored.update(zip([q["qid"] for q in t2], word_logprobs(tok, model, t2) if t2 else []))
    out = []
    for q in qs:
        per = scored[q["qid"]]
        mass = float(sum(torch.exp(torch.tensor(list(per.values())))))
        z = torch.logsumexp(torch.tensor(list(per.values())), 0).item()
        probs = {a: round(float(torch.exp(torch.tensor(v - z))), 4) for a, v in per.items()}
        pred = max(probs, key=probs.get)
        out.append({"method": METHOD, "qid": q["qid"], "task": q["task"], "condition": q["condition"], "item_id": q["item_id"],
                    "idiom_id": q["idiom_id"], "source": q["source"], "gold": q["gold"], "pred": pred,
                    "correct": None if q["gold"] is None else pred == q["gold"],
                    "chosen_kind": q.get("kinds", {}).get(pred), "probs": probs, "mass": round(mass, 4)})
    return out


def run_model(name: str, qs: list[dict], batch: int) -> None:
    short, hf_id = resolve(name)
    path = out_dir(short) / "e1.jsonl"
    done = {json.loads(line)["qid"] for line in open(path, encoding="utf-8")} if path.exists() else set()
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[E1] {short}: {len(todo)} of {len(qs)} questions to score", flush=True)
    if not todo:
        return
    tok, model = load_model(hf_id)
    tok.padding_side = "left"
    cand = {a: answer_token_ids(tok, [a, " " + a]) for a in LETTERS}
    write_meta(short, hf_id, model, {"e1_letter_token_ids": cand, "e1_t2_words": T2_FORMS})
    todo.sort(key=lambda q: len(q.get("sentence", "") + q.get("A", "") + q.get("B", "")))   # less padding
    t0, i, bs = time.time(), 0, batch
    with open(path, "a", encoding="utf-8") as f:
        while i < len(todo):
            chunk = todo[i:i + bs]
            try:
                rows = score(tok, model, chunk, cand)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"  OOM -> batch {bs}", flush=True)
                continue
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            i += len(chunk)
            if (i // len(chunk)) % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    free(model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit-items", type=int, help="only N random sentences (quick test)")
    a = ap.parse_args()
    full = load_full()
    qs = questions(full)
    if a.limit_items:
        keep = set(limit_items(full, a.limit_items).item_id)
        qs = [q for q in qs if q["item_id"] in keep]
    for m in a.models:
        run_model(m, qs, a.batch)

"""Aggregate data/results/gpu/<model>/* into tables and figures in data/results/gpu/summary/.

    python experiments/analyze.py                 # all models that have results
    python experiments/analyze.py --tag _limit20  # results of a quick test run

summary_metrics.csv: one row per model with the headline number of every experiment.
E1: accuracy per model x condition (T1, T2); T2 per class ("так" recall, "ні" recall, balanced accuracy),
    because small models tend to answer "так" to everything; format compliance (probability mass on allowed
    answers); what PARTIAL is read as; ORIG accuracy by sentence source.
E2: surprisal per condition; per-item jump vs ORIG for SYN_LIT and PARTIAL (span and the 3 tokens after it).
    Expected if the idiom is stored as a unit: the PARTIAL jump > the SYN_LIT jump.
E3: cosine to ORIG per layer, for span / right context / whole sentence. Main contrasts: SYN_LIT vs PARTIAL
    (meaning vs form) and NEUTRAL vs LIT_CONTEXT (same meaning in another context vs same words, other meaning);
    first layer where the meaning-side variant is closer to ORIG, on average and for the majority of items.
E4: probe figurative (ORIG or NEUTRAL) vs literal (LIT_CONTEXT) on span vectors, logistic regression per layer,
    5-fold GroupKFold by idiom (test idioms are never seen in training); chance = 0.5; shuffled-label control.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS  # noqa: E402

SUMMARY = RESULTS / "summary"
ORDER = ["ORIG", "SYN_LIT", "SYN_LIT_MATCHED", "SYN_IDIOM", "ANT_LIT", "ANT_IDIOM", "RAND", "PARTIAL",
         "NEUTRAL", "NEUTRAL_SYN_LIT", "NEUTRAL_ANT_LIT", "LIT_CONTEXT"]
POOLS = ("span", "right", "sent")
CONTRASTS = [("SYN_LIT", "PARTIAL"), ("NEUTRAL", "LIT_CONTEXT")]       # (meaning side, form side)
COLORS = {"SYN_LIT": "#2a78d6", "SYN_IDIOM": "#7b5cd6", "PARTIAL": "#d9822b", "ANT_LIT": "#c2410c",
          "RAND": "#6b7280", "NEUTRAL": "#0f8a6c", "LIT_CONTEXT": "#b4236c"}
METRICS: dict[str, dict] = {}                                          # model -> headline metrics


def models() -> list[Path]:
    return sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name not in ("summary", "logs"))


def put(model: str, **kv) -> None:
    METRICS.setdefault(model, {}).update(kv)


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(SUMMARY / f"{name}.csv")
    print(f"\n=== {name}\n{df.round(3).to_string()}")


def order_models(names) -> list[str]:
    """Roughly by family and size, for readable tables."""
    def key(n):
        size = "".join(c for c in n.split("-")[-1] if c.isdigit() or c == ".")
        return (n.split("-")[0], float(size) if size else 0.0)
    return sorted(names, key=key)


# ---------------------------------------------------------------- E1
def e1() -> None:
    frames = []
    for d in models():
        p = d / "e1.jsonl"
        if p.exists():
            frames.append(pd.DataFrame(map(json.loads, open(p, encoding="utf-8"))).assign(model=d.name))
    if not frames:
        return
    a = pd.concat(frames)
    graded = a[a.gold.notna()].copy()
    graded["correct"] = graded.correct.astype(float)          # object dtype (None for PARTIAL) -> numeric
    for task in ("T1", "T2"):
        t = graded[graded.task == task]
        tab = t.pivot_table(index="model", columns="condition", values="correct", aggfunc="mean")
        tab = tab.reindex(order_models(tab.index))[[c for c in ORDER if c in tab]]
        tab["ALL"] = t.groupby("model").correct.mean()
        save(tab * 100, f"e1_{task}_accuracy")
        heatmap(tab.drop(columns="ALL") * 100, f"E1 {task}: accuracy, %", f"e1_{task}_accuracy.png")
        for m, v in tab["ALL"].items():
            put(m, **{f"e1_{task}_acc": v})
    t1 = graded[graded.task == "T1"]
    for m, g in t1.groupby("model"):
        put(m, e1_T1_acc_ORIG=g[g.condition == "ORIG"].correct.mean(),
            e1_T1_acc_NEUTRAL=g[g.condition == "NEUTRAL"].correct.mean(),
            e1_T1_acc_LIT_CONTEXT=g[g.condition == "LIT_CONTEXT"].correct.mean())
    t2 = graded[graded.task == "T2"]
    per_class = t2.groupby(["model", "gold"]).correct.mean().unstack()
    per_class = per_class.rename(columns={"yes": "recall_tak", "no": "recall_ni"})
    per_class["balanced_acc"] = per_class.mean(axis=1)
    per_class["share_answered_tak"] = t2.groupby("model").pred.apply(lambda s: (s == "yes").mean())
    # Threshold-free separation: a model can rank "same" above "different" well and still say "ні" to everything
    # (a calibration bias); ROC-AUC of P(так) shows the two apart. 0.5 = chance.
    from sklearn.metrics import roc_auc_score
    p_yes = t2.probs.map(lambda p: p["yes"])
    per_class["roc_auc"] = t2.assign(p=p_yes).groupby("model").apply(
        lambda g: roc_auc_score(g.gold == "yes", g.p) if g.gold.nunique() == 2 else float("nan"))
    save(per_class.reindex(order_models(per_class.index)) * 100, "e1_T2_per_class")
    for m, r in per_class.iterrows():
        put(m, e1_T2_balanced_acc=r.balanced_acc, e1_T2_recall_tak=r.recall_tak, e1_T2_recall_ni=r.recall_ni,
            e1_T2_roc_auc=r.roc_auc)
    mass = a.groupby(["model", "task"]).mass.mean().unstack()
    save(mass.reindex(order_models(mass.index)), "e1_format_mass")
    part = a[(a.task == "T1") & (a.condition == "PARTIAL")]
    if not part.empty:
        save(pd.crosstab(part.model, part.chosen_kind, normalize="index"), "e1_partial_reading")
    orig = graded[(graded.task == "T1") & (graded.condition == "ORIG")]
    if "source" in orig and orig.source.notna().any():
        save(orig.pivot_table(index="model", columns="source", values="correct", aggfunc="mean") * 100,
             "e1_T1_orig_by_source")
    wrong = graded[(graded.task == "T1") & (graded.correct == 0)]
    if not wrong.empty:
        save(pd.crosstab([wrong.model, wrong.condition], wrong.chosen_kind, normalize="index"), "e1_T1_error_kinds")


def heatmap(tab: pd.DataFrame, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(1.0 * tab.shape[1] + 2.5, 0.5 * tab.shape[0] + 1.6))
    ax.imshow(tab.values.astype(float), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if v == v:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 60 else "black")
    ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(tab.shape[0]), tab.index, fontsize=9)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(SUMMARY / fname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- E2 / E3
def load_probe(d: Path, tag: str) -> pd.DataFrame | None:
    files = sorted((d / f"lm_probe{tag}").glob("shard_*.parquet"))
    return pd.concat(map(pd.read_parquet, files)) if files else None


def e2(df: pd.DataFrame, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["span_surprisal_mean", "span_first_surprisal", "post_surprisal_mean", "sent_surprisal_mean"]
    per_cond = (df.groupby("condition")[cols].mean().reindex([c for c in ORDER if c in set(df.condition)])
                .assign(model=model).set_index("model", append=True))
    base = df[df.condition == "ORIG"].set_index("item_id")
    jumps = {}
    for c in ("SYN_LIT", "SYN_LIT_MATCHED", "PARTIAL", "RAND"):
        v = df[df.condition == c].set_index("item_id")
        both = v.index.intersection(base.index)
        jumps[c] = {"span_jump": (v.loc[both, "span_surprisal_mean"] - base.loc[both, "span_surprisal_mean"]).mean(),
                    "post_jump": (v.loc[both, "post_surprisal_mean"] - base.loc[both, "post_surprisal_mean"]).mean(),
                    "n_items": len(both)}
    jumps = pd.DataFrame(jumps).T.assign(model=model).set_index("model", append=True)
    if {"PARTIAL", "SYN_LIT"} <= set(jumps.index.get_level_values(0)):
        put(model, e2_post_jump_PARTIAL=jumps.loc[("PARTIAL", model), "post_jump"],
            e2_post_jump_SYN_LIT=jumps.loc[("SYN_LIT", model), "post_jump"],
            e2_post_jump_PARTIAL_minus_SYN_LIT=jumps.loc[("PARTIAL", model), "post_jump"]
            - jumps.loc[("SYN_LIT", model), "post_jump"])
    return per_cond, jumps


def curves_vs_orig(df: pd.DataFrame, pool: str) -> pd.DataFrame:
    col = f"cos_{pool}_vs_ORIG"
    rows = df[df.condition != "ORIG"].dropna(subset=[col])
    out = {}
    for c, g in rows.groupby("condition"):
        arr = np.stack(g[col].to_numpy())
        out[c] = np.nanmean(arr, axis=0)
    out = pd.DataFrame(out)
    out.index.name = "layer"
    return out


def crossover(df: pd.DataFrame, pool: str, meaning: str, form: str, curves: pd.DataFrame) -> tuple:
    """First layer where `meaning` is closer to ORIG than `form`: on average, and for > 50% of items."""
    if not {meaning, form} <= set(curves):
        return None, None, None
    col = f"cos_{pool}_vs_ORIG"
    a = df[df.condition == meaning].dropna(subset=[col]).set_index("item_id")[col]
    b = df[df.condition == form].dropna(subset=[col]).set_index("item_id")[col]
    both = a.index.intersection(b.index)
    share = np.nanmean([np.array(a[i]) > np.array(b[i]) for i in both], axis=0) if len(both) else np.array([])
    mean_x = next((int(l) for l in curves.index if curves[meaning][l] > curves[form][l]), None)
    share_x = next((int(l) for l, v in enumerate(share) if v > 0.5), None)
    return mean_x, share_x, float(share.max()) if share.size else None


def e2_e3(tag: str) -> None:
    surpr, jumps, cross, curves_all = [], [], [], {}
    for d in models():
        df = load_probe(d, tag)
        if df is None:
            continue
        s, j = e2(df, d.name)
        surpr.append(s)
        jumps.append(j)
        n_layers = len(df.dropna(subset=["cos_span_vs_ORIG"]).iloc[0].cos_span_vs_ORIG) - 1
        for pool in POOLS:
            if f"cos_{pool}_vs_ORIG" not in df:
                continue
            curves = curves_vs_orig(df, pool)
            curves.to_csv(SUMMARY / f"e3_{d.name}_{pool}{tag}.csv")
            curves_all[(d.name, pool)] = curves
            for meaning, form in CONTRASTS:
                mx, sx, smax = crossover(df, pool, meaning, form, curves)
                cross.append({"model": d.name, "pool": pool, "contrast": f"{meaning} vs {form}",
                              "first_layer_mean": mx, "first_layer_majority": sx, "n_layers": n_layers,
                              "rel_depth_majority": None if sx is None else sx / n_layers,
                              "max_share_items": smax})
                if pool == "span":
                    put(d.name, **{f"e3_span_{meaning}_vs_{form}_rel_depth": None if sx is None else sx / n_layers})
    if not surpr:
        return
    save(pd.concat(surpr), f"e2_surprisal{tag}")
    save(pd.concat(jumps), f"e2_jump_vs_ORIG{tag}")
    save(pd.DataFrame(cross).set_index(["model", "pool", "contrast"]), f"e3_crossover{tag}")
    for pool in POOLS:
        e3_plot({m: c for (m, p), c in curves_all.items() if p == pool}, pool, tag)


def e3_plot(lines: dict, pool: str, tag: str) -> None:
    if not lines:
        return
    names = order_models(lines)
    cols = min(4, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False, sharey=True)
    for ax, model in zip(axes.flat, names):
        curves = lines[model]
        depth = np.linspace(0, 1, len(curves))
        for c in ("SYN_LIT", "PARTIAL", "NEUTRAL", "LIT_CONTEXT", "RAND"):
            if c in curves:
                ax.plot(depth, curves[c], label=c, color=COLORS[c], lw=1.8,
                        ls="--" if c in ("NEUTRAL", "LIT_CONTEXT") else "-")
        ax.set_title(model, loc="left", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("relative layer depth")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"cosine to ORIG ({pool})")
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"E3: cosine to ORIG by layer, pooling = {pool}", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(SUMMARY / f"e3_similarity_{pool}{tag}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- E4
def e4(tag: str, folds: int = 5) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    results, curves = [], {}
    for d in models():
        files = sorted((d / f"lm_probe{tag}").glob("shard_*.npz"))
        meta = load_probe(d, tag)
        if not files or meta is None:
            continue
        X = np.concatenate([np.load(f)["X"] for f in files]).astype(np.float32)       # [N, L+1, D]
        vid = np.concatenate([np.load(f)["variant_id"] for f in files])
        m = meta.set_index("variant_id").loc[vid]
        lit_items = set(m.item_id[m.condition == "LIT_CONTEXT"])
        for pos_cond in ("ORIG", "NEUTRAL"):
            sel = m.condition.isin([pos_cond, "LIT_CONTEXT"]).to_numpy() & m.item_id.isin(lit_items).to_numpy()
            y = (m.condition[sel] == pos_cond).astype(int).to_numpy()
            groups = m.idiom_id[sel].to_numpy()
            k = min(folds, len(set(groups)))
            if k < 2 or len(set(y)) < 2:
                continue
            y_shuf = np.random.default_rng(42).permutation(y)
            aucs, ctrl = [], []
            for layer in range(X.shape[1]):
                Xl = X[sel, layer]
                pred, pred_c = np.zeros(len(y)), np.zeros(len(y))
                for tr, te in GroupKFold(k).split(Xl, y, groups):
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
                    pred[te] = clf.fit(Xl[tr], y[tr]).predict_proba(Xl[te])[:, 1]
                    pred_c[te] = clf.fit(Xl[tr], y_shuf[tr]).predict_proba(Xl[te])[:, 1]
                aucs.append(roc_auc_score(y, pred))
                ctrl.append(roc_auc_score(y_shuf, pred_c))
            best = int(np.argmax(aucs))
            n_layers = X.shape[1] - 1
            results.append({"model": d.name, "contrast": f"{pos_cond} vs LIT_CONTEXT", "n": int(sel.sum()),
                            "idioms": len(set(groups)), "best_layer": best, "n_layers": n_layers,
                            "best_rel_depth": best / n_layers, "best_auc": aucs[best],
                            "control_auc_at_best": ctrl[best], "auc_layer0": aucs[0], "auc_last": aucs[-1]})
            curves[(d.name, pos_cond)] = aucs
            put(d.name, **{f"e4_{pos_cond}_vs_LIT_best_auc": aucs[best], f"e4_{pos_cond}_vs_LIT_best_rel_depth":
                           best / n_layers})
    if not results:
        return
    save(pd.DataFrame(results).set_index(["model", "contrast"]), f"e4_probe{tag}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, pos in zip(axes, ("ORIG", "NEUTRAL")):
        for model in order_models({m for m, p in curves if p == pos}):
            aucs = curves[(model, pos)]
            ax.plot(np.linspace(0, 1, len(aucs)), aucs, lw=1.6, label=model)
        ax.axhline(0.5, color="#6b7280", lw=1, ls="--")
        ax.set_title(f"E4: {pos} (figurative) vs LIT_CONTEXT (literal)", loc="left", fontsize=10)
        ax.set_xlabel("relative layer depth")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ROC-AUC, unseen idioms (chance 0.5)")
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(SUMMARY / f"e4_probe{tag}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="'_limitN' to analyse a quick test run")
    a = ap.parse_args()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    e1()
    e2_e3(a.tag)
    e4(a.tag)
    if METRICS:
        save(pd.DataFrame(METRICS).T.reindex(order_models(METRICS)), f"summary_metrics{a.tag}")
    print(f"\ntables and figures -> {SUMMARY}")

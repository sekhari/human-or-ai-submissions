"""Phase 1 exploratory data analysis.

Usage:
    python -m src.eda                      # writes to results/eda/
    python -m src.eda --skip-slow          # skip near-duplicate + adversarial validation

Outputs (results/eda/):
    summary.md                 key numbers from every section
    style_features.csv         per-document hand-crafted features (all splits)
    feature_label_auroc.csv    single-feature AUROC for human vs AI (train)
    feature_shift.csv          train vs test distribution shift per feature (KS)
    clusters.csv               topic clusters: size per split, AI rate, top terms
    near_duplicates.csv        document pairs with cosine >= threshold
    top_tokens.csv             most label-discriminative tokens
    oov_top_test.csv           test docs with the most out-of-train-vocabulary tokens
    figures/*.png
"""
import argparse
import time
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from src.data import load_all, load_split, near_duplicate_pairs
from src.features import HOMOGLYPH_RE, LAYOUT_FEATURES, WORD_RE, build_features

ROOT = Path(__file__).resolve().parents[1]
SPLIT_COLORS = {"train": "#4C72B0", "unlabeled": "#55A868", "test": "#C44E52"}
LABEL_NAMES = {0: "human", 1: "AI"}


class Report:
    """Collects markdown lines and echoes them to stdout."""

    def __init__(self):
        self.lines = []

    def h(self, text):
        self.add(f"\n## {text}\n")

    def add(self, text=""):
        print(text)
        self.lines.append(text)

    def table(self, df, floatfmt=".3f"):
        self.add(df.to_markdown(floatfmt=floatfmt))
        self.add()

    def save(self, path):
        path.write_text("\n".join(self.lines) + "\n")


def savefig(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"{name}.png", dpi=130)
    plt.close(fig)


def directional_auc(y, x):
    """AUROC of a single feature, flipped so it is >= 0.5; sign tells the direction."""
    auc = roc_auc_score(y, x)
    return (auc, "+AI") if auc >= 0.5 else (1 - auc, "+human")


# ---------------------------------------------------------------- sections
def section_overview(df, rep, data_dir):
    rep.h("1. Overview & integrity")
    summary = df.groupby("split").agg(
        rows=("id", "size"), id_min=("id", "min"), id_max=("id", "max"),
        null_text=("text", lambda s: s.isna().sum()),
        empty_text=("text", lambda s: (s.str.strip() == "").sum()),
        ai_rate=("label", "mean"))
    rep.table(summary)
    rep.add(f"- duplicate ids overall: {df.id.duplicated().sum()}")
    starter = load_split("train_starter_2000", data_dir)
    train = df[df.split == "train"].set_index("id")
    in_train = starter.id.isin(train.index)
    same = (train.loc[starter.id[in_train], "text"].values == starter.text[in_train].values).all()
    rep.add(f"- train_starter_2000: {in_train.sum()}/{len(starter)} ids in train.csv, identical text: {same}")
    sub = load_split("sample_submission", data_dir)
    test_ids = set(df.loc[df.split == "test", "id"])
    rep.add(f"- sample_submission: {len(sub)} rows, columns {list(sub.columns)}, ids match test: {set(sub.id) == test_ids}")


def section_length(df, feats, rep, out_dir):
    rep.h("2. Length")
    tmp = df[["split", "label"]].assign(words=feats.n_words, chars=feats.n_chars)
    tmp["group"] = np.where(tmp.split == "train", tmp.label.map(LABEL_NAMES), tmp.split)
    rep.table(tmp.groupby("group").words.describe())
    tr = tmp[tmp.split == "train"]
    for col in ("words", "chars"):
        auc, d = directional_auc(tr.label, tr[col])
        rep.add(f"- length-only AUROC ({col}): {auc:.3f} ({d}) -- how much a pure length shortcut is worth")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.arange(70, 360, 10)
    for lab, name in LABEL_NAMES.items():
        axes[0].hist(tr.words[tr.label == lab], bins=bins, alpha=0.55, label=name)
    axes[0].set(title="Train: word count by label", xlabel="words", ylabel="docs")
    axes[0].legend()
    for split, c in SPLIT_COLORS.items():
        axes[1].hist(tmp.words[tmp.split == split], bins=bins, histtype="step", lw=2,
                     density=True, label=split, color=c)
    axes[1].set(title="Word count by split (density)", xlabel="words")
    axes[1].legend()
    savefig(fig, out_dir, "length_distribution")

    # AI rate per length bin: is there a length/label interaction?
    tr = tr.assign(bin=pd.qcut(tr.words, 5))
    rep.add("\nAI rate per word-count quintile (train):\n")
    rep.table(tr.groupby("bin", observed=True).label.agg(["size", "mean"]))


def section_style(df, feats, rep, out_dir):
    rep.h("3. Stylometric / formatting features")
    tr = df.split == "train"
    rows = []
    for col in feats.columns:
        x = feats.loc[tr, col]
        if x.nunique() < 2:
            continue
        auc, d = directional_auc(df.loc[tr, "label"], x)
        rows.append({"feature": col, "auroc": auc, "direction": d,
                     "mean_human": x[df.label == 0].mean(), "mean_ai": x[df.label == 1].mean()})
    auc_df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    auc_df.to_csv(out_dir / "feature_label_auroc.csv", index=False)
    rep.add("Top 15 single-feature AUROCs (human vs AI, train):\n")
    rep.table(auc_df.head(15).set_index("feature"))

    # all features together, as a quick ceiling for a style-only model
    y = df.loc[tr, "label"].astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), feats[tr], y,
                          cv=cv, method="predict_proba")[:, 1]
    rep.add(f"- all style features, HistGradientBoosting 5-fold CV AUROC: {roc_auc_score(y, p):.3f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    top = auc_df.head(20).iloc[::-1]
    ax.barh(top.feature, top.auroc, color=["#C44E52" if d == "+AI" else "#4C72B0" for d in top.direction])
    ax.axvline(0.5, color="k", lw=0.8)
    ax.set(title="Single-feature AUROC (red: higher => AI)", xlim=(0.45, 1))
    savefig(fig, out_dir, "feature_auroc")

    # train vs test shift per feature
    shift = []
    for col in feats.columns:
        a, b = feats.loc[tr, col], feats.loc[df.split == "test", col]
        shift.append({"feature": col, "ks_stat": ks_2samp(a, b).statistic,
                      "mean_train": a.mean(), "mean_test": b.mean()})
    shift = pd.DataFrame(shift).sort_values("ks_stat", ascending=False)
    shift.to_csv(out_dir / "feature_shift.csv", index=False)
    rep.add("\nLargest train-vs-test feature shifts (KS statistic):\n")
    rep.table(shift.head(10).set_index("feature"))


def section_shortcuts(df, feats, rep):
    rep.h("3b. Formatting shortcut check")
    rep.add("Line-break layout may be an artefact of how each source was collected rather than of authorship;\n"
            "if test passages were re-formatted (or adversarially edited) such a shortcut would not transfer.\n")
    flags = pd.DataFrame({
        "has_blank_line": feats.blank_lines > 0,
        "has_newline": feats.n_lines > 1,
        "has_markdown": feats.markdown_marks > 0,
        "likely_non_english": feats.stopword_ratio < 0.2,
    })
    group = np.where(df.split == "train", "train_" + df.label.map(LABEL_NAMES), df.split)
    rep.table(flags.groupby(group).mean())

    tr = df.split == "train"
    y = df.loc[tr, "label"].astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, cols in [("all style features", list(feats.columns)),
                       ("without line-layout features", [c for c in feats.columns if c not in LAYOUT_FEATURES])]:
        p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), feats.loc[tr, cols], y,
                              cv=cv, method="predict_proba")[:, 1]
        rep.add(f"- {name}: 5-fold CV AUROC {roc_auc_score(y, p):.3f}")
    has_blank = feats.loc[tr, "blank_lines"] > 0
    rep.add(f"- P(AI | train doc has a blank line) = {y[has_blank].mean():.3f} (n={has_blank.sum()})")


def section_unicode(df, feats, rep):
    rep.h("4. Unicode & possible adversarial artefacts")
    flags = pd.DataFrame({
        "split": df.split, "label": df.label,
        "has_non_ascii": feats.non_ascii_ratio > 0,
        "has_zero_width": feats.zero_width > 0,
        "has_homoglyph": feats.homoglyph_words > 0,
        "has_curly_quote": feats.p_curly_quote > 0,
        "has_emdash": feats.p_emdash > 0,
    })
    flags["group"] = np.where(flags.split == "train", "train_" + flags.label.map(LABEL_NAMES), flags.split)
    rep.add("Share of documents with each artefact:\n")
    rep.table(flags.groupby("group")[[c for c in flags if c.startswith("has_")]].mean())

    for split in ("train", "test"):
        chars = Counter(c for t in df.loc[df.split == split, "text"] for c in t if ord(c) > 127)
        top = ", ".join(f"{c!r}({unicodedata.name(c, '?')[:18]})x{n}" for c, n in chars.most_common(12))
        rep.add(f"- most common non-ASCII chars in {split}: {top}")
    ex = df[(df.split == "test") & (feats.homoglyph_words > 0)]
    for _, r in ex.head(3).iterrows():
        rep.add(f"- test id {r.id} homoglyph words: {HOMOGLYPH_RE.findall(r.text)[:5]}")


def section_tokens(df, rep, out_dir):
    rep.h("5. Label-discriminative tokens (train, smoothed log-odds)")
    tr = df[df.split == "train"]
    cv = CountVectorizer(lowercase=True, min_df=20, binary=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    X = cv.fit_transform(tr.text)
    y = tr.label.values.astype(int)
    ai = np.asarray(X[y == 1].sum(0)).ravel() + 1
    hu = np.asarray(X[y == 0].sum(0)).ravel() + 1
    logodds = np.log(ai / (y == 1).sum()) - np.log(hu / (y == 0).sum())
    vocab = np.array(cv.get_feature_names_out())
    out = pd.DataFrame({"token": vocab, "log_odds_ai": logodds, "docs_ai": ai - 1, "docs_human": hu - 1})
    out.sort_values("log_odds_ai").to_csv(out_dir / "top_tokens.csv", index=False)
    order = np.argsort(logodds)
    rep.add("- most AI-leaning: " + ", ".join(f"`{vocab[i]}`" for i in order[::-1][:30]))
    rep.add("- most human-leaning: " + ", ".join(f"`{vocab[i]}`" for i in order[:30]))


def section_clusters(df, rep, out_dir, k, seed):
    rep.h(f"6. Topic / domain clusters (TF-IDF -> SVD -> KMeans, k={k})")
    tfidf = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5, max_features=30000, sublinear_tf=True)
    X = tfidf.fit_transform(df.text)
    svd = TruncatedSVD(100, random_state=seed)
    Z = Normalizer().fit_transform(svd.fit_transform(X))
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    df = df.assign(cluster=km.labels_)

    terms = np.array(tfidf.get_feature_names_out())
    centroids = svd.inverse_transform(km.cluster_centers_)
    counts = pd.crosstab(df.cluster, df.split)[["train", "unlabeled", "test"]]
    share = counts / counts.sum(0)
    out = pd.DataFrame({
        "n_train": counts.train, "n_unlabeled": counts.unlabeled, "n_test": counts.test,
        "test_over_train": (share.test / share.train.replace(0, np.nan)).round(2),
        "ai_rate_train": df[df.split == "train"].groupby("cluster").label.mean(),
        "top_terms": [", ".join(terms[np.argsort(c)[::-1][:8]]) for c in centroids],
    })
    out.to_csv(out_dir / "clusters.csv")
    rep.add("`test_over_train` > 1 means the topic is over-represented in test (possible unseen domain).\n")
    rep.table(out, floatfmt=".2f")

    fig, ax = plt.subplots(figsize=(11, 4))
    share.plot.bar(ax=ax, color=[SPLIT_COLORS[s] for s in share.columns])
    ax.set(title="Cluster share within each split", xlabel="cluster", ylabel="fraction of split")
    savefig(fig, out_dir, "cluster_share")
    df[["id", "split", "cluster"]].to_csv(out_dir / "cluster_assignments.csv", index=False)
    return df


def section_near_dups(df, rep, out_dir, threshold):
    rep.h(f"7. Exact and near duplicates (word 1-2gram TF-IDF cosine >= {threshold})")
    norm = df.text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    exact = df[norm.duplicated(keep=False)]
    rep.add(f"- exact duplicates (after lowercasing/whitespace): {len(exact)} docs, "
            f"by split: {exact.split.value_counts().to_dict()}")

    p = near_duplicate_pairs(df.text, threshold)
    if p.empty:
        rep.add("- no near-duplicate pairs found")
        return
    p["id_a"], p["id_b"] = df.id.values[p.i], df.id.values[p.j]
    p["split_a"], p["split_b"] = df.split.values[p.i], df.split.values[p.j]
    p["label_a"], p["label_b"] = df.label.values[p.i], df.label.values[p.j]
    p["kind"] = p[["split_a", "split_b"]].apply(lambda r: "-".join(sorted(r)), axis=1)
    p.drop(columns=["i", "j"]).sort_values("cosine", ascending=False).to_csv(out_dir / "near_duplicates.csv", index=False)
    rep.add(f"- near-duplicate pairs by split combination: {p.kind.value_counts().to_dict()}")
    tt = p[p.kind == "train-train"]
    if len(tt):
        rep.add(f"- train-train pairs with conflicting labels: {(tt.label_a != tt.label_b).sum()} / {len(tt)}"
                " (same source rewritten by AI? -> validation split should keep them together)")


def section_adversarial(df, feats, rep, seed):
    rep.h("8. Adversarial validation (can a model tell train from test / unlabeled?)")
    rep.add("AUROC ~0.5 = same distribution; high AUROC = strong shift.\n")
    rows = []
    for other in ("test", "unlabeled"):
        m = df.split.isin(["train", other])
        y = (df.loc[m, "split"] == other).astype(int)
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        text_model = make_pipeline(TfidfVectorizer(min_df=3, sublinear_tf=True, max_features=50000),
                                   LogisticRegression(max_iter=2000, C=1.0))
        p_text = cross_val_predict(text_model, df.loc[m, "text"], y, cv=cv, method="predict_proba")[:, 1]
        p_style = cross_val_predict(HistGradientBoostingClassifier(random_state=seed), feats[m], y,
                                    cv=cv, method="predict_proba")[:, 1]
        rows.append({"train_vs": other, "auroc_word_tfidf": roc_auc_score(y, p_text),
                     "auroc_style": roc_auc_score(y, p_style)})
        if other == "test":
            # train docs that look most like test are a candidate "test-like" validation subset
            train_scores = pd.Series(p_text[(y == 0).values], index=df.loc[m & (df.split == "train"), "id"])
            rep.add(f"- train docs with P(test) > 0.5: {(train_scores > 0.5).sum()}")
            top = TfidfVectorizer(min_df=3, sublinear_tf=True, max_features=50000)
            Xt = top.fit_transform(df.loc[m, "text"])
            lr = LogisticRegression(max_iter=2000).fit(Xt, y)
            vocab = np.array(top.get_feature_names_out())
            order = np.argsort(lr.coef_[0])
            rep.add("- words most indicative of test: " + ", ".join(f"`{w}`" for w in vocab[order[::-1][:25]]))
            rep.add("- words most indicative of train: " + ", ".join(f"`{w}`" for w in vocab[order[:25]]))
    rep.add()
    rep.table(pd.DataFrame(rows).set_index("train_vs"))


def section_oov(df, rep, out_dir):
    rep.h("9. Out-of-vocabulary rate (typos / unseen domains)")
    train_vocab = Counter(w.lower() for t in df.loc[df.split == "train", "text"] for w in WORD_RE.findall(t))
    vocab = {w for w, n in train_vocab.items() if n >= 2}

    def oov(text):
        words = [w.lower() for w in WORD_RE.findall(text)]
        missing = [w for w in words if w not in vocab]
        return len(missing) / max(len(words), 1), missing

    res = df.text.map(oov)
    df = df.assign(oov_rate=res.str[0], oov_words=res.str[1])
    group = np.where(df.split == "train", "train_" + df.label.map(LABEL_NAMES), df.split)
    rep.add("OOV rate w.r.t. train vocabulary (count >= 2; train itself is optimistic by construction):\n")
    rep.table(df.groupby(group).oov_rate.describe())
    top = df[df.split == "test"].nlargest(30, "oov_rate")
    top.assign(oov_sample=top.oov_words.map(lambda w: " ".join(w[:15])), preview=top.text.str[:120])[
        ["id", "oov_rate", "oov_sample", "preview"]].to_csv(out_dir / "oov_top_test.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    for g in ["unlabeled", "test"]:
        ax.hist(df.oov_rate[group == g], bins=50, range=(0, 0.3), histtype="step", lw=2, density=True,
                label=g, color=SPLIT_COLORS[g])
    ax.set(title="OOV rate vs train vocabulary", xlabel="fraction of words unseen in train")
    ax.legend()
    savefig(fig, out_dir, "oov_rate")


def section_examples(df, rep, seed):
    rep.h("10. Random examples (first 200 chars)")
    groups = [("train human", (df.split == "train") & (df.label == 0)),
              ("train AI", (df.split == "train") & (df.label == 1)),
              ("test", df.split == "test")]
    for name, m in groups:
        rep.add(f"\n**{name}**\n")
        for _, r in df[m].sample(5, random_state=seed).iterrows():
            rep.add(f"- id {r.id}: `{r.text[:200]!r}`")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=ROOT / "Data", type=Path)
    ap.add_argument("--out-dir", default=ROOT / "results" / "eda", type=Path)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--clusters", default=12, type=int)
    ap.add_argument("--dup-threshold", default=0.8, type=float)
    ap.add_argument("--skip-slow", action="store_true", help="skip near-duplicates and adversarial validation")
    args = ap.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rep = Report()
    rep.add("# EDA summary\n")
    rep.add(f"Generated by `python -m src.eda` (seed={args.seed}).")
    df = load_all(args.data_dir)
    feats = build_features(df)
    pd.concat([df[["id", "split", "label"]], feats], axis=1).to_csv(args.out_dir / "style_features.csv", index=False)

    section_overview(df, rep, args.data_dir)
    section_length(df, feats, rep, args.out_dir)
    section_style(df, feats, rep, args.out_dir)
    section_shortcuts(df, feats, rep)
    section_unicode(df, feats, rep)
    section_tokens(df, rep, args.out_dir)
    section_clusters(df, rep, args.out_dir, args.clusters, args.seed)
    if not args.skip_slow:
        section_near_dups(df, rep, args.out_dir, args.dup_threshold)
        section_adversarial(df, feats, rep, args.seed)
    section_oov(df, rep, args.out_dir)
    section_examples(df, rep, args.seed)

    rep.add(f"\n_Runtime: {time.time() - t0:.0f}s_")
    rep.save(args.out_dir / "summary.md")
    print(f"\nWrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

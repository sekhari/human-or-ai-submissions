"""Paired blend experiment."""

import gc
if 'final_model' in globals():
    del final_model
gc.collect()
torch.cuda.empty_cache()
trainer = train_or_load(fit, "validation")

paired = pd.read_csv(run / "v2_validation_probes.csv.gz", keep_default_na=False)
expected = val.set_index("id")
assert paired["variant"].nunique() == 4
rows = []

def standardized(values):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all() and values.std() > 0
    return (values - values.mean()) / values.std()

for variant, group in paired.groupby("variant", sort=False):
    g = group.set_index("id")
    assert g.index.is_unique and set(g.index) == set(expected.index)
    g = g.loc[expected.index]
    assert np.array_equal(g["label"], expected["label"])
    if variant == "original":
        assert np.array_equal(g["text"], expected["text"])

    v2 = standardized(g["v2_score"])
    transformer = standardized(margins(trainer, g["text"].tolist()))
    blend = 0.75 * v2 + 0.25 * transformer
    rows.append({
        "variant": variant,
        "v2": roc_auc_score(g["label"], v2),
        "transformer": roc_auc_score(g["label"], transformer),
        "blend_75_25": roc_auc_score(g["label"], blend),
    })

result = pd.DataFrame(rows).set_index("variant")
result.loc["worst_case"] = result.min()
result["blend_minus_v2"] = result["blend_75_25"] - result["v2"]
result.to_csv(run / "paired_blend_auc.csv")
display(result.round(6))

probes = pd.read_csv(run / "v2_validation_probes.csv.gz", keep_default_na=False)
char_scores = pd.read_csv(run / "char_validation_scores.csv")
paired = probes.merge(char_scores, on=["id", "variant"], validate="one_to_one")
assert len(paired) == len(probes) == 8000

expected = val.set_index("id")
rows = []

for variant, group in paired.groupby("variant", sort=False):
    g = group.set_index("id")
    assert g.index.is_unique and set(g.index) == set(expected.index)
    g = g.loc[expected.index]
    assert np.array_equal(g["label"], expected["label"])

    transformer = standardized(margins(trainer, g["text"].tolist()))
    current = 0.75 * standardized(g["v2_score"]) + 0.25 * transformer
    candidate = 0.80 * standardized(current) + 0.20 * standardized(g["char_score"])

    rows.append({
        "variant": variant,
        "current": roc_auc_score(g["label"], current),
        "with_char": roc_auc_score(g["label"], candidate),
    })

comparison = pd.DataFrame(rows).set_index("variant")
previous = pd.read_csv(run / "paired_blend_auc.csv", index_col="variant")
assert np.allclose(
    comparison["current"],
    previous.loc[comparison.index, "blend_75_25"],
    atol=0.00001, rtol=0,
), "The current blend did not reproduce the previous validation run."

comparison.loc["worst_case"] = comparison.min()
comparison["gain"] = comparison["with_char"] - comparison["current"]
comparison.to_csv(run / "character_blend_validation.csv")
display(comparison.round(6))

# Set RUN_TRAIN_DIAGNOSTIC=1 to measure the in-sample AUROC of the validation fit.
if os.environ.get("RUN_TRAIN_DIAGNOSTIC") == "1":
    train_v2 = pd.read_csv(run / "v2_train_scores.csv").set_index("id")
    assert train_v2.index.is_unique and set(train_v2.index) == set(fit.id)
    train_margin = 0.75 * standardized(train_v2.loc[fit.id, "v2_score"]) + 0.25 * standardized(
        margins(trainer, fit.text.tolist()))
    train_auc = roc_auc_score(fit.label, train_margin)
    val_auc = result.loc["original", "blend_75_25"]
    pd.DataFrame([{"train_auc": train_auc, "validation_auc": val_auc, "gap": train_auc - val_auc}]).to_csv(
        run / "selected_train_validation_auc.csv", index=False)

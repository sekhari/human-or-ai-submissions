# Normalized lexical features and teacher transfer

Issue #15 implements the fixed normalization, lexical comparison, style-teacher
transfer, and six-candidate blend ablation in notebook 15. It reuses notebook 10's
readable NB/original-text definitions and notebook 13's exact 147-feature schema.
These dependencies are merged. All fitting stays inside original training rows.

## Lexical arms

NFKC, canonical quotes/dashes/ellipsis, removal of Unicode Cf, whitespace collapse,
and lowercase are deterministic per passage. Word1-3 caps at 120,000 features and
char3-5 at 180,000, both min_df 2 and float32. Three training folds select the
binary NB-weighted C1 arm. TF-IDF uses C16. The two-view TF-IDF arm trains original
and normalized copies at .5 weight each and averages their inference margins.
All folds refit their full features and classifier. All three training AUROCs are
1.0, which is not held-out performance.

| Arm | Mean training CV AUROC | Validation AUROC | Full fit seconds |
|---|---:|---:|---:|
| tfidf | 0.952118 | 0.956511 | 27.046 |
| nb | 0.961977 | 0.966657 | 29.104 |
| two_view | 0.958631 | 0.963252 | 48.950 |

## Teacher transfer

The original-style HistGB teacher fits internal 6,400 rows only. It accepts 2,205
of 3,000 eligible unlabeled rows at confidence .995: 1,012 human and 1,193 AI,
with pseudo weighted mass 551.25. Control and treatment independently fit the same
verified-training vocabulary. Only the treatment's NB ratio also sees assigned
pseudo labels, unweighted in class token frequencies. Classifier pseudo weight
stays .25. This differs from #7's frozen-IDF classifier-only update.

On internal 1,600 rows, normalized NB control AUROC is .959956 and transfer is
.964239, a point gain of .004283. No interval is claimed. Both train AUROCs are
1.0. Accepted pseudo labels are unknown truth and can copy teacher mistakes.

## Six-way ablation

Style uses clipped logit probabilities, original text uses two standardized
margins plus conditional blank-line offset 2, and NB uses the transfer treatment's
margin. Reference means/stds come from eligible unlabeled text, never test batches.
The original-text specialist reference excludes literal blank-line rows. The
rule means literal two consecutive newlines, not notebook 13's broader blank-line
regex. It activates only when no human fit row has that literal pattern.

All blend settings and guardrails are fixed before fitting. Select maximum
original internal AUROC, with remove-blank and collapse each at least hist-.002.
Exact ties prefer pure hist. The selected weights are .5 style, .25 original
text, .25 transferred NB. Pure original text fails the remove-blank guardrail.

| Candidate | Original | Remove blank lines | Collapse whitespace | Eligible |
|---|---:|---:|---:|---|
| hist | 0.997975 | 0.982770 | 0.809845 | True |
| hist75_old25 | 0.998808 | 0.987973 | 0.852427 | True |
| hist50_old50 | 0.998969 | 0.988691 | 0.880247 | True |
| hist75_nb25 | 0.998900 | 0.992081 | 0.906537 | True |
| hist50_old25_nb25 | 0.999022 | 0.992444 | 0.931547 | True |
| old_soft | 0.994148 | 0.957148 | 0.908414 | False |

Selections are saved before frozen validation. Fresh 8,000-row components confirm
original validation AUROC .998492, remove-blank .992831, and collapse .931158.
Formatting sensitivity remains despite the strong original score. Full-data
teacher transfer accepts 2,156 rows with pseudo weighted mass 539. This is a
confirmed ablation, not selection of the final competition model or a Kaggle score.
Repeated validation reuse and uncalibrated blended scores limit interpretation.

## Reproduce and hand off

Recorded run time is 852.772 seconds; the internal section takes 171.078 seconds
and full blend refit 135.936 seconds. Model reloads match exactly. An independent
Python process matches all 2,000 saved class-1 probabilities exactly and verifies
all artifact hashes without notebook globals, fits, or downloads. Unicode,
normalization idempotence, stress transformation, and frozen transfer vocabulary
checks passed. Notebook schema/syntax and diff checks passed. The first worktree
preflight failure is preserved with null AUROC; its missing audit copy was
restored by the pinned hash before the successful run.

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=3600 notebooks/15_normalized_transfer.ipynb
uv run --locked python tools/check_notebooks.py
uv run --locked python tools/check_text_handoff.py 15_normalized_transfer
```

The safe record is `results/15_normalized_transfer/20261110T202717584850Z.json`.
`artifacts/15_normalized_transfer/latest.json` names lexical models, the full
ablation blend, aligned predictions, hashes, and `refit_spec.json`. The specification
includes exact schema, settings, weights, rule state, and all normalization stats.
Method definitions stay in notebook 15; a full run rebuilds from raw files.

For #17 selection, use the saved `internal_style`, `internal_old_soft`, and
`internal_nb_treatment` models, or fresh internal refits. Check the ordered internal
ID hashes first. Do not use the full 8,000-fit model on internal 1,600 rows, since
it has seen those labels. The augmented-style experiment must repeat its selection
on internal rows as specified in #17. Notebook 18 still needs standalone final
refits from raw files, not past joblib artifacts. No report template was opened.

All pseudo-label rows, raw passages, model files, feature schema artifacts, and
row predictions remain ignored. Teammate review is requested. No upload occurs.

# Equal-word prefix whitespace diagnostic — November 9, 2026

**The same 100 words produce very different rankings when their original
whitespace is preserved.** This post-hoc, zero-fit diagnostic follows the
[completed seven-fit study](roberta_context_results_20261109.md). It was motivated
by the first fold's clean/whitespace/prefix contrast; the separate rule and
source hashes were frozen before the canonical model or this diagnostic
produced its results. It is exploratory development evidence, not an untouched
confirmation or a competition-score estimate.

Use the fixed final-epoch `canonical_8000` checkpoint, trained on the original
8,000 fitting IDs. The same 1,627 eligible held originals (857 human, 770 AI)
have at least 150 whitespace words. No new model fitting, optimizer update,
checkpoint selection, seed search or test access occurred.

| Input on the identical eligible original IDs | RoBERTa AUROC |
| --- | ---: |
| Original full passage | 0.999327 |
| First 100 words, original leading/interword whitespace preserved | 0.990948 |
| The same first 100 words, joined with one ASCII space | 0.782456 |

For the preserved variant, take the original substring through the end of the
100th non-whitespace word. Both prefix variants have exactly the same word
sequence, punctuation, case and original ID; only leading/interword whitespace
differs. All 1,627 prefixes are fully retained in both variants at the 512-token
limit, so this contrast is not caused by tokenizer truncation. Average full BPE
counts are 139.20 preserved and 132.41 joined, including special tokens.

Preserved minus joined AUROC is **+0.208492**, with paired class-stratified
original-ID bootstrap interval **[+0.184160, +0.233630]** (1,000/1,000 valid draws,
seed551). This interval is conditional on one fitted model and these reused
originals; it excludes training variation and post-hoc selection adjustment.
The intervention changes the tokenizer representation and positions as a
consequence of changing whitespace; it does not isolate individual newline,
indentation or repeated-space mechanisms.

Original whitespace differs from the joined variant on 1,212 originals. The
0.5-threshold classification changes on 20.84% of all eligible originals; mean
absolute P(AI) change is 0.20513. These are sensitivity descriptions, not a
threshold chosen from the held labels. Recomputing every joined-prefix margin
and probability from the saved checkpoint reproduces the primary predictions
with maximum error **0.0** before the preserved variant is evaluated.

The path-specific point differences are 0.008379 from full text to preserved
prefix and 0.208492 from preserved to joined prefix. They support attributing
most of this constructed prefix test's loss to the whitespace intervention on
these same originals, rather than interpreting it as a general inability to
classify 100-word text. They do not establish that the model uses only layout,
that layout is an invalid authorship signal, or that all future short texts
behave the same way.

The next focused training hypothesis is to normalize whitespace before both
fitting and inference while retaining the encoder, labels, folds, optimizer,
epochs and seed. Whether that retains the clean/topic gains requires a new
controlled fit; it is not answered by this inference-only diagnostic. The
current raw-text candidate and its unfavorable sensitivity results remain
available as measured tradeoffs, without a new quality gate or blend claim.

Execution took 87.75 native-process seconds on MPS in `python_course`. Python
warnings and native stderr are empty; no unsupported-op CPU fallback was
observed. The independent standard-library audit checks equal-word IDs/hashes,
all cached joined predictions, both AUROCs, every paired bootstrap replicate
and the interval, checkpoint/source/log hashes, and zero training updates.
It passed without loading a model or running another inference.

Evidence under
`results/supervised/a_roberta_context_20261109_v1_study/`:

- `exploratory_prefix_whitespace_plan.json` and `exploratory_source/`: frozen
  adaptive hypothesis and source copies.
- `exploratory_prefix_whitespace/predictions.csv`, `metrics.json`,
  `paired_bootstrap.json`, `joined_reproduction_receipt.json` and
  `independent_verification.json`: complete numeric evidence, with no passages.
- `postprocess_execution_receipts.json`: actual command, timing and native-log
  hashes. The scientific comparison and its source hashes are unchanged.

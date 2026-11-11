# Reframing the score-improvement search

Execution update: the [seven-fit RoBERTa comparison](roberta_context_results_20261109.md)
and [equal-word prefix diagnostic](roberta_prefix_whitespace_results_20261109.md)
are now complete in `python_course` on the M4 Pro. Clean random/topic macro
AUROCs improve to 0.996673/0.995652, with substantial whitespace sensitivity.
The diagnostic isolates a 0.208492 AUROC difference from preserving whitespace
at fixed 100-word content. The historical research-stage statements below refer
to the earlier review; they are not the current execution status.

Status: research and a zero-fit signal inventory, November 9, 2026. No new
classifier training, model download, dependency installation, test-text access
or candidate export occurred in this review. Compute budget is not yet confirmed;
the initially inspected Anaconda base Python lacks PyTorch, Transformers and
Tokenizers. The user subsequently selected `python_course` for all model
training; that environment already has PyTorch and Transformers installed.

Hardware follow-up: the user confirmed an M4 Pro Mac; a read-only probe verified
48 GiB unified memory and 14 CPU cores. The
[M4 Pro refinement](m4pro_model_strategy_20261109.md) now prioritizes RoBERTa-base
fine-tuning as a local candidate. MPS runtime/throughput have not been tested.

## Main recommendation

Prioritize **a genuinely different information source and its complementarity
with WC120**. The largest untested representation family is a generic pretrained
contextual encoder, fine-tuned only on the supplied training labels. Retain the
current WC120 as an anchor, then test whether contextual or language-model
probability signals improve its difficult rankings. This is a research judgment,
not evidence that a particular architecture will improve the hidden score.

The previous crop/normalization studies establish useful sensitivities and
tradeoffs, but do not establish that matching those constructed conditions is
the main route to better competition predictions. A smaller clean-development
score can still accompany a useful cross-domain gain; an arbitrary clean-loss
cutoff should not define model validity.

## What the inventory changes

The original eight Q1 classifier families mostly share word TF-IDF or its
128-dimensional SVD. The earlier neural model was:

`word TF-IDF -> SVD128 -> StandardScaler -> MLP128`

It was not a pretrained language encoder. Changing the classifier after this
same representation did not explore contextual syntax/semantics or token
predictability learned from large-scale language pretraining.

The new ID-aligned inventory used 13 saved prediction sets on the reused
2,000-row development set. No fitting or new test predictions were involved.
WC120's rank correlation is 0.9876 with C60 and 0.9944 with quote WC120. These
similarities suggest limited new signal from mixing those near variants, though
correlation alone cannot prove that an ensemble is useless.

Fixed blends with existing nonlinear representation controls were measured:

| Existing development prediction | AUROC | Difference from WC120 |
| --- | ---: | ---: |
| WC120 | 0.970006 | reference |
| WC120 + SVD MLP, equal probability weights | 0.966774 | -0.003232 |
| WC120 + SVD boosting, equal probability weights | 0.970595 | +0.000589 |
| WC120 + SVD boosting, equal percentile-rank weights | 0.968070 | -0.001936 |

This is exploratory reuse of development labels, not OOF-selected blending or
new submission evidence. The tiny best gain does not justify treating arbitrary
mixing of the existing models as a demonstrated solution. Models with different
errors can also introduce many new errors; low correlation is not enough.
The rank transform here depends on the evaluation cohort and is not a frozen
production transform. All candidate-export flags are false.

Artifacts: `experiments/supervised/signal_inventory_20261109.py` and
`results/supervised/signal_inventory_20261109/`, including input/source hashes,
model inventory, pairwise error diagnostics and fixed-blend results.

## Candidate routes, ranked by the question they answer

| Route | New information | Minimal discriminating experiment | Main limitation |
| --- | --- | --- | --- |
| Generic pretrained encoder | Contextual representation rather than only local counts | One fixed small checkpoint: frozen features + LR as a throughput/probe reference, then one fixed supervised fine-tuning recipe; compare standalone and fixed combination with WC120 on matched random/topic splits | Can still learn topics/source artifacts; frozen-probe failure does not rule out fine-tuning |
| Small local causal-LM probability features | Conditional token surprise, rank and distributional shape | Freeze one small generic LM; extract document-level NLL summaries, surprise variation and token-rank fractions; fit a simple supervised head and test WC120 complementarity | Predictability depends on subject, human style and text length; lower perplexity is not an AI label |
| BPE subwords + NB-weighted linear classifier | A different token unit and label-derived weighting | Fit BPE only inside fitting folds; compare the same binary subword counts with/without NB log-count-ratio weighting | Still a surface representation; overlap with character features may be large and OOD gains are unproved |
| Group-aware data use / supplied unlabeled learning | Changes coverage or influence of examples rather than just model capacity | First isolate class-conditional topic-balanced weights, or isolate unlabeled representation learning before adding one pseudo-label round | Proxy groups are imperfect; unlabeled data may cover familiar sources and high-confidence pseudo-labels can reinforce shortcuts |

### Contextual model: first choice if acceleration is available

One reasonable initial checkpoint is
[`distilbert/distilroberta-base`](https://huggingface.co/distilbert/distilroberta-base),
a generic six-layer language model intended for downstream fine-tuning. This
introduces pretrained language representation without using an external
AI-detector's supervised labels. The exact checkpoint revision, license and
all task-specific fitting should be recorded. A contextual model may still
encode unwanted domain directions; this is documented in
[Restricted Embeddings](https://aclanthology.org/2024.findings-emnlp.992/).
That paper's improvement figures are not estimates for this competition.

Start with one checkpoint and one training recipe, retaining identical supplied
label IDs and complete fold-contained fitting. An initial throughput check on
a small training-only subset can establish feasible batch size/device settings;
it must not decide which scientific result to report. A full three-fold
comparison comes after feasibility, without a broad architecture/seed search.

Do not select a new model only by whether its standalone clean AUROC exceeds
0.970006. A useful test is whether it corrects WC120's mistakes on held topics
and contributes to a predeclared fixed combination. Begin with one fixed blend;
if learning combination weights later, learn them from fitting-only OOF
predictions and assess them outside that weight-fitting boundary.

### Probability signal: distinct, with a smaller-model starting point

[GLTR](https://aclanthology.org/P19-3019/) motivates probability, rank and entropy
features for generated-text analysis. Its early-generator and human-assistance
results do not establish modern AI-detector AUROC. A compact generic causal LM
is a practical signal probe; summarize probabilities locally and retain only
document-level features. Fit any scaling/head using permitted training rows.

[Binoculars](https://proceedings.mlr.press/v235/hans24a.html) uses a contrast
between two related pretrained language models. The
[official implementation](https://github.com/ahans30/Binoculars) defaults to
two Falcon7B variants, so it is not the first CPU route. Smaller-model
substitutions are new experiments, not reproductions of its reported results or
thresholds. There is direct counterevidence against a universal perplexity fix:
[SINAI's SemEval study](https://aclanthology.org/2024.semeval-1.216/) examines
Transformer/perplexity methods and substantial training/test distribution
differences. Treat this as a possible complementary signal, not a magic rule.

### CPU route: subwords, with a matched weighting control

BPE changes the representation unit; it should be investigated as a low-cost
branch, not sold as a proven competition trick. The
[Tokenizers components documentation](https://huggingface.co/docs/tokenizers/en/components)
describes the available mechanisms. The matched weighting experiment is
motivated by [Wang and Manning](https://aclanthology.org/P12-2018/); its original
sentiment/topic task results do not establish gains here.

Preserve the ordinary binary-count control to isolate NB weighting. Learn
tokenizer merges, vocabulary, feature filtering, class ratios and calibration
inside their permitted fitting boundaries. Do not copy transductive recipes
that learn from competition test text or import external AI-labelled corpora.

## Data coverage is a separate hypothesis

The existing marginal audit finds supplied unlabeled data closer to labeled
data than test data on digits/newlines. This does not prove source identity,
but gives no assurance that the 3,000 unlabeled rows fill unseen test coverage.
Keep U0 supervised, U1 additional unlabeled representation learning, and U2
additional pseudo-labeling separate. Coordinate the existing Q3 work before
implementing a duplicate study; the local placeholder is not proof that a
teammate has done no work.

Topic groups also differ in size and difficulty. If revisiting data weighting,
hold each class's total weight fixed, report effective sample size and subgroup
counts, and change only the weighting rule. A harder group can be informative
or noisy; simply maximizing the worst proxy-group result is not automatically
better for hidden global AUROC.

## A score-oriented selection process

Use a small candidate panel rather than repairing WC120 one feature at a time:
one contextual candidate, one probability or subword candidate, and WC120.
For each, retain clean random-CV, topic-proxy CV, natural-length subgroups and
fixed sensitivity scores, plus error/rank complementarity and cost.

Keep the non-dominated tradeoffs visible. A candidate improving topic transfer
while modestly losing clean performance deserves comparison, not automatic
rejection at a nominal 0.003 boundary. Conversely a large prefix-stress gain
does not by itself forecast a hidden-test gain. Uncertainty and model-selection
reuse should accompany all rankings; no new untouched holdout has appeared.

Global AUROC includes positive/negative pairs from different domains. It is not
just an average of within-domain AUROCs, and unknown test composition does not
justify inventing a clean/topic weighted score. Public/private percentages also
do not specify familiar/unfamiliar-domain percentages. Evaluate any final
combination's actual global rankings, with weights selected away from the
evaluation labels.

The confirmed M4 Pro/48-GiB hardware makes a local contextual-encoder experiment
the first route to measure, with small-LM probability features as an independent
signal branch. BPE/NB remains a cheaper reference, rather than the primary route
selected from an assumed CPU-only limit. Actual backend support and runtime must
be measured before committing a multi-fold budget. The common objective
is new useful information, followed by validated complementarity with WC120.
No route presently has a verified public/private gain, and no numerical score
increase should be promised before an experiment.

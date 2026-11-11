# Experiment figures and external report writing

The team writes its report in a separate Overleaf project. Do not generate or
upload report prose, TeX section stubs, report build entrypoints, or report PDFs
to this code repository. Later writing assistance should use the actual
implementation, configurations and measured results.

This directory retains only experimental figures and this workflow note.
Figures are named with question/run identifiers and refer to recorded runs.
The expanded Q1 figures can be regenerated from existing evidence without
fitting models or generating report text:

```sh
python -m experiments.supervised.plot_expanded_q1 --prefix a_q1_expanded_20261108_v2
```

Exact grids, selected settings, metrics and uncertainty estimates are in
`results/supervised/`; reproduction commands and dependencies are in
`experiments/supervised/README.md`. Private AUROC is recorded only after release.
Keep Overleaf access/share links and local report drafts outside uploaded files.


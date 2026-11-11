# Notebook format

1. Goal:two sentences max.
2. Setup:imports,seed,paths,configuration.
3. Checks:allowed IDs,data hashes,split.
4. Method:actual readable feature/model code.
5. Run:time the fit or search.
6. Results:measured tables/plots,including losing arms.
7. Takeaway:three bullets max,tried/found/next.
8. Save:metadata and safe aggregates.

Scaffolds are plans,not finished experiments. Replace NotImplementedError only after implementation.
Do not fill outputs with target scores. No raw passage or row-prediction displays in committed notebooks.
Use repo-relative paths and fresh kernels. Notebook02 exposes loaders,evaluation,and recording helpers
through %run ./02_split_metrics.ipynb without running fits or downloads.
Save one results/NN_name/run-ID.json per run. Include status,Git SHA,UTC time,command,
versions/lock hash,data/split/config hashes,seed,fit/eval sizes,metrics,runtime,artifact hashes.
Final full-data runs use validation_auroc=null and link their separate held-out confirmation.
Keep models,predictions,and private executed copies under ignored artifacts/ or submissions/.

uv sync --locked
uv run jupyter lab

Use nbconvert with a fresh kernel to execute completed notebooks. Respect issue dependencies.
Keep optional neural packages isolated with a recorded version set.

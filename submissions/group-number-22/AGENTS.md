# Working rules

Read PROJECT_CONTEXT.md, docs/notebook-contract.md, and your issue first.

## Style
- Fourth-year undergrad. About20% caveman: short,direct,clear.
- No em dashes. No hype. Explain what we tried,why,and what happened.
- Simple Python. Short comments explain why. Keep exact settings and useful details.

## Workflow
- Experiment code belongs in notebooks,including features,fitting,evaluation,and plots.
- Shared notebook02 helpers must not train or download on import. Do not hide models in large Python modules.
- One issue per branch:codex/issue-N-short-name. One PR per issue. Merge after local checks pass. The user waived teammate review on November 10, 2026.
- Aim for2-3 useful commits per issue. Never pad,backdate,spoof authors,or manufacture history.
- Respect dependencies and notebook ownership. Keep shared files coordinated.
- Restart and Run All before marking done. Save real metrics and failed experiments.

## Integrity
- Fit learned transforms,tuning,and calibration inside training only. Never fit or pseudo-label validation/test rows during development.
- Use supplied verified labels only. No ID features,no hidden-label access,no remote passage-classification service.
- AUROC is primary. Keep the fixed split. Do not keep trying seeds until a result looks good.
- Keep secrets,raw text,models,and row predictions ignored. Commit safe aggregates only.
- Record source revision,data/split hashes,versions,settings,seed,timing,and measured results.
- Do not claim an experiment ran until it did. A score target is not a measurement.
- Jalil owns uploads. Team quota is shared,five/day. No automatic Kaggle uploads.
- Follow course rules for collaboration and required final disclosures.
- Report template remains deferred. Do not inspect,edit,render,or compile it before the final report stage.

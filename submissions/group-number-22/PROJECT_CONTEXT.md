# Human or AI text classification

Group Number22: Jalil-g,rdevasia25,Gtpatel02.
Predict P(AI),human0,AI1. Metric:AUROC. Goal:public AUROC>=.98,with valid experiments.
Baseline and learning-curve experiments have run on the frozen split. Baseline validation AUROC: 0.932728. See docs/word-baseline.md and docs/learning-curve.md. The public-score target remains a goal; no Kaggle result has been measured in this repository.

Data:train10000 balanced,unlabeled3000,test3000,starter2000,sample3000.
Starter is a training subset,not extra labels. Reserve stratified8000/2000,seed42.
Fit learned features and tune inside training. Final chosen model can refit on all10000.
Public leaderboard900 rows;private2100. Private score remains unknown until released.
Competition slug:cornell-tech-cs-5785-combined-2026-ai-vs-human-text.

Plan:baseline,learning curve,word/character comparisons,self-training,error analysis,
style features,expanded text models,formatting rules,seed/stress checks,augmentation,
normalized lexical models,local ELECTRA,and controlled blends. Keep negative results.

Original allocation was six issues each. Issue #11 transferred from Gtpatel02 to rdevasia25. See docs/work-plan.md. Commit meaningful work,not equal-count padding.
Jalil coordinates Kaggle submissions. Five/day is shared across the whole team.
Supplied deadline:November11,2026. Confirm exact cutoff,portal,and final-selection mechanics.
Report work starts only at the final stage. Follow required course disclosure rules in final deliverables.

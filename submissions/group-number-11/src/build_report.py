"""Generate the editable report from measured experiment logs and team metadata."""
import json
import subprocess
from pathlib import Path
from train import ROOT


def load(name):
    return json.loads((ROOT/'results'/name).read_text())


def esc(value):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#',
             '_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in str(value))


def table(headers, rows, spec=None):
    spec=spec or ('l'+'r'*(len(headers)-1))
    return ('\\begin{tabularx}{\\linewidth}{@{}'+spec+'@{}}\n\\toprule\n'
        +' & '.join(headers)+r' \\ \midrule'+'\n'
        +'\n'.join(' & '.join(str(c) for c in row)+r' \\' for row in rows)
        +'\n\\bottomrule\n\\end{tabularx}\n')


def main():
    meta=json.loads((ROOT/'report_metadata.json').read_text())
    ex=load('experiments.json');sel=load('selection.json');a=load('analysis.json')
    curve=load('learning_curve.json');unl=load('unlabeled.json');rob=load('robustness.json');imp=load('improvement.json')
    errors=json.loads((ROOT/'artifacts/errors_for_review.json').read_text())
    get=lambda n:next(r for r in ex if r['name']==n)
    fmt=lambda x:f'{x:.6f}'
    prior=sel.get('previous_best',sel['best'])
    hypotheses={
        961:'Short film review with broad evaluative statements; these overlap AI review language despite colloquial spelling.',
        215:'Polished, aphoristic review prose and parallel sentences may resemble generated commentary.',
        7384:'Short review with repeated movie judgments; little text supports reliable style estimation.',
        9403:'Technical exposition with repeated specialist terms; lexical repetition may be mistaken for AI regularity.',
        921:'Formal definitional prose and repeated conceptual terms resemble generated explanations.',
        3061:'Conversational first-person parenting discussion, quoted post framing and irregular phrasing resemble human forum text.',
        2515:'Recipe instructions mixed with blog attribution and quoted titles resemble scraped human source material.',
        9298:'An encyclopedia-like biography with dates, names and irregular spacing resembles human reference prose.',
        5603:'Informal first-person game discussion and specific gaming details resemble a personal forum post.',
        5498:'Ingredient quantities and procedural line breaks resemble human recipe formatting.'}
    pr=meta['pr_url']
    pr_text=r'\url{'+pr+'}' if pr else 'Pending: no course PR has been opened from this workspace.'
    public=f"{meta['new_public_auc']:.5f}" if meta['new_public_auc'] is not None else 'Not yet submitted'
    private=fmt(meta['private_auc']) if meta['private_auc'] is not None else 'Not revealed'
    interpretation=('The public AUROC '+public+' was supplied in a Kaggle screenshot by the team. '
        f"It is {a['val_auc']-meta['new_public_auc']:.6f} below validation, so random-split performance is optimistic for the public subset. "
        'Formatting sensitivity is a measured limitation, but the cause of the public gap is not established. Private performance remains unknown.'
        if meta['new_public_auc'] is not None else
        'A high in-distribution validation AUROC does not establish a 0.98--0.99 public or private score. Formatting sensitivity is a measured limitation. No new Kaggle result has been supplied.')
    status=('The team uploaded the new-model CSV and reported public AUROC '+public+
        '. The previous public score was 0.92433. Private results remain pending. See the title page for the course PR.'
        if meta.get('kaggle_uploaded_by_user') else
        'The CSV and report are prepared locally; a new Kaggle upload and its results remain pending. See the title page for the course PR.')
    parts=[r'''\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,xcolor,tabularx,booktabs,array,enumitem,hyperref,fancyhdr,lastpage,titlesec,amsmath,microtype,needspace,ragged2e,graphicx}
\definecolor{navy}{HTML}{19334C}
\definecolor{teal}{HTML}{236B82}
\definecolor{muted}{HTML}{52616B}
\hypersetup{colorlinks=true,linkcolor=teal,urlcolor=teal,pdftitle={Group Number 11: Human or AI?}}
\titleformat{\section}{\Large\bfseries\color{navy}}{\thesection.}{0.6em}{}
\titlespacing*{\section}{0pt}{14pt}{7pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}\setlength{\headheight}{15pt}
\renewcommand{\arraystretch}{1.12}
\newcolumntype{Y}{>{\RaggedRight\arraybackslash}X}
\newcommand{\q}[2]{\needspace{4\baselineskip}\par\addvspace{7pt}\textbf{#1}\quad\textit{#2}\par\nobreak}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\color{muted}Cornell Tech | Applied Machine Learning}
\fancyhead[R]{\small\color{muted}Group Number 11}
\fancyfoot[C]{\small\color{muted}Page \thepage\ of \pageref{LastPage}}
\begin{document}
\thispagestyle{empty}
{\color{teal}\bfseries CORNELL TECH | APPLIED MACHINE LEARNING}\par
\vspace{0.18in}
{\Huge\bfseries\color{navy}Human or AI?}\par
\vspace{0.07in}
{\LARGE\color{navy}Style-aware text classification}\par
\vspace{0.16in}
\textbf{Kaggle competition report | Group Number 11}\par
''',esc(meta['report_date'])+r'\par\vspace{0.1in}',
        table(['Team member','NetID'],[[esc(m['name']),r'\texttt{'+m['netid']+'}'] for m in meta['members']],'Yr'),
        r'\textbf{GitHub PR:} '+pr_text+r'\par',
        r'\textbf{Submission branch:} \texttt{'+esc(meta['submission_branch'])+r'}. Development branch: \texttt{'+esc(meta['development_branch'])+r'}.',
        r'\section*{Team information and model summary}',
        r'\q{Q0.1}{Team and contributions.}'+esc(meta['contributions']),
        r'''\q{Q0.2}{Final approach.}
We represent each passage with word TF-IDF, character TF-IDF, and handcrafted writing-style statistics.
The selected classifier averages a word linear SVM, character logistic regression and style-feature LightGBM with weights 0.125, 0.125 and 0.75.
All model selection uses one fixed stratified 2,000-row validation set; the final models are then refitted on all 10,000 labeled passages.
IDs are identifiers only; no external labels, pretrained weights or external prediction APIs are used.
The output is a continuous AI-authorship score in $[0,1]$ for every test ID.
\q{Q0.3}{Recorded scores.}''',
        table(['Metric','AUROC'],[
            ['Selected fixed validation',fmt(a['val_auc'])],
            ['Previous Kaggle public (user-reported, earlier model)',fmt(meta['previous_public_auc'])],
            ['New-model Kaggle public',public],['Kaggle private',private]],'Yr'),
        r'\textbf{Interpretation.} '+interpretation+r'''
\newpage
\section{Model comparison and selection}
\q{Q1.1}{Preprocessing and representations.}
We decode literal backslash-n and backslash-t sequences into actual whitespace without discarding case, punctuation or paragraph boundaries.
Word TF-IDF uses scikit-learn's default word token pattern, 1--2 grams, minimum document frequency 2, maximum document frequency 0.995, Unicode accent stripping, sublinear term frequency, L2 normalization and at most 180,000 features.
Character TF-IDF uses 2--5 grams, minimum document frequency 3, sublinear term frequency, L2 normalization and the same feature cap.
Our custom preprocessor preserves case, including for word features.
Style features measure lexical diversity, hapax frequency, word/sentence lengths, punctuation, paragraph structure, function-word rates and several simple regex patterns.
These represent complementary lexical and structural evidence.
Each vocabulary and IDF transform is fitted on training texts only; row-local style extraction learns no corpus statistics.
\q{Q1.2}{Model comparison on the same validation set.}''',
        table(['Model / features','Train AUC','Val. AUC','Seconds'],[
            ['Word LR, $C=4$',fmt(get('word_lr_4')['train_auc']),fmt(get('word_lr_4')['val_auc']),f"{get('word_lr_4')['seconds']:.2f}"],
            ['Word linear SVM, $C=1$',fmt(get('word_svm_1')['train_auc']),fmt(get('word_svm_1')['val_auc']),f"{get('word_svm_1')['seconds']:.2f}"],
            ['Character LR, $C=16$',fmt(get('char_lr_16')['train_auc']),fmt(get('char_lr_16')['val_auc']),f"{get('char_lr_16')['seconds']:.2f}"],
            ['Word NB-weighted LR, $C=4$',fmt(get('word_nb_4')['train_auc']),fmt(get('word_nb_4')['val_auc']),f"{get('word_nb_4')['seconds']:.2f}"],
            ['Style LightGBM, 31 leaves / 700 trees',fmt(get('style_31_700')['train_auc']),fmt(get('style_31_700')['val_auc']),f"{get('style_31_700')['seconds']:.2f}"],
            ['Selected three-component blend',fmt(a['train_auc']),fmt(a['val_auc']),'Shared fits']], 'Yrrr'),
        r'''\small Times are local CPU feature-construction plus classifier-fit wall time, excluding scoring and checkpoint I/O; shared feature construction is attributed to each setting. They are approximate, not a hardware benchmark.\normalsize
\q{Q1.3}{Hyperparameters and selection protocol.}''',
        table(['Approach','Candidates','Selected'],[
            ['Word LR / SVM',r'$C\in\{0.1,1,4,16\}$','SVM, $C=1$'],
            ['Character LR / SVM',r'$C\in\{0.1,1,4,16\}$','LR, $C=16$'],
            ['NB-weighted LR (both representations)',r'$C\in\{0.1,1,4\}$','Not retained'],
            ['Style LightGBM',r'Leaves $\{7,15,31\}$; trees $\{300,700\}$','31 / 700']], 'YYl'),
        r'''Seed 20261023 fixes a stratified 8,000/2,000 split. We compare 28 base settings and select the highest validation AUC within each representation.
LightGBM uses learning rate 0.03, minimum child size 40, feature fraction 0.85 and L2 regularization 5.
We evaluate 15 blend-grid entries: style weight $s\in\{0,.25,.5,.75,1\}$, with remaining weight divided between word and character components using $c\in\{0,.5,1\}$. Thus weights are $((1-s)(1-c),(1-s)c,s)$; pure-style endpoints repeat.
No Kaggle feedback or test labels select settings. Reusing one validation set can still produce selection optimism.
\q{Q1.4}{Final model choice.}
The selected blend scores 0.997975, versus 0.997234 for style boosting alone and 0.977193 for the best character model.
It adds complementary lexical evidence while remaining laptop-friendly.
The word/style blend scores 0.997969, only six concordant pairs per million below the selected blend; this is effectively a near tie, not persuasive evidence of superiority.
We retain the grid winner consistently rather than infer a meaningful difference.
The SVM margin is passed through a sigmoid for blending; it is not calibrated, and the combined score should not be read as individual authorship certainty.
All selected components are refitted on the full labeled set only after selection.
\newpage
\section{Effect of training data size}
\q{Q2.1}{Controlled learning curve.}
We fix word logistic regression at $C=4$ and use stratified samples from the same 8,000-row training partition with seed 20261023. Each size gets a freshly fitted vocabulary and IDF, preventing smaller training sets from borrowing the full corpus representation. The 2,000 validation examples are unchanged. The smaller subsets are sampled separately; nesting is not required for this comparison.
''',
        table(['Training rows','Training AUROC','Validation AUROC','Seconds'],[
            [r['n'],fmt(r['train_auc']),fmt(r['val_auc']),f"{r['seconds']:.2f}"] for r in curve]),
        r'''\q{Q2.2}{Learning-curve figure.}
\begin{center}\includegraphics[width=.96\linewidth]{learning_curve.pdf}\end{center}
\q{Q2.3}{Interpretation.}
Training AUC is 1.0 at every size, whereas validation AUC increases from 0.853259 at 500 examples to 0.949536 at 8,000.
This is an overfitting pattern: a high-dimensional linear classifier separates its own examples even when the learned vocabulary and weights do not generalize as well.
The increase from 5,000 to 8,000 examples is about 0.0100, so additional labeled data remains useful for this configuration.
The curve does not show substantial training underfitting and does not establish that this model has saturated.
Only one sample per size is measured; the curve has sampling variability, and no asymptotic rate is inferred.
It is a learning curve for word LR, not evidence that the final style-heavy blend would improve by the same amount.
\newpage
\section{Learning from unlabeled data}
\q{Q3.1}{Method and leakage prevention.}
We test unsupervised representation learning using the additional 3,000 unlabeled texts.
The supervised baseline fits word vocabulary and IDF only on the 8,000 training texts.
The alternate representation fits vocabulary and IDF on these same texts plus all 3,000 unlabeled texts.
Both classifiers use logistic regression with $C=4$ and train on exactly the same 8,000 true labels.
Neither validation texts nor validation labels enter either representation fit, and test texts are excluded.
This experiment does not create pseudo-labels or assume unlabeled authorship.
The fixed holdout isolates the effect of changing corpus statistics, although it cannot establish whether the same change would help hidden test domains.
\q{Q3.2}{Measured comparison.}''',
        table(['Method','Validation AUROC'],[
            ['Supervised word LR only',fmt(unl['baseline_auc'])],
            ['Training + unlabeled vocabulary/IDF',fmt(unl['unlabeled_auc'])]],'Yr'),
        r'''\q{Q3.3}{Did unlabeled data help?}
No improvement was observed: validation AUC changes by ''',f"{unl['unlabeled_auc']-unl['baseline_auc']:+.6f}",r'''.
This difference is tiny and should be interpreted as essentially neutral rather than strong evidence of harm.
Additional documents change vocabulary coverage and IDF weights but supply no new class information.
They may add domain vocabulary that is not useful for authorship discrimination; this is a hypothesis, not a measured causal explanation.
Pseudo-label noise cannot explain this result because no pseudo-labels were used.
The final submission therefore uses the supervised representation fits.
A future self-training experiment would need a fixed confidence rule, comparison on the same holdout, and explicit checks for confirmation bias rather than an assumption that more unlabeled data always helps.
\vspace{.15in}
\textbf{Data accounting.} The starter file is a subset of the supplied labeled set and is never appended as additional training data. The 3,000 test texts are used only for inference. No hidden labels are recovered from the dataset's external source or from external model services.
\newpage
\section{Generalization and error analysis}
\q{Q4.1}{Training--validation gap and uncertainty.}
The selected blend has training AUC ''',fmt(a['train_auc']),r''' and validation AUC ''',fmt(a['val_auc']),r'''.
The small ranking gap does not imply source or generator independence: both partitions are random subsets of the same labeled pool.
A 1,000-replicate class-stratified bootstrap gives a conditional validation interval of [''',f"{a['bootstrap_95_ci'][0]:.6f}, {a['bootstrap_95_ci'][1]:.6f}",r'''].
It holds the selected fitted model fixed and omits model-selection and distribution-shift uncertainty.
No exact text duplicates cross the split. Two validation passages have nearest-training word-TF-IDF cosine similarity at least 0.90; excluding them leaves AUC 0.997971.
This similarity check reduces concern about those particular passages but is not a full near-duplicate or shared-source audit.
\q{Q4.2}{Observable text-length subgroups.}
We count whitespace-delimited words after decoding escaped line breaks. The fixed bins test whether the detector relies on passage length or becomes unreliable on shorter passages, without pretending to know hidden domains or generator identities.
''',table(['Words','Count','Human / AI','AUROC'],[
            [esc(g['bin']),g['n'],f"{g['human']} / {g['ai']}",fmt(g['auc'])] for g in a['length_bins']],'Yrrr'),
        r'''All three bins have AUC near 0.998, with the longest bin slightly higher. Differences of this size and unequal class counts do not establish a reliable subgroup advantage. Length is only one observable grouping; it does not test new writing domains, unseen generators or adversarial rewriting. In particular, strong within-bin performance can coexist with reliance on paragraph formatting and lexical-diversity artifacts.
\begin{center}\includegraphics[width=\linewidth]{diagnostics.pdf}\end{center}
\q{Q4.3}{Threshold and error selection.}
At threshold 0.5, the confusion matrix is TN=993, FP=7, FN=24, TP=976 (accuracy 0.9845).
False positives are human passages with score at least 0.5; false negatives are AI passages below 0.5.
The next page shows the five highest-scoring false positives and five lowest-scoring false negatives. These are deliberately extreme cases, not a representative error sample.
\newpage
{\Large\bfseries\color{navy}Generalization: concrete errors}\par
The observed content descriptions below come from reviewing the supplied validation passages. Proposed explanations are hypotheses; we did not intervene on each passage to establish causation.
''',table(['ID','Type','P(AI)','Observed content and possible failure mechanism'],[
            [e['id'],e['type'],f"{e['probability']:.4f}",esc(hypotheses[e['id']])] for e in errors],'rrrY'),
        r'''\q{Q4.4}{Recurring patterns and possible remedies.}
Among these selected errors, three human false positives are film reviews and two are formal explanatory passages.
Two AI false negatives are recipe passages; other AI misses use conversational, biographical or game-discussion framing.
These observations show that familiar genre conventions overlap between labels; they do not identify the hidden generator or prove that a passage was paraphrased.
Short reviews also offer few tokens for reliable style estimation.
One remedy is label-preserving formatting augmentation, tested in Q5, so line breaks become less decisive.
Another is to add more diverse labeled examples of these genres, but that proposal is untested and external supervised labels are outside this experiment's scope.
Our feature-gain analysis identifies blank-line rate, lexical diversity and newline frequency as important model inputs.
Gain importance measures how this fitted tree model uses features, not a causal property of human or AI writing.
The whitespace stress test provides stronger evidence that formatting changes can damage the current model.
\newpage
\section{Open-ended improvements and insights}
\q{Q5.1}{Hypothesis.}
Lexical models and style features should make different errors, so their blend may improve ranking over either representation alone.
The fixed-holdout results support this as a useful experiment: the character model is substantially weaker than style boosting but can still contribute complementary information.
After the blend experiment, feature-gain inspection showed paragraph-format dependence. We therefore also tested whether whitespace augmentation could preserve native-text AUC while improving robustness to a formatting shift.
This second question matters because random splits retain acquisition and formatting regularities that may not survive different preprocessing or adversarial edits.
\q{Q5.2}{Exact changes and controlled comparison.}
For the blend improvement, we keep the labeled split and trained components fixed and change only the mixture weights, using the finite grid described in Q1.3.
For the augmentation follow-up, each training passage contributes its original and whitespace-collapsed version with the same supplied label.
The 8,000 source training passages become 16,000 transformed training rows, not additional independent labeled examples.
We compare 15- and 31-leaf style models with 700 trees and style weights 0.75 or 1.0; lexical components stay fixed.
Before executing this follow-up, we specified an acceptance rule: native AUC must be within 0.0005 of the original blend and collapsed-text AUC must be at least 0.98. Eligible candidates would be ranked by native AUC.
\q{Q5.3}{Measured results.}''',
        table(['Configuration','Native AUC','Collapsed AUC'],[
            ['Character LR alone',fmt(get('char_lr_16')['val_auc']),'Not isolated'],
            ['Style boosting alone',fmt(get('style_31_700')['val_auc']),'Not isolated'],
            ['Selected three-component blend',fmt(sel['selected']['val_auc']),fmt(next(r['auc'] for r in rob['stress_tests'] if r['transformation']=='collapse_whitespace'))],
            ['Augmented 31-leaf style + lexical blend',fmt(next(r['val_auc'] for r in imp['candidates'] if r['leaves']==31 and r['style_weight']==.75)),fmt(next(r['collapsed_auc'] for r in imp['candidates'] if r['leaves']==31 and r['style_weight']==.75))]],'Yrr'),
        r'''The selected blend reuses the component fits from Q1.2; prediction adds three component passes.
The 31-leaf augmented candidate takes ''',f"{next(r['seconds'] for r in imp['candidates'] if r['leaves']==31 and r['style_weight']==.75):.2f}",r''' seconds for feature construction and fitting with cached original features; this excludes the already fitted lexical components and is not a full-pipeline speed comparison.
Removing size and sentence/paragraph-count features from the original style model yields AUC ''',fmt(rob['ablation']['ablated_auc']),r''' (baseline 0.997234). This partial ablation still retains newline-character frequency, so it does not establish formatting independence.
\q{Q5.4}{What we learned and limitations.}
Blending gives a small native-text ranking improvement, but it does not solve formatting dependence.
Whitespace augmentation raises collapsed-text AUC from 0.877512 to 0.978602 for the strongest native-text augmented candidate, while reducing native AUC from 0.997975 to 0.997503.
No candidate passes the prespecified 0.98 collapsed-text requirement, so the original model remains the final submission model.
ASCII punctuation normalization retains AUC 0.997797, and bracket-reference removal retains 0.997425; whitespace is the clearly larger measured vulnerability.
The three frozen-setting secondary random splits score ''',', '.join(fmt(r['ensemble_auc']) for r in rob['secondary_splits']),r'''. They support random-split stability but overlap the selection pool and do not constitute external validation.
These results support the narrow claim that the model ranks supplied unmodified validation passages well.
They do not establish robustness to hidden adversarial transformations or a specific Kaggle score.
A next experiment would canonicalize formatting throughout training and inference and compare its worst-transformation AUC under a separately reserved evaluation protocol.
\newpage
\section{Code, reproducibility and AI assistance}
\q{Q6.1}{PR, environment and entrypoints.}
\textbf{PR URL:} ''',pr_text,r'''
\textbf{Location:} \texttt{submissions/group-number-11/}.\par
CPU experiments used macOS arm64, Python 3.14.2 and four compute threads. Exact package versions are recorded in \texttt{environment.txt}; install dependencies from \texttt{requirements.txt}. Main split and model seed: 20261023. Secondary split seeds: 20261024, 20261025, 20261026. LightGBM has deterministic mode enabled; small numeric differences across platforms remain possible.
\begin{quote}\small
\begin{verbatim}
DATA_DIR=cornell-tech-cs-5785-combined-2026-ai-vs-human-text
python submissions/group-number-11/src/train.py \
  --data "$DATA_DIR" --final
python submissions/group-number-11/src/predict.py \
  --test "$DATA_DIR/test.csv" \
  --sample "$DATA_DIR/sample_submission.csv" \
  --out submissions/group-number-11/output/submission.csv
python submissions/group-number-11/src/verify.py \
  --data "$DATA_DIR"
\end{verbatim}
\end{quote}
The team's \texttt{reproduce.sh} entrypoint runs the full training, robustness, augmentation, analysis and report pipeline; invoke it via Bash from the repository root with the data-directory argument. The README provides the exact command and setup instructions. Validation metrics, selection logs, data checksums and the final submission checksum are retained. Large caches, model checkpoints and competition data remain local and are excluded from Git.
\q{Q6.2}{Development history and lessons.}
''']
    commits=subprocess.check_output(['git','log','--reverse','--format=%h %s','--','submissions/group11','submissions/group-number-11'],cwd=ROOT.parents[1]).decode().strip().splitlines()
    parts.append(r'\begin{itemize}[leftmargin=1.5em,itemsep=3pt]')
    for line in commits[:3]:
        commit_hash,title=line.split(' ',1)
        commit_base=meta.get('repository_url') or (pr.split('/pull/')[0] if pr else None)
        link=(r'\href{'+commit_base+'/commit/'+commit_hash+'}{'+commit_hash+'}') if commit_base else r'\texttt{'+commit_hash+'}'
        parts.append(r'\item '+link+': '+esc(title))
    parts.extend([r'''\end{itemize}
The initial pipeline established one reusable holdout and showed that style features outperform sparse lexical baselines on this data.
The generalization audit showed that the score survives removing two high-similarity validation examples but fails a whitespace-collapse stress test.
The augmentation follow-up measured a substantial robustness gain that still missed its predefined acceptance criterion. These lessons are recorded as actual experiments rather than retrospective claims that every change helped.
Commit hashes identify development history; inspect the hosted PR or use \texttt{git show HASH} locally.
\q{Q6.3}{AI assistance and verification.}
OpenAI Codex assisted Yingzi Gu with pipeline design, code, debugging, experiment execution, analysis and report drafting.
Experiments were run locally on the supplied data; offline scores come from their logs, while the public score was supplied by the team.
Automated checks verify the fixed split, validation labels/AUC, data hashes, output IDs/order and finite probability bounds.
No external LLM API was used to label or predict passages; no pretrained model weights or external supervised examples were used.
The team supplied the individual roles recorded in Q0.1, and reports that all three members also trained models and explored stronger configurations.
This describes the team's reported division of work, while the implementation and experiment execution in this rebuilt branch were assisted by Codex.
Before course submission, the team needs to review the code, interpretations and latest leaderboard result.
\vspace{.1in}
\textbf{Sources.} The supplied \emph{Kaggle Competition Template Report} PDF/LaTeX provides the protocol and Q0--Q6 requirements; the course README provides the repository layout. Offline results come from saved experiment logs; public results were supplied by the team. No external label source was consulted.
\vspace{.1in}
\textbf{Submission status.} ''',status,r'''
\end{document}
'''])
    text='\n'.join(parts)
    disclosures=meta.get('individual_ai_disclosures',[])
    if disclosures:
        marker=text.index(r'\vspace{.1in}',text.index(r'\q{Q6.3}'))
        individual=('The individual disclosures below also cover earlier experiments retained in the merged team history.\n\n'
                    +'\n\n'.join(esc(item['statement']) for item in disclosures)+'\n\n')
        text=text[:marker]+individual+text[marker:]
    (ROOT/'report/main.tex').write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
    if meta.get('optimization_final'):
        from report_optimization import upgrade_report
        upgrade_report(meta)
    if disclosures:
        path=ROOT/'report/main.tex'
        marker=r'\q{Q6.3}{AI assistance and verification.}'
        page=r'\newpage'+'\n'+r'\section*{AI assistance and individual contributions}'+'\n'
        path.write_text(path.read_text().replace(marker,page+marker,1))
    print(ROOT/'report/main.tex')


if __name__=='__main__':main()

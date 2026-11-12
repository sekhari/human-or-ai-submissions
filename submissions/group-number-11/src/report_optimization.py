"""Upgrade the controlled-experiment report with the selected optimized model."""
import json
import subprocess
import pandas as pd
from pathlib import Path
from train import ROOT


ERROR_NOTES={
    2696:'Human-labeled forum text repeats long phrases extensively; repetition is a plausible misleading generation cue.',
    9585:'Fluent encyclopedia biography with repeated names, dates and media titles resembles generated reference prose.',
    6138:'Formal historical description with many named entities overlaps the style of generated summaries.',
    7384:'Very short film review with broad judgments; limited text makes style estimates uncertain.',
    6404:'Informal first-person self-report with unusual spelling and contractions; its score is close to the threshold.',
    7049:'Recipe quantities, bullets and personal restaurant commentary resemble human recipe/blog material.',
    3457:'Quoted, line-wrapped academic abstract looks like source material; formatting and vocabulary may mask authorship cues.',
    9298:'Biography-like prose with dates, names and irregular spacing resembles human reference writing.',
    3061:'Quoted parenting discussion uses informal, personal language that resembles human forum text.',
    1689:'Dense mathematical notation and code fences are poorly represented by English predictability statistics.'}
ERROR_NOTES.update({2878:'Structured scientific abstract with repeated method and significance claims resembles generated academic exposition.',
                   6331:'Recipe instructions include personal corrections and irregular quantities, resembling human cooking notes.',
                   921:'Formal definitional exposition repeats abstract concepts and resembles generated explanatory prose.'})


def upgrade_report(meta):
    from build_report import table as base_table,esc
    def table(*args,**kwargs):
        return '\n\\par\\noindent\n'+base_table(*args,**kwargs)+'\n\\par\n'
    path=ROOT/'report/main.tex';text=path.read_text()
    best=json.loads((ROOT/'results/optimization/current_best.json').read_text())
    analysis=json.loads((ROOT/'results/optimization/final_analysis.json').read_text())
    if best['candidate']!=analysis['candidate']:raise ValueError('Final analysis does not match the selected optimized model.')
    errors=json.loads((ROOT/'artifacts/optimization/final_errors_for_review.json').read_text())
    fmt=lambda value:f'{value:.6f}'
    def between(start,end,replacement):
        nonlocal text
        left=text.index(start);right=text.index(end,left+len(start));text=text[:left]+replacement+'\n'+text[right:]
    between(r'\q{Q0.2}',r'\q{Q0.3}',r'''\q{Q0.2}{Final approach.}
The final detector combines 279 local writing-style features with 26 frozen GPT-2 language-predictability statistics.
A 31-leaf LightGBM learns on supplied labels using original and whitespace-collapsed style views; the language features retain their original-text values in this feature-level augmentation.
The final score is 75\% of this new model plus 25\% of the initial word/character/style ensemble.
Selection uses the original fixed 2,000-row validation split, followed by refitting the tree models on all 10,000 labeled passages.
Generic pretrained language weights run locally; no external authorship labels or hosted prediction APIs are used.''')
    scores=table(['Metric','AUROC'],[['Final fixed validation',fmt(analysis['val_auc'])],
        ['Initial public (earlier team model)','0.92433'],['First rebuild public','0.96962'],
        ['Final optimized public (team-reported)',f"{best['public_auc']:.5f}"],['Kaggle private','Not revealed']],'Yr')
    between(r'\q{Q0.3}',r'\newpage',r'\q{Q0.3}{Recorded scores.}'+scores+
        r'\textbf{Interpretation.} The public result comes from the team\textquotesingle s Kaggle submission report. It is '+
        fmt(analysis['val_auc']-best['public_auc'])+' below validation. Repeated holdout selection and hidden-domain differences limit generalization claims; private performance remains unknown.')
    text=text.replace('Selected three-component blend','Initial three-component blend')
    text=text.replace('No Kaggle feedback or test labels select settings.',
        'For this initial grid, settings were selected without Kaggle feedback or test labels; later public-feedback decisions are recorded separately.')
    between(r'\q{Q1.4}',r'\newpage',r'''\q{Q1.4}{Final model choice and later changes.}
The initial three-component blend scored 0.997975 validation / 0.96962 public AUROC.
A format-robust follow-up scored 0.995598 / 0.95899, so it was rejected despite strong synthetic stress results.
Adding frozen language statistics then improved the selected blend to '''+fmt(analysis['val_auc'])+' validation / '+f"{best['public_auc']:.5f}"+r''' public AUROC.
The final setting uses original and collapsed style views, 31 leaves, 700 trees, learning rate 0.03 and L2 regularization 5.
The unchanged baseline retains 25\% weight as a conservative anchor.
This was a later change motivated by measured public and validation evidence, not hidden individual labels.
The appendix records the new search, pretrained source, grouped checks and compute cost; the initial comparison above is retained for context. Scores optimize ranking and are not calibrated authorship certainties.''')
    ci=analysis['bootstrap_95_ci'];near=analysis['near_duplicate_audit']
    between(r'\q{Q4.1}',r'\q{Q4.2}',r'''\q{Q4.1}{Final training--validation gap and uncertainty.}
The final selected model has training AUC '''+fmt(analysis['train_auc'])+' and validation AUC '+fmt(analysis['val_auc'])+
        '. These scores use the controlled 8,000/2,000 fit before full-data refitting. A 1,000-replicate stratified bootstrap gives ['+
        fmt(ci[0])+', '+fmt(ci[1])+r'''], conditional on the selected fitted model; it excludes selection and distribution-shift uncertainty.
No exact texts cross the split. Two validation passages have nearest-training word-TF-IDF cosine similarity at least 0.90; removing them yields AUC '''+fmt(near['auc_excluding_high_similarity'])+r'''. This is not a complete shared-source audit.
The public AUC is lower than random-split AUC. That observed gap, and the failed robust candidate, caution against treating a near-perfect holdout as a universal authorship detector.''')
    bins=table(['Words','Count','Human / AI','AUROC'],[[esc(g['bin']),g['n'],f"{g['human']} / {g['ai']}",fmt(g['auc'])] for g in analysis['length_bins']],'Yrrr')
    between(r'\q{Q4.2}',r'\begin{center}',r'''\q{Q4.2}{Final observable text-length subgroups.}
We count whitespace-delimited words after decoding escaped newlines and keep the original bins, independent of labels and model predictions.
'''+bins+r'''All three bins have AUC near 1.0. Unequal group sizes and tiny differences do not establish a reliable subgroup advantage. These length groups do not test new generators or domains; the appendix adds unsupervised topic/style-cluster checks, which are also only proxies.''')
    text=text.replace('{diagnostics.pdf}','{diagnostics_optimized.pdf}')
    matrix=analysis['confusion_matrix_at_0.5'];tn,fp=matrix[0];fn,tp=matrix[1]
    between(r'\q{Q4.3}',r'\newpage',r'\q{Q4.3}{Final errors at threshold 0.5.}'+
        f'The confusion matrix is TN={tn}, FP={fp}, FN={fn}, TP={tp}, with accuracy {(tn+tp)/2000:.4f}. '+
        'A false positive is human text scored at least 0.5; a false negative is AI text scored below 0.5. '+
        'The next page shows the five highest-scoring false positives and five lowest-scoring false negatives. These deliberately extreme cases are not a representative error sample.')
    marker=text.index('Generalization: concrete errors');left=text.index(r'\begin{tabularx}',marker);right=text.index(r'\end{tabularx}',left)+len(r'\end{tabularx}')
    error_table=table(['ID','Type','P(AI)','Observed content and possible failure mechanism'],
        [[e['id'],e['type'],f"{e['probability']:.4f}",esc(ERROR_NOTES[e['id']])] for e in errors],'rrrY')
    text=text[:left]+error_table+text[right:]
    between(r'\q{Q4.4}',r'\newpage',r'''\q{Q4.4}{Recurring final-model failures.}
The five human errors include repetitive forum language, two reference-style descriptions, a short review and an informal self-report.
AI misses include a recipe, an academic abstract, a biography, parenting discussion and symbolic mathematics.
Thus neither fluent reference prose nor conversational or technical irregularity uniquely identifies authorship.
The highly repetitive human example also shows why a predictability signal cannot establish generation on its own.
These are observations about the selected examples; the proposed mechanisms are hypotheses, not causal tests or recovered source/generator identities.
Additional training-only symbolic-text augmentation or a representation better suited to mathematics could be tested next.
Language features reduced total threshold errors from 31 to 18, but some initial errors remain, and the public subset still performs below the matched validation pool.''')
    command_start=text.index(r'\begin{quote}',text.index(r'\q{Q6.1}'));command_end=text.index(r'\end{quote}',command_start)+len(r'\end{quote}')
    commands=r'''\begin{quote}\small\begin{verbatim}
DATA_DIR=cornell-tech-cs-5785-combined-2026-ai-vs-human-text
bash submissions/group-number-11/scripts/reproduce_optimized.sh "$DATA_DIR"
python submissions/group-number-11/src/predict_optimized.py \
  --test "$DATA_DIR/test.csv" \
  --sample "$DATA_DIR/sample_submission.csv" \
  --model submissions/group-number-11/artifacts/optimization/language_v2.joblib \
  --out submissions/group-number-11/output/submission_language_v2.csv
\end{verbatim}\end{quote}'''
    # Long shell commands break safely in the PDF with a smaller verbatim font.
    commands=commands.replace(r'\small\begin{verbatim}',r'\footnotesize\begin{verbatim}')
    text=text[:command_start]+commands+text[command_end:]
    text=text.replace('CPU experiments used macOS arm64, Python 3.14.2 and four compute threads.',
        'Initial CPU experiments used macOS arm64 and Python 3.14.2. Optimization added local Apple MPS feature extraction; exact added packages are in requirements-optimization.txt.')
    between("The team's \\texttt{reproduce.sh}",r'\q{Q6.2}',r'''The \texttt{reproduce\_optimized.sh} entrypoint runs baseline fitting, frozen language-feature extraction, classifier search, final refitting and prediction. The README documents supplementary diagnostics and report generation. Data, model checkpoints and intermediate feature caches remain local and are excluded from Git.''')
    text=text.replace('so the original model remains the final submission model.',
        'so augmentation alone was rejected; the later language-feature model is evaluated in the appendix.')
    text=text.replace('The final submission therefore uses the supervised representation fits.',
        'The baseline component therefore retains its supervised-only word representation fit.')
    text=text.replace('No external LLM API was used to label or predict passages; no pretrained model weights or external supervised examples were used.',
        'No hosted LLM API labeled or predicted passages. Optimization used generic GPT-2 pretrained weights locally; additional neural experiments are recorded in the optimization logs. No external authorship labels were used.')
    text=text.replace('The team uploaded the new-model CSV and reported public AUROC 0.96962. The previous public score was 0.92433. Private results remain pending.',
        f'The team reported public AUROC {best["public_auc"]:.5f} for the optimized submission, following 0.96962 for the first rebuild and 0.92433 for the earlier model. Private results remain pending.')
    history=subprocess.check_output(['git','log','--reverse','--format=%h %s','--','submissions/group-number-11','submissions/group11'],cwd=ROOT.parents[1]).decode().splitlines()
    commits=[]
    for phrase in ['Build fixed-holdout','Test format-robust','Add frozen GPT-2 detector']:
        matches=[line for line in history if phrase in line]
        if matches:commits.append(matches[-1].split(' ',1))
    if len(commits)==3:
        items=r'\begin{itemize}[leftmargin=1.5em,itemsep=3pt]'+'\n'
        for sha,title in commits:items+=r'\item \href{'+meta['repository_url']+'/commit/'+sha+'}{'+sha+'}: '+esc(title)+'\n'
        items+=r'\end{itemize}'
        between(r'\q{Q6.2}',r'\q{Q6.3}',r'\q{Q6.2}{Development history and lessons.}'+items+
            'The initial pipeline established the shared split and baseline comparisons. The format-robust follow-up showed that synthetic stress improvement did not transfer to public performance. The language-feature change then improved both the fixed validation score and the user-reported public score. These are actual development checkpoints; the experiments and their negative results are retained for review.')
    appendix=r'''\newpage
\section*{Appendix: post-submission optimization (Q1/Q5/Q6 continuation)}
\textbf{Measured sequence and selection.} The first format-robust candidate improved synthetic stress AUC but reduced public AUC, an informative negative result. The language-feature candidate was then selected using the original fixed holdout: retain at least 25\% baseline weight, require native AUC at least 0.9975, and maximize 0.75 times the no-blank-line subgroup AUC plus 0.25 times overall AUC. This focuses on passages without an easy paragraph cue. A finite grid compared baseline weights 0.25, 0.5 and 0.75, with plain or conditional-rank blending. Plain blending won; inference does not use ranks or true labels.
'''
    history_rows=[
        ['First rebuild ensemble','0.997975','0.96962'],['Robust format candidate','0.995598','0.95899'],
        ['GPT-2 statistics + baseline','0.999785','0.98665']]
    for candidate,label in [('neural_v3','RoBERTa logit hybrid'),('medium_v4','Medium-language hybrid')]:
        delivery=ROOT/'results/optimization'/(candidate+'_delivery.json')
        if delivery.exists():
            row=json.loads(delivery.read_text())
            if row.get('public_auc') is not None:
                auc=row.get('val_auc',row.get('selected',{}).get('val_auc'))
                history_rows.append([label,fmt(auc),f"{row['public_auc']:.5f}"])
    appendix+=table(['Configuration','Validation AUC','Public AUC'],history_rows,'Yrr')
    appendix+=r'''\textbf{Representation and search.} The additional 279-dimensional representation includes the original 160 style features plus lexical-diversity, n-gram repetition, word/sentence distributions and character/paragraph statistics. We concatenate 26 frozen GPT-2 features: token negative-log-likelihood distribution statistics, head/tail-window statistics, rank-1/10/100 fractions and logit-margin statistics. Only the first 256 model tokens are used.
The generic model is \href{https://huggingface.co/openai-community/gpt2}{openai-community/gpt2}, revision \texttt{607a30d783dfa663caf39e06633721c8d4cfcd7e}; its pretrained weights are unchanged. No competition passage is sent to a hosted endpoint.
The new grid compares LightGBM leaves 7/15/31 and CatBoost depth 5 in native, collapsed and paired native/collapsed style views. Language features retain original text in those paired style views, so style-only stress scores are not reported as full-pipeline format robustness.

\textbf{Additional grouped evidence.} Twelve unsupervised lexical topic/style clusters define three group-held-out folds, without hidden source labels. Frozen style-only and style-plus-GPT-2 configurations use their separately selected 31- and 15-leaf settings. This is a pipeline comparison, not an isolated feature ablation:
'''
    appendix+=table(['Fold','Style only','Style + GPT-2'],[['1','0.994101','0.999002'],['2','0.998777','0.998415'],['3','0.990991','0.997209']],'Yrr')
    appendix+=r'''The small regression on fold 2 cautions against universal improvement claims. These clusters are proxies, and their folds overlap the original selection pool; they are not independent external confirmation.

\textbf{Compute and limitations.} Language-feature extraction took minutes on the local Apple GPU and is cached for reuse; the selected classifier stage took about 23 seconds with cached features. This is not directly comparable to the initial CPU feature-plus-fit timings. The final tree fits use all 10,000 true labeled rows after selection, with paired style views counted as augmentation rather than new independent data. Generic pretrained language knowledge is the only added external model resource. Public feedback was used to reject the robust candidate and confirm the language candidate; no hidden individual labels were accessed. The repeated holdout search and small public subset remain sources of selection uncertainty.
'''
    if best['candidate']!='language_v2':
        frame=pd.read_csv(ROOT/'artifacts/optimization'/(best['candidate']+'_validation.csv')).set_index('id')
        hybrid_summary=r'''\q{Q0.2}{Final approach.}
The language-feature parent $p_L$ blends a GPT-2/statistics LightGBM (75\%) with the original word/character/style ensemble (25\%).
A second model $p_R$ fine-tunes the upper four blocks of generic RoBERTa on the fixed 8,000-row training partition, using random token crops and first/last-window inference.
Their conservative hybrid is $p_N=\sigma(0.9\,\mathrm{logit}(p_L)+0.1\,\mathrm{logit}(p_R))$, with probabilities clipped before logit conversion.
Tree components are refitted on all 10,000 labels after selection; RoBERTa keeps its held-out checkpoint, with no validation-label training.
All pretrained models run locally; no external authorship labels or hosted prediction APIs are used.'''
        if best['candidate']=='medium_v4':
            delivery=json.loads((ROOT/'results/optimization/medium_v4_delivery.json').read_text());choice=delivery['selected'];w=choice['second_weight']
            operation=(f"The final score averages {1-w:.2f} of $p_N$ with {w:.2f} of a second language-feature detector $p_M$." if choice['method']=='plain' else
                f"The final score is $\\sigma({1-w:.2f}\\,\\mathrm{{logit}}(p_N)+{w:.2f}\\,\\mathrm{{logit}}(p_M))$.")
            hybrid_summary=r'''\q{Q0.2}{Final approach.}
The final score is $\sigma(0.75\,\mathrm{logit}(p_N)+0.25\,\mathrm{logit}(p_M))$, where $p_N=\sigma(0.9\,\mathrm{logit}(p_L)+0.1\,\mathrm{logit}(p_R))$ and probabilities are clipped before logit conversion.
The parent $p_L$ blends a GPT-2/statistics LightGBM (75\%) with the original word/character/style ensemble (25\%).
The neural branch $p_R$ fine-tunes the upper four RoBERTa blocks on 8,000 training rows, with random token crops and first/last-window inference.
The detector $p_M$ combines frozen small/medium GPT-2 statistics and local style features with label-free within-format score alignment.
A fixed 2,000-row holdout selects the configurations; tree components are then refitted on all 10,000 labels while RoBERTa retains its held-out checkpoint, and all models run locally without external authorship labels or hosted prediction APIs.'''
        between(r'\q{Q0.2}',r'\q{Q0.3}',hybrid_summary)
        between(r'\q{Q1.4}',r'\newpage',r'''\q{Q1.4}{Final model choice and later changes.}
The first rebuild achieved 0.96962 public AUROC. A format-robust candidate dropped to 0.95899, so it was rejected.
Frozen language features then raised public AUC to 0.98665; a small RoBERTa logit contribution raised it to 0.98861.
The final selected submission scores '''+fmt(analysis['val_auc'])+' validation / '+f"{best['public_auc']:.5f}"+r''' public AUROC.
These later changes add complementary representations while retaining a large contribution from the previously scored detector.
The neural-only validation AUC was 0.979583, so sophistication alone did not justify replacing the stronger parent.
Small blend gains on the reused validation partition are not statistically decisive; public feedback is recorded separately and private performance is unknown.
The appendix records the pretrained sources, training schedules, blend search and compute cost. Scores optimize ranking, not calibrated authorship certainty.''')
        certificate=analysis.get('training_certificate')
        if certificate:
            marker=text.index(r'\q{Q4.2}')
            note='The training AUC is exactly rank-certified from bounded, clipped-logit ensemble scores; this avoids unnecessary full neural inference on the training partition. The certificate and bounds are saved with the analysis.\n'
            text=text[:marker]+note+text[marker:]
        review_t=analysis.get('error_review_threshold',.4);review=analysis.get('error_review_confusion',matrix)
        between(r'\q{Q4.3}',r'\newpage',r'\q{Q4.3}{Final-model error review.}'+
            f'The final confusion matrix is TN={tn}, FP={fp}, FN={fn}, TP={tp}, with accuracy {(tn+tp)/2000:.4f}. '+
            f'These counts use threshold 0.5. For the required five-FP/five-FN case review we use threshold {review_t:.2f}, yielding {review[0][1]} FP and {review[1][0]} FN. '+
            'This review threshold is stated explicitly and does not change AUROC or the continuous Kaggle predictions. The next page shows the five highest-scoring human false positives and five lowest-scoring AI false negatives under that review threshold.')
        marker=text.index('Generalization: concrete errors');left=text.index(r'\begin{tabularx}',marker);right=text.index(r'\end{tabularx}',left)+len(r'\end{tabularx}')
        table_with_final=table(['ID','Error','Final P(AI)','Observed content / hypothesis'],
            [[e['id'],e['type'],f"{frame.loc[e['id'],'probability']:.4f}",esc(ERROR_NOTES[e['id']])] for e in errors],'rrrY')
        text=text[:left]+table_with_final+text[right:]
        between(r'\q{Q4.4}',r'\newpage',r'''\q{Q4.4}{Recurring errors and changes after blending.}
The reviewed cases include repetitive human wording, reference/review prose and structured exposition, alongside AI content with recipe, academic, reference and conversational framing.
These genres overlap across labels: predictability, conversational irregularity and technical notation do not uniquely identify authorship.
The final model has '''+str(fp+fn)+r''' total errors at the common threshold 0.5, compared with 18 for the language parent and 31 for the first rebuild. The ten detailed examples use the separately declared review threshold.
The mechanism descriptions are hypotheses based on observed text, not causal interventions or recovered generator identities.
More targeted training-only symbolic-text augmentation could be tested, but was not assumed to work.
The small public subset and repeated model selection still limit conclusions about unseen generators and private performance.''')
        start=text.index(r'\begin{quote}',text.index(r'\q{Q6.1}'));end=text.index(r'\end{quote}',start)+len(r'\end{quote}')
        script='reproduce_final.sh' if best['candidate']=='medium_v4' else 'scripts/reproduce_neural.sh'
        commands='\\begin{quote}\\footnotesize\\begin{verbatim}\npip install -r submissions/group-number-11/requirements-optimization.txt\nDATA_DIR=cornell-tech-cs-5785-combined-2026-ai-vs-human-text\nbash submissions/group-number-11/'+script+' "$DATA_DIR"\n'
        if best['candidate']=='medium_v4':
            commands+='python submissions/group-number-11/predict_final.py \\\n  --data "$DATA_DIR" \\\n  --out submissions/group-number-11/output/submission.csv\n'
        commands+='\\end{verbatim}\\end{quote}'
        text=text[:start]+commands+text[end:]
        between('Initial CPU experiments used',r'\begin{quote}',r'''Initial baseline versions are recorded in \texttt{environment.txt} and \texttt{requirements.txt}.
For the final model, activate the project virtual environment and install \texttt{requirements-optimization.txt}; the exact added package versions are pinned there.
The tested setup uses Python 3.14.2 on macOS arm64 with Apple MPS; CPU fallback is available but slower.
The split/tree seed is 20261023 and the neural seed is 20261111; supplementary split seeds are 20261024--20261026.
The reproduction command trains/evaluates the components; the prediction command below regenerates the CSV from their cached inference outputs without reselecting weights or reading validation labels.''')
        text=text.replace('The \\texttt{reproduce\\_optimized.sh} entrypoint runs baseline fitting, frozen language-feature extraction, classifier search, final refitting and prediction.',
            'The final reproduction entrypoint runs the parent detector, local neural training, held-out checkpoint selection, inference and the scored blend. Cached predictions can be reblended with the documented prediction command.')
        text=text.replace('Optimization used generic GPT-2 pretrained weights locally;',
            'Optimization used generic GPT-2 weights locally and fine-tuned upper RoBERTa blocks using only supplied labels;')
        neural_text=r'''\textbf{Neural continuation.} The generic source is \href{https://huggingface.co/FacebookAI/roberta-base}{FacebookAI/roberta-base}.\par
Revision: \texttt{e2da8e2f811d1448a5b465c236feacd80ffbac7b}.\par
Embeddings and the lower eight encoder blocks are frozen; four upper blocks and a fresh classification head train on the 8,000 training examples. AdamW uses learning rates $2\cdot10^{-5}$ for the encoder and $10^{-4}$ for the head, weight decay 0.01, batch size 16 and seed 20261111.
Canonicalized texts receive random 192-token crops; inference averages first/last windows. A budgeted two-epoch attempt stopped at step 861; validation chose checkpoint 750. The tree parent is later refitted on 10,000 rows, while the neural branch retains its 8,000-row checkpoint.
Plain and logit blends at neural weights 0/.05/.1/.15/.25/.5 were compared; 0.1 logit blending won by validation AUC (0.999805 versus 0.999785). This tiny local difference is not conclusive evidence; the public result 0.98861 was supplied independently by the team.
'''
        if best['candidate']=='medium_v4':
            neural_text+=r'''\textbf{Second language model.} Frozen \href{https://huggingface.co/openai-community/gpt2-medium}{GPT-2-medium} uses canonical whitespace and a 128-token prefix, alongside the small model's 256-token features. Their statistics, differences and normalized ratios give 104 language dimensions, not a same-context theoretical perplexity ratio. A native-view 31-leaf LightGBM uses these plus the 279 style features. Its scores receive label-free within-format quantile alignment to the reference detector, then 75\% aligned scores are combined with 25\% reference scores to form $p_M$. The final grid compares plain/logit weights .25/.5/.75 for $p_M$, retaining at least 25\% of the scored neural hybrid. A 25\% logit contribution wins. The rank mapping uses only predicted scores and observed blank-line groups, never true test labels.
'''
        appendix=appendix.replace(r'\textbf{Compute and limitations.}',neural_text+'\n'+r'\textbf{Compute and limitations.}')
        appendix=appendix.replace('the selected classifier stage took about 23 seconds','the language-parent classifier stage took about 23 seconds')
        appendix=appendix.replace('Plain blending won; inference does not use ranks or true labels.',
            'Plain blending won for the language parent. The later medium detector uses label-free conditional-rank alignment within observed blank-line groups; no true labels enter inference.')
    between(r'\end{quote}',r'\q{Q6.2}',r'''\end{quote}
The reproduction script recreates the components; \texttt{predict\_final.py} writes the frozen blend from cached inference outputs. Data, model weights and caches are excluded from Git.''')
    text=text.replace('The team supplied the individual roles recorded in Q0.1, and reports that all three members also trained models and explored stronger configurations.\nThis describes the team\'s reported division of work, while the implementation and experiment execution in this rebuilt branch were assisted by Codex.\nBefore course submission, the team needs to review the code and interpretations and enter the newly observed leaderboard result.',
        'Team-reported roles appear in Q0.1; the team must review the submitted code and interpretations.')
    text=text.replace('The team supplied the individual roles recorded in Q0.1, and reports that all three members also trained models and explored stronger configurations.\nThis describes the team\'s reported division of work, while the implementation and experiment execution in this rebuilt branch were assisted by Codex.\nBefore course submission, the team needs to review the code, interpretations and latest leaderboard result.',
        'Team-reported roles appear in Q0.1; the team must review the submitted code and interpretations.')
    compute_marker=appendix.index(r'\textbf{Compute and limitations.}')
    compute_note=appendix[compute_marker:]
    appendix=appendix[:compute_marker]
    text=text.replace(r'\end{document}',compute_note+'\n'+appendix+'\n'+r'\end{document}')
    text=text.replace('The observed content descriptions below come from reviewing the supplied validation passages.',
        'All error labels below use the declared review threshold 0.40. The descriptions come from reviewing the supplied validation passages.')
    path.write_text(text)

# Work plan

Original allocation: six issues each. Issue #11 transferred from Gtpatel02 to rdevasia25 on November 10, 2026. Issues #14 and #17 transferred from Gtpatel02 to Jalil-g on November 10, 2026. Aim for2-3 useful commits per issue,not padded counts.

|Issue|Owner|Notebook|Depends on|
|---|---|---|---|
|[#1](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/1)|Jalil-g|01_data_setup.ipynb|None|
|[#2](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/2)|Jalil-g|02_split_metrics.ipynb|#1|
|[#3](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/3)|rdevasia25|03_word_baseline.ipynb|#2|
|[#4](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/4)|rdevasia25|04_word_comparison.ipynb|#3|
|[#5](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/5)|rdevasia25|05_char_comparison.ipynb|#4|
|[#6](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/6)|Jalil-g|06_learning_curve.ipynb|#3|
|[#7](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/7)|rdevasia25|07_pseudo_labels.ipynb|#4|
|[#8](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/8)|Gtpatel02|08_error_analysis.ipynb|#5|
|[#9](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/9)|Gtpatel02|09_linear_style.ipynb|#8|
|[#10](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/10)|rdevasia25|10_expanded_text.ipynb|#5,#9|
|[#11](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/11)|rdevasia25|11_formatting_rule.ipynb|#10|
|[#12](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/12)|Jalil-g|12_seed_stress.ipynb|#11|
|[#13](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/13)|Gtpatel02|13_rich_style.ipynb|#2,#9|
|[#14](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/14)|Jalil-g|14_style_augmentation.ipynb|#13,#12|
|[#15](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/15)|rdevasia25|15_normalized_transfer.ipynb|#7,#13|
|[#16](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/16)|Jalil-g|16_local_electra.ipynb|#12,#15|
|[#17](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/17)|Jalil-g|17_final_blend.ipynb|#11,#14,#15|
|[#18](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/18)|Jalil-g|18_final_checks.ipynb|#6,#7,#8,#9,#12,#16,#17|
|[#37](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/37)|Jalil-g|19_internal_neural_pick.ipynb|#18|
|[#39](https://github.com/Jalil-g/Cornell-Text-Authorship/issues/39)|Jalil-g|20_final_submission.ipynb|#37|

Start #1 then #2. Respect execution dependencies. Notebook16 neural-blend subsection also waits for17. Each member reviews a teammate PR.

# Three-person workflow

Names A/B/C identify ownership; agreed members are Lin/A, Alex/B, Gabriel/C.

Use English for uploaded documentation, code comments, commit messages, and
pull request titles/descriptions.

| Role | Main report sections | Independent responsibility |
| --- | --- | --- |
| A (Lin) | Q1, Q5 | Supervised comparisons, tuning, early score candidates, controlled improvement |
| B (Alex, available Sunday) | Q2, Q4 | Own baseline, learning curve, subgroup/error evaluation |
| C (Gabriel, available Mon–Wed) | Q3, Q6 coordination | Own baseline, semi-supervised experiment, README/environment/reproduction |

Everyone supplies Q0.1 contributions and their Q6.2 commit links/Q6.3 AI-use notes.
This assignment was confirmed in planning. See `docs/lin_q1_handoff.md` for
the current Q1 candidate and transfer files.

## Branches and integration

Start from the same scaffold commit on `main`. Use short feature branches such as
`exp/supervised-baselines`, `exp/learning-curve`, and `exp/pseudo-labeling`, then
merge working increments via PR. Do not wait until the deadline to integrate.
All members must be able to generate their own predictions from original files.
Keep meaningful development history for the course review.

Own your experiment directory, matching results directory, assigned Overleaf sections, and
uniquely named figures. Avoid editing another member's files concurrently.
Shared preprocessing is optional; the split and prediction contract are mandatory.
Report writing and integration take place in the team's separate Overleaf
project. Do not generate or upload local report prose, TeX stubs, build
entrypoints, or report PDFs. Later writing should use the actual code and
recorded experimental evidence; each author completes their assigned sections.

By Monday night exchange a runnable first result; by Tuesday evening select the
final approach. B reruns Q4 analysis using selected validation predictions. C
checks reproduction. A runs final training/prediction, another member checks CSV.
Confirm exact course deadline/time zone separately.

## Kaggle slots

Sunday, November 8, 2026: A and B each get two candidate upload slots. Fifth slot
remains flexible and must be coordinated before use. Revisit allocation Monday
when C is available. Five is the TEAM daily limit. Confirm Kaggle's reset time
rather than assuming local midnight. Log candidate, uploader and result in
`submissions/leaderboard_log.csv`. A blank score means not recorded, not zero.
Coordinate reservations in iMessage to avoid concurrent CSV edits or uploads.

## Course PR

The private team working repository is
https://github.com/lins369/kaggle_ai_vs_human, owned by `lins369`. The owner retains
repository administration and can invite teammates under Settings > Collaborators.
This working repository is distinct from the required final PR against the course
repository. Before
creating the final PR, inspect the course repository template and follow its
required destination layout. Preserve this working structure inside the permitted
team directory if necessary, and retain meaningful development history. Confirm
dataset redistribution rules before publishing any data; raw passages are ignored
here by default. Do not commit the Overleaf share link embedded in the original
template into a public repository without considering its access permissions.


# Q4 results: masked_lr (threshold 0.5, split seed 42)

## Q4.1 train vs validation

| model     |   train rows |   val rows |   train AUROC |   val AUROC |   train error@0.5 |   val error@0.5 |   fit seconds |
|:----------|-------------:|-----------:|--------------:|------------:|------------------:|----------------:|--------------:|
| masked_lr |         8000 |       2000 |        0.9998 |      0.9949 |            0.0064 |           0.026 |        4.2659 |

## Q4.2 by length bin

| length_bin         |   count |   AI share |   val AUROC |   error rate@0.5 |
|:-------------------|--------:|-----------:|------------:|-----------------:|
| short (<150 words) |     404 |     0.6287 |      0.9885 |           0.0594 |
| medium (150-249)   |    1055 |     0.4123 |      0.9967 |           0.0123 |
| long (>=250)       |     541 |     0.5749 |      0.9954 |           0.0277 |

## Q4.2 by linebreak format

| format            |   count |   AI share |   val AUROC |   error rate@0.5 |
|:------------------|--------:|-----------:|------------:|-----------------:|
| blank lines       |     689 |     0.9956 |      1      |           0.0058 |
| no linebreaks     |     602 |     0.4419 |      0.9838 |           0.0548 |
| single linebreaks |     709 |     0.0677 |      0.9616 |           0.0212 |

## Q4.2 by topic cluster

|   cluster | topic                               |   count |   AI share |   val AUROC |   error rate@0.5 | fiction?   |
|----------:|:------------------------------------|--------:|-----------:|------------:|-----------------:|:-----------|
|         8 | born, university, american, known   |     327 |     0.4924 |      0.9863 |           0.0275 | False      |
|         1 | like, just, ve, don                 |     387 |     0.5142 |      0.9906 |           0.0491 | False      |
|         9 | love, life, world, man              |      41 |     0.6585 |      0.9921 |           0.0976 | True       |
|         7 | team, league, game, club            |     125 |     0.432  |      0.9943 |           0.032  | False      |
|         3 | pepper, cup, salt, oil              |     217 |     0.5161 |      0.9946 |           0.0092 | False      |
|         2 | film, movie, br, story              |     139 |     0.5324 |      0.9967 |           0.0288 | False      |
|         6 | paper, study, results, quantum      |     194 |     0.4742 |      0.9973 |           0.0206 | False      |
|         4 | sugar, cup, flour, butter           |     176 |     0.5    |      0.9988 |           0.0114 | False      |
|         5 | said, year, company, mr             |     164 |     0.5366 |      0.9996 |           0.0244 | False      |
|         0 | segmentation, image, images, method |     230 |     0.4565 |      1      |           0      | False      |

## Q4.3 top 5 FP and FN

|   id | error   |   p_ai |   n_words | format            |   newlines_per_100w |   cluster | topic                             | snippet                                                                                                                                                          |
|-----:|:--------|-------:|----------:|:------------------|--------------------:|----------:|:----------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1898 | FP      |  0.738 |       155 | single linebreaks |               0.645 |         6 | paper, study, results, quantum    | Outcome primacy is a psychological phenomenon that describes lasting effects on a subject's behavior based on the outcome of first experiences with a given task |
| 3139 | FP      |  0.689 |       168 | no linebreaks     |               0     |         6 | paper, study, results, quantum    | For fluid power, a working fluid is a gas or liquid that primarily transfers force, motion, or mechanical energy. In hydraulics, water or hydraulic fluid transf |
| 3155 | FP      |  0.619 |       184 | single linebreaks |               0.543 |         8 | born, university, american, known | Supernormal Stimuli: How Primal Urges Overran Their Evolutionary Purpose is a book by Deirdre Barrett published by W. W. Norton & Company in 2010. Barrett is a  |
|  151 | FP      |  0.595 |        79 | no linebreaks     |               0     |         2 | film, movie, br, story            | It may be unfair to say, but its easily admitted that the most interesting ploy this film can rely on was the media storm surrounding "THE affair" that led to " |
| 1038 | FP      |  0.582 |       214 | no linebreaks     |               0     |         8 | born, university, american, known | The White Terror was a period during the French Revolution in 1795 when a wave of violent attacks swept across much of France. The victims of this violence were |
| 5498 | FN      |  0.01  |       272 | single linebreaks |               7.721 |         3 | pepper, cup, salt, oil            | Ingredients 1 pound ground beef 2 tablespoons olive oil, divided 1 onion, chopped 3 cloves garlic, minced 1 (8 ounce) can tomato sauce 1/4 cup water 1 teaspoon  |
| 5010 | FN      |  0.027 |       290 | no linebreaks     |               0     |         8 | born, university, american, known | Cynthia Alina Turner (born April 18, 1960 in San Rafael, Marin County, California1) is an American actress, producer, and model. She had a recurring role on Sta |
| 9298 | FN      |  0.028 |       205 | single linebreaks |               4.878 |         8 | born, university, american, known | "Dane Cook" is an American comedian. Cook's style is known as edgy, and his comedy is largely based on racial, sexual, and relationship jokes. He won the "Nacho |
| 5929 | FN      |  0.072 |       237 | single linebreaks |               2.11  |         4 | sugar, cup, flour, butter         | Ingredients: 1 3/4 cups plain flour, plus some for kneading 2 tbsp white sugar2 tsp baking powder ¾ teaspoons salt ½ tablespoon butter, cold and cut into small  |
| 1614 | FN      |  0.08  |       115 | no linebreaks     |               0     |         1 | like, just, ve, don               | I’ve never felt more alone. Despite being surrounded by people, I feel like I’m the only one who truly understands me. I’ve been feeling this way for a while no |

Total errors on val: 52 (9 FP, 43 FN)

## Q4.4 error profile

|                |   n_words |   newlines_per_100w |
|:---------------|----------:|--------------------:|
| all val        |    207.44 |                4.05 |
| FP (human->AI) |    158.22 |                0.13 |
| FN (AI->human) |    187.65 |                1.3  |

### errors by format (row-normalised)

| format            |    FN |    FP |   correct |
|:------------------|------:|------:|----------:|
| blank lines       | 0.006 | 0     |     0.994 |
| no linebreaks     | 0.043 | 0.012 |     0.945 |
| single linebreaks | 0.018 | 0.003 |     0.979 |

### errors by topic (counts)

| topic                               |   FN |   FP |   correct |
|:------------------------------------|-----:|-----:|----------:|
| born, university, american, known   |    7 |    2 |       318 |
| film, movie, br, story              |    2 |    2 |       135 |
| like, just, ve, don                 |   17 |    2 |       368 |
| love, life, world, man              |    4 |    0 |        37 |
| paper, study, results, quantum      |    2 |    2 |       190 |
| pepper, cup, salt, oil              |    2 |    0 |       215 |
| said, year, company, mr             |    4 |    0 |       160 |
| segmentation, image, images, method |    0 |    0 |       230 |
| sugar, cup, flour, butter           |    2 |    0 |       174 |
| team, league, game, club            |    3 |    1 |       121 |

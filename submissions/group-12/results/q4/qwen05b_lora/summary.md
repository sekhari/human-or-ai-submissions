# Q4 results: qwen05b_lora (threshold 0.5, split seed 42)

## Q4.1 train vs validation

| model        |   train rows |   val rows |   train AUROC |   val AUROC |   train error@0.5 |   val error@0.5 |   fit seconds |
|:-------------|-------------:|-----------:|--------------:|------------:|------------------:|----------------:|--------------:|
| qwen05b_lora |         8000 |       2000 |        0.9998 |      0.9986 |            0.0036 |           0.008 |           nan |

## Q4.2 by length bin

| length_bin         |   count |   AI share |   val AUROC |   error rate@0.5 |
|:-------------------|--------:|-----------:|------------:|-----------------:|
| short (<150 words) |     404 |     0.6287 |      0.9986 |           0.0124 |
| medium (150-249)   |    1055 |     0.4123 |      0.9991 |           0.0066 |
| long (>=250)       |     541 |     0.5749 |      0.9978 |           0.0074 |

## Q4.2 by linebreak format

| format            |   count |   AI share |   val AUROC |   error rate@0.5 |
|:------------------|--------:|-----------:|------------:|-----------------:|
| blank lines       |     689 |     0.9956 |      0.9976 |           0.0015 |
| no linebreaks     |     602 |     0.4419 |      0.9974 |           0.0133 |
| single linebreaks |     709 |     0.0677 |      0.9849 |           0.0099 |

## Q4.2 by topic cluster

|   cluster | topic                               |   count |   AI share |   val AUROC |   error rate@0.5 | fiction?   |
|----------:|:------------------------------------|--------:|-----------:|------------:|-----------------:|:-----------|
|         8 | born, university, american, known   |     327 |     0.4924 |      0.9947 |           0.0092 | False      |
|         3 | pepper, cup, salt, oil              |     217 |     0.5161 |      0.9978 |           0.0138 | False      |
|         1 | like, just, ve, don                 |     387 |     0.5142 |      0.9988 |           0.0103 | False      |
|         4 | sugar, cup, flour, butter           |     176 |     0.5    |      0.9997 |           0.0114 | False      |
|         9 | love, life, world, man              |      41 |     0.6585 |      1      |           0      | True       |
|         6 | paper, study, results, quantum      |     194 |     0.4742 |      1      |           0.0052 | False      |
|         5 | said, year, company, mr             |     164 |     0.5366 |      1      |           0.0061 | False      |
|         2 | film, movie, br, story              |     139 |     0.5324 |      1      |           0.0072 | False      |
|         0 | segmentation, image, images, method |     230 |     0.4565 |      1      |           0      | False      |
|         7 | team, league, game, club            |     125 |     0.432  |      1      |           0.008  | False      |

## Q4.3 top 5 FP and FN

|   id | error   |   p_ai |   n_words | format            |   newlines_per_100w |   cluster | topic                             | snippet                                                                                                                                                          |
|-----:|:--------|-------:|----------:|:------------------|--------------------:|----------:|:----------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 7376 | FP      |  1     |       229 | blank lines       |               3.93  |         3 | pepper, cup, salt, oil            | 2 medium onions 4 cloves of garlic 1/2-1 fresh red chilli 1 bunch fresh thyme 14 ounces canned tomatoes (chop them roughly if they're whole) extra virgin olive  |
| 6728 | FP      |  1     |       252 | single linebreaks |               3.175 |         4 | sugar, cup, flour, butter         | ngredients: 2 large eggs 1 tsp water 3 TBSP buttermilk 1/3 cup Pamela's Bread Mix 1 tsp baking powder 1 slice cheese of choice-cut to bread size 1 slice Canadia |
| 5715 | FP      |  1     |       344 | single linebreaks |               4.36  |         3 | pepper, cup, salt, oil            | 1 head Garlic 1 teaspoon Olive Oil 30 ounces, weight Canned Garbanzo Beans, Drained (Reserve Liquid), Or 2 Cups Dry Garbanzo Beans, Soaked Or Quickly Cooked (Al |
| 9651 | FP      |  0.929 |       149 | no linebreaks     |               0     |         1 | like, just, ve, don               | I just want to be able to stay up late and talk about the most random topics till we fall asleep together. I want to be able to be able to meet up with someone  |
| 3326 | FP      |  0.918 |       114 | no linebreaks     |               0     |         2 | film, movie, br, story            | The Academy Award winning war epic Braveheart is an extraordinary film that set a new bar for the genre. The story follows a Scottish peasant named William Wall |
| 5498 | FN      |  0     |       272 | single linebreaks |               7.721 |         3 | pepper, cup, salt, oil            | Ingredients 1 pound ground beef 2 tablespoons olive oil, divided 1 onion, chopped 3 cloves garlic, minced 1 (8 ounce) can tomato sauce 1/4 cup water 1 teaspoon  |
| 7995 | FN      |  0     |       156 | single linebreaks |               0.641 |         8 | born, university, american, known | Education and career Breit was born 10 April, 1934 in Brooklyn, New York. He served in United States Air Force from 1952 to 1968 and was stationed on several ba |
| 5010 | FN      |  0     |       290 | no linebreaks     |               0     |         8 | born, university, american, known | Cynthia Alina Turner (born April 18, 1960 in San Rafael, Marin County, California1) is an American actress, producer, and model. She had a recurring role on Sta |
| 8486 | FN      |  0     |        84 | no linebreaks     |               0     |         1 | like, just, ve, don               | I am renewing my first class medical and the AME asked me a few questions that I was not expecting. He asked about my history of anxiety and depression and if I |
| 5929 | FN      |  0.005 |       237 | single linebreaks |               2.11  |         4 | sugar, cup, flour, butter         | Ingredients: 1 3/4 cups plain flour, plus some for kneading 2 tbsp white sugar2 tsp baking powder ¾ teaspoons salt ½ tablespoon butter, cold and cut into small  |

Total errors on val: 16 (7 FP, 9 FN)

## Q4.4 error profile

|                |   n_words |   newlines_per_100w |
|:---------------|----------:|--------------------:|
| all val        |    207.44 |                4.05 |
| FP (human->AI) |    206.86 |                3.29 |
| FN (AI->human) |    180    |                1.16 |

### errors by format (row-normalised)

| format            |    FN |    FP |   correct |
|:------------------|------:|------:|----------:|
| blank lines       | 0     | 0.001 |     0.999 |
| no linebreaks     | 0.01  | 0.003 |     0.987 |
| single linebreaks | 0.004 | 0.006 |     0.99  |

### errors by topic (counts)

| topic                               |   FN |   FP |   correct |
|:------------------------------------|-----:|-----:|----------:|
| born, university, american, known   |    2 |    1 |       324 |
| film, movie, br, story              |    0 |    1 |       138 |
| like, just, ve, don                 |    3 |    1 |       383 |
| love, life, world, man              |    0 |    0 |        41 |
| paper, study, results, quantum      |    0 |    1 |       193 |
| pepper, cup, salt, oil              |    1 |    2 |       214 |
| said, year, company, mr             |    1 |    0 |       163 |
| segmentation, image, images, method |    0 |    0 |       230 |
| sugar, cup, flour, butter           |    1 |    1 |       174 |
| team, league, game, club            |    1 |    0 |       124 |

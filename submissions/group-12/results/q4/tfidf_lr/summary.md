# Q4 results: tfidf_lr (threshold 0.5, split seed 42)

## Q4.1 train vs validation

| model    |   train rows |   val rows |   train AUROC |   val AUROC |   train error@0.5 |   val error@0.5 |   fit seconds |
|:---------|-------------:|-----------:|--------------:|------------:|------------------:|----------------:|--------------:|
| tfidf_lr |         8000 |       2000 |             1 |      0.9511 |            0.0012 |          0.1125 |        15.641 |

## Q4.2 by length bin

| length_bin         |   count |   AI share |   val AUROC |   error rate@0.5 |
|:-------------------|--------:|-----------:|------------:|-----------------:|
| short (<150 words) |     404 |     0.6287 |      0.9234 |           0.1658 |
| medium (150-249)   |    1055 |     0.4123 |      0.9545 |           0.1014 |
| long (>=250)       |     541 |     0.5749 |      0.971  |           0.0943 |

## Q4.2 by linebreak format

| format            |   count |   AI share |   val AUROC |   error rate@0.5 |
|:------------------|--------:|-----------:|------------:|-----------------:|
| blank lines       |     689 |     0.9956 |      0.9665 |           0.1597 |
| no linebreaks     |     602 |     0.4419 |      0.9523 |           0.1196 |
| single linebreaks |     709 |     0.0677 |      0.9667 |           0.0606 |

## Q4.2 by topic cluster

|   cluster | topic                               |   count |   AI share |   val AUROC |   error rate@0.5 | fiction?   |
|----------:|:------------------------------------|--------:|-----------:|------------:|-----------------:|:-----------|
|         7 | team, league, game, club            |     125 |     0.432  |      0.8615 |           0.168  | False      |
|         9 | love, life, world, man              |      41 |     0.6585 |      0.8783 |           0.1951 | True       |
|         8 | born, university, american, known   |     327 |     0.4924 |      0.9158 |           0.1468 | False      |
|         5 | said, year, company, mr             |     164 |     0.5366 |      0.9181 |           0.1585 | False      |
|         6 | paper, study, results, quantum      |     194 |     0.4742 |      0.926  |           0.1546 | False      |
|         1 | like, just, ve, don                 |     387 |     0.5142 |      0.9503 |           0.124  | False      |
|         0 | segmentation, image, images, method |     230 |     0.4565 |      0.9687 |           0.0783 | False      |
|         2 | film, movie, br, story              |     139 |     0.5324 |      0.9788 |           0.0791 | False      |
|         3 | pepper, cup, salt, oil              |     217 |     0.5161 |      0.9904 |           0.0369 | False      |
|         4 | sugar, cup, flour, butter           |     176 |     0.5    |      0.9923 |           0.0398 | False      |

## Q4.3 top 5 FP and FN

|   id | error   |   p_ai |   n_words | format            |   newlines_per_100w |   cluster | topic                             | snippet                                                                                                                                                          |
|-----:|:--------|-------:|----------:|:------------------|--------------------:|----------:|:----------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 7562 | FP      |  0.924 |       280 | single linebreaks |               5.714 |         4 | sugar, cup, flour, butter         | 1 c. sliced ripe banana 1/3 c. buttermilk 1 1/2 Tbsp. oil 1 1/2 Tbsp. honey 1 1/3 c. Robin Hood Best For Bread homestyle white flour 2/3 c. Robin Hood Best For  |
| 5234 | FP      |  0.893 |       133 | no linebreaks     |               0     |         1 | like, just, ve, don               | Title's the best I could think of to describe this issue but this is... A strange one. Basically, my Moonlander had been working fine for some time, with LEDs s |
| 3743 | FP      |  0.891 |       153 | no linebreaks     |               0     |         1 | like, just, ve, don               | my WIP I'm currently working on is a zombie fiction book that takes quite a bit from 28 Days Later in that it focuses on a few select characters and their strug |
| 3671 | FP      |  0.864 |       165 | no linebreaks     |               0     |         8 | born, university, american, known | Barbara Carrellas is an author, sex educator, performance artist, and certified sexologist accredited through the American College of Sexologists. She facilitat |
| 2002 | FP      |  0.809 |       145 | no linebreaks     |               0     |         1 | like, just, ve, don               | Leopold the firsts was first offered the Greek Throne. Why did the greek people offer it to him rather than one of their nobles? Also why did he refuse a free m |
| 3062 | FN      |  0.032 |       170 | blank lines       |               3.529 |         1 | like, just, ve, don               | Hello guys, I’m in a pretty good mood today but this post isn’t what you think. I got injected with a small dose of testosterone 2 days ago. It’s my first shot  |
| 9490 | FN      |  0.034 |       288 | blank lines       |               9.028 |         3 | pepper, cup, salt, oil            | Meatloaf: Incredible and Edible! (serves 8 to 10) 1 pkg panko bread crumbs 1 tablespoon finely minced onion 1 tablespoon finely minced garlic 1 tablespoon sugar |
| 1524 | FN      |  0.049 |       124 | blank lines       |               9.677 |         6 | paper, study, results, quantum    | One of the main difficulties of optical quantum measurements is the presence of thermal noise. A squeezed state of light is known to have an enhanced sensitivit |
|  473 | FN      |  0.064 |       252 | blank lines       |               1.587 |         1 | like, just, ve, don               | "I'm 27 year old non-Eu doctor with an EU medical diploma (I got my diploma in the Netherlands). There's an opening for general practitioner in Portugal. I unde |
| 1792 | FN      |  0.071 |       186 | blank lines       |               2.151 |         8 | born, university, american, known | The Beaune Altarpiece, also called The Last Judgement, is a large polyptych painting created by the Flemish artist Rogier van der Weyden. It dates from the mid- |

Total errors on val: 225 (67 FP, 158 FN)

## Q4.4 error profile

|                |   n_words |   newlines_per_100w |
|:---------------|----------:|--------------------:|
| all val        |    207.44 |                4.05 |
| FP (human->AI) |    181.61 |                2.13 |
| FN (AI->human) |    197.92 |                4.28 |

### errors by format (row-normalised)

| format            |    FN |    FP |   correct |
|:------------------|------:|------:|----------:|
| blank lines       | 0.16  | 0     |     0.84  |
| no linebreaks     | 0.061 | 0.058 |     0.88  |
| single linebreaks | 0.016 | 0.045 |     0.939 |

### errors by topic (counts)

| topic                               |   FN |   FP |   correct |
|:------------------------------------|-----:|-----:|----------:|
| born, university, american, known   |   36 |   12 |       279 |
| film, movie, br, story              |    8 |    3 |       128 |
| like, just, ve, don                 |   23 |   25 |       339 |
| love, life, world, man              |    7 |    1 |        33 |
| paper, study, results, quantum      |   18 |   12 |       164 |
| pepper, cup, salt, oil              |    7 |    1 |       209 |
| said, year, company, mr             |   23 |    3 |       138 |
| segmentation, image, images, method |   14 |    4 |       212 |
| sugar, cup, flour, butter           |    5 |    2 |       169 |
| team, league, game, club            |   17 |    4 |       104 |

"""Rule-based genre detection for error analysis: recipe, abstract, poetry, or other.

Keyword and line-format rules, checked on samples of each bucket. Recipes are reliable; "poetry"
(4+ short, non-bulleted lines) is reliable on test.csv but rare and noisy in train.csv, where most
hits are short-line prose such as biographies.
"""

from __future__ import annotations

import re

import numpy as np

RECIPE_TERMS = re.compile(
    r"\b(ingredients?|tablespoons?|teaspoons?|tbsp|tsp|preheat|cups? (of )?\w+|oven|simmer|bake|"
    r"\d+ (ounces?|oz|pounds?|lbs?|grams?|g|cloves?|pinch))\b", re.I)
ABSTRACT_TERMS = re.compile(
    r"\b(we propose|in this paper|this paper|we present|our (method|approach|model)|state-of-the-art|"
    r"experiments? (show|demonstrate))\b", re.I)
BULLET = re.compile(r"\s*([-*•]|\d+[.)])\s")
GENRES = ("recipe", "abstract", "poetry", "other")


def is_recipe(text: str) -> bool:
    return len(RECIPE_TERMS.findall(text)) >= 2


def genre(text: str) -> str:
    if is_recipe(text):
        return "recipe"
    if ABSTRACT_TERMS.search(text):
        return "abstract"
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) >= 4:
        bulleted = np.mean([bool(BULLET.match(line)) for line in lines])
        if bulleted < 0.2 and np.median([len(line) for line in lines]) < 60:
            return "poetry"
    return "other"

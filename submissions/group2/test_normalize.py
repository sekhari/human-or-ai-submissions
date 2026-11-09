"""Run: python -m pytest test_normalize.py   (or python test_normalize.py)"""
from normalize import normalize, raw_flags


def test_ellipsis_and_em_dash_survive():
    assert normalize("Wait… what — really…") == "Wait… what — really…"
    assert normalize("…") == "…"


def test_curly_quotes_untouched():
    assert normalize("it’s “fine”") == "it’s “fine”"


def test_zero_width_removed():
    assert normalize("he​l﻿l‍o") == "hello"


def test_lookalikes_mapped():
    assert normalize("Неllо wоrld") == "Hello world"  # Cyrillic Н, о
    assert normalize("αρple") == "apple"  # Greek


def test_whitespace_collapsed():
    assert normalize("  a \t\n b  c  ") == "a b c"


def test_raw_flags():
    f = raw_flags("Неllо​ — wait… … x")
    assert f == {"has_zero_width": 1, "has_lookalike": 1, "has_unusual_ws": 1,
                 "n_em_dash": 1, "n_ellipsis": 2}
    assert raw_flags("plain text.") == {"has_zero_width": 0, "has_lookalike": 0,
                                       "has_unusual_ws": 0, "n_em_dash": 0, "n_ellipsis": 0}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

"""Check notebook 11 rule eligibility and hard-score mapping on synthetic text."""
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


root = Path(__file__).resolve().parents[1]
def snapshot():
    return {str(p): p.stat().st_mtime_ns for folder in ["configs", "results"]
            for p in (root / folder).rglob("*") if p.is_file()}

before = snapshot()
shell = InteractiveShell.instance()
original_directory = Path.cwd()
try:
    os.chdir(root / "notebooks")
    os.environ["FORMATTING_RULE_HELPERS_ONLY"] = "1"
    with patch.object(LogisticRegression, "fit", side_effect=AssertionError("Unexpected fit on import")), \
         patch.object(LinearSVC, "fit", side_effect=AssertionError("Unexpected fit on import")):
        result = shell.run_cell("%run ./11_formatting_rule.ipynb")
        if result.error_in_exec:
            raise result.error_in_exec
    assert snapshot() == before and "parts" not in shell.user_ns
    ns = shell.user_ns
    frame = pd.DataFrame({"id": [f"synthetic-{i}" for i in range(12)],
        "text": ["human grass journal river"] * 6 + ["machine summary tokens pattern"] * 6,
        "label": [0] * 6 + [1] * 6})
    frame.loc[9:, "text"] += "\n\nExtra artificial paragraph."
    reference = frame[["id", "text"]].assign(text=lambda f: f.text + " referenceonlymarker")
    shared = None
    for candidate in ns["FORMATTING_GRID"]:
        arm = ns["FormattingArm"](candidate).fit(frame, reference,
            shared=shared if candidate["name"] == "rule_all_wordchar16" else None)
        assert arm.rule_active_ == candidate["rule"]
        assert np.isfinite(arm.decision_function(reference.text)).all()
        if candidate["name"] == "reference":
            shared = arm
        if candidate["name"] == "rule_all_wordchar16":
            assert arm.model_ is shared.model_ and arm.reused_
        if candidate["view"] == "word13":
            assert "referenceonlymarker" not in arm.model_.named_steps["features"].vocabulary_
    # A human counterexample disables the rule even if specialist filtering drops it.
    frame.loc[0, "text"] += "\n\nHuman counterexample."
    candidate = next(c for c in ns["FORMATTING_GRID"] if c["name"] == "rule_no_blank_case_svm1")
    arm = ns["FormattingArm"](candidate).fit(frame, reference)
    assert arm.audit_["human_literal_blank_before_filter"] == 1
    assert arm.audit_["used_human"] == 5 and not arm.rule_active_
    assert not arm.components(reference.text)[2].any()
    base = np.array([-4., 4., -2., 2.])
    flags = np.array([False, False, True, True])
    scores = base + 1000 * flags
    probabilities = ns["hard_format_probabilities"](base, flags, scores)
    assert ((probabilities[flags] >= .99) & (probabilities[flags] < 1)).all()
    assert ((probabilities[~flags] > 0) & (probabilities[~flags] < .99)).all()
    np.testing.assert_array_equal(np.argsort(scores), np.argsort(probabilities))
    # Fixed bands cannot preserve a score order that crosses the flag blocks.
    # Fail openly rather than normalizing an evaluation batch to hide that case.
    try:
        ns["hard_format_probabilities"](np.array([2000., -2000.]),
            np.array([False, True]), np.array([2000., -1000.]))
    except ValueError:
        pass
    else:
        raise AssertionError("An incompatible hard-score ordering passed")
finally:
    os.chdir(original_directory)

print("Synthetic checks passed: all eight pipelines, no-fit import, shared reference, training-only rule and vocabulary, hard ranges/order.")
print("No competition score was recorded.")

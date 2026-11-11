"""Check notebook 03 methods on synthetic text. No competition run or score."""
import json
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell


root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / "notebooks/03_word_baseline.ipynb").read_text())
shell = InteractiveShell.instance()
original_directory = Path.cwd()
try:
    # nbconvert starts notebook kernels in the notebook directory too.
    os.chdir(root / "notebooks")
    for cell_id in ["baseline-03", "baseline-07"]:
        cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
        result = shell.run_cell("".join(cell["source"]))
        if result.error_in_exec:
            raise result.error_in_exec
    namespace = shell.user_ns
    model = namespace["make_baseline"]()
    texts = ["human diary river grass"] * 4 + ["machine summary tokens pattern"] * 4
    model.fit(texts, [0] * 4 + [1] * 4)
    vocabulary = dict(model.named_steps["tfidf"].vocabulary_)
    idf = model.named_steps["tfidf"].idf_.copy()
    frame = pd.DataFrame({
        "id": ["validation-b", "validation-a"],
        "text": ["unseenword privateword diary", "unseenword privateword tokens"],
    })
    predictions = namespace["prediction_frame"](model, frame)
    assert predictions.id.tolist() == frame.id.tolist()
    assert model.named_steps["tfidf"].vocabulary_ == vocabulary
    np.testing.assert_array_equal(model.named_steps["tfidf"].idf_, idf)
    assert "unseenword" not in vocabulary
    assert "privateword" not in vocabulary
    assert predictions.p_ai.between(0, 1).all()
    np.testing.assert_array_equal(predictions.p_ai, model.predict_proba(frame.text)[:, 1])
    assert list(model.classes_) == [0, 1]
    assert model.named_steps["tfidf"].transform(frame.text).dtype == np.float32
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.joblib"
        csv_path = Path(directory) / "predictions.csv"
        joblib.dump(model, model_path)
        restored = namespace["prediction_frame"](joblib.load(model_path), frame)
        np.testing.assert_array_equal(predictions.p_ai, restored.p_ai)
        predictions.to_csv(csv_path, index=False)
        saved = pd.read_csv(csv_path, dtype={"id": "string"}, float_precision="round_trip")
        assert saved.id.tolist() == frame.id.tolist()
        np.testing.assert_array_equal(saved.p_ai, predictions.p_ai)
        # Reverse prediction order. Evaluation must still join by ID.
        truth = pd.DataFrame({"id": frame.id, "label": [0, 1]})
        evaluate = namespace["evaluate_predictions"]
        assert evaluate(truth, saved.iloc[::-1]) == evaluate(truth, predictions)
finally:
    os.chdir(original_directory)

print("Synthetic checks passed: training-only vocabulary/IDF, class 1, ID alignment, reload and CSV round trip.")
print("No competition model was trained and no competition score was recorded.")

"""
validate_svm.py

End-to-end validation of the ported SVM pipeline (pure-Python TD
feature extraction + pure-Python matmul/argmax inference) against
sklearn, on the same raw cycle-4 test windows saved by
fit_and_save_lda.py. Mirrors validate_lda.py's discipline: report
mismatch count and max decision-score error before trusting anything
on the ESP32.

Usage:
    python validate_svm.py
"""

import joblib
import numpy as np

from train_lda_own import extract_features
from td_features_pure import extract_features_pure  # adjust name if yours differs
from svm_infer import predict as pure_predict

MODEL_PATH = "svm_model.joblib"
TEST_WINDOWS_PATH = "test_windows.npz"


def main():
    svm = joblib.load(MODEL_PATH)

    tw = np.load(TEST_WINDOWS_PATH, allow_pickle=True)
    X_test, y_test = tw["X_test"], tw["y_test"]

    # sklearn side: full-batch featurization + predict/decision_function
    X_test_feat, _ = extract_features(X_test)
    sklearn_preds = svm.predict(X_test_feat)
    sklearn_scores = svm.decision_function(X_test_feat)  # (n_samples, n_classes)

    mismatches = 0
    max_score_err = 0.0
    max_feat_err = 0.0

    for i in range(len(X_test)):
        # pure side: per-window raw -> TD features -> matmul/argmax,
        # exactly as it will run on the ESP32
        x_feat_pure = extract_features_pure(X_test[i])
        pure_class, pure_label, pure_scores = pure_predict(x_feat_pure)

        feat_err = max(abs(a - b) for a, b in zip(x_feat_pure, X_test_feat[i]))
        max_feat_err = max(max_feat_err, feat_err)

        if pure_class != sklearn_preds[i]:
            mismatches += 1
            print(f"MISMATCH at row {i}: pure={pure_class} sklearn={sklearn_preds[i]}")

        row_err = max(
            abs(pure_scores[c] - sklearn_scores[i][c]) for c in range(len(pure_scores))
        )
        max_score_err = max(max_score_err, row_err)

    n = len(X_test)
    print(f"\n{mismatches}/{n} mismatches")
    print(f"Max TD feature error:     {max_feat_err:.3e}")
    print(f"Max decision-score error: {max_score_err:.3e}")

    if mismatches == 0:
        print("Port is exact and safe to flash.")
    else:
        print("Port has prediction mismatches -- do not flash until resolved.")


if __name__ == "__main__":
    main()
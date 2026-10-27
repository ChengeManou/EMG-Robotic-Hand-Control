"""
validate_knn.py

Checks that the pure-Python KNN produces the SAME predictions as sklearn
on the saved cycle-4 test windows. Any differing prediction is a failure.
Differences are further labelled TIE (pure class has the max vote, so only
the tie-break order differs) or REAL (pure picked a class with fewer votes).
"""

import joblib
import numpy as np

from train_lda_own import extract_features
from td_features_pure import extract_td_features
from knn_infer import knn_predict as pure_predict  # adjust name if yours differs

MODEL_PATH = "knn_model.joblib"
TEST_WINDOWS_PATH = "test_windows.npz"
EPS = 1e-9


def main():
    knn = joblib.load(MODEL_PATH)

    tw = np.load(TEST_WINDOWS_PATH, allow_pickle=True)
    X_test, y_test = tw["X_test"], tw["y_test"]

    X_test_feat, _ = extract_features(X_test)
    sklearn_preds = knn.predict(X_test_feat)
    sklearn_votes = knn.predict_proba(X_test_feat)
    classes = list(knn.classes_)

    pure_preds = []
    n_diff = n_tie = n_real = 0
    max_vote_err = 0.0
    max_feat_err = 0.0

    for i in range(len(X_test)):
        x_feat_pure = extract_td_features(X_test[i])
        pure_class, pure_label, pure_votes = pure_predict(x_feat_pure)
        pure_preds.append(pure_class)

        max_feat_err = max(
            max_feat_err,
            max(abs(a - b) for a, b in zip(x_feat_pure, X_test_feat[i])),
        )
        max_vote_err = max(
            max_vote_err,
            max(abs(pure_votes[c] - sklearn_votes[i][c]) for c in range(len(pure_votes))),
        )

        if pure_class != sklearn_preds[i]:
            n_diff += 1
            pure_idx = classes.index(pure_class)
            if abs(sklearn_votes[i][pure_idx] - sklearn_votes[i].max()) < EPS:
                n_tie += 1
                print(f"DIFF (tie-break) row {i}: pure={pure_class} sklearn={sklearn_preds[i]}")
            else:
                n_real += 1
                print(f"DIFF (REAL)      row {i}: pure={pure_class} sklearn={sklearn_preds[i]}")

    n = len(X_test)
    pure_preds = np.array(pure_preds)
    agree = n - n_diff

    print(f"\nPrediction agreement with sklearn: {agree}/{n} ({100*agree/n:.3f}%)")
    print(f"Differing predictions: {n_diff}  (tie-break: {n_tie}, real: {n_real})")
    print(f"Accuracy vs true labels -> sklearn: {np.mean(sklearn_preds == y_test):.4f}, "
          f"pure: {np.mean(pure_preds == y_test):.4f}")
    print(f"Max TD feature error:    {max_feat_err:.3e}")
    print(f"Max vote-fraction error: {max_vote_err:.3e}")

    if n_diff == 0:
        print("Predictions identical to sklearn -- safe to flash.")
    elif n_real == 0:
        print("Only tie-break differences -- fix argmax order in knn_infer.py and re-run.")
    else:
        print("Real prediction mismatches -- do not flash until resolved.")


if __name__ == "__main__":
    main()
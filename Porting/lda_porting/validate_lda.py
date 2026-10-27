"""
validate_lda.py

Confirms the ported pipeline (td_features_pure.py + lda_infer.py,
using weights from lda_weights.py) produces the same predictions
and near-identical decision scores as the original sklearn LDA
model, on the exact same raw test windows (cycle 4).

Run order:
    1. train_lda_own.py       (sanity check accuracy - already done: 87.18%)
    2. fit_and_save_lda.py    (saves lda_model.joblib + test_windows.npz)
    3. export_lda_weights.py  (saves lda_weights.py)
    4. validate_lda.py        (this script)

Only after this passes with zero mismatches should lda_weights.py,
lda_infer.py, and td_features_pure.py be copied onto the ESP32.
"""

import numpy as np
import joblib

from train_lda_own import extract_td_features as extract_td_features_numpy
import td_features_pure
import lda_infer


MODEL_PATH = "lda_model.joblib"
TEST_WINDOWS_PATH = "test_windows.npz"


def main():

    print()
    print("=" * 70)
    print("VALIDATE PORTED LDA PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load sklearn model and raw test windows
    # --------------------------------------------------------

    lda = joblib.load(MODEL_PATH)

    data = np.load(TEST_WINDOWS_PATH)
    X_test = data["X_test"]        # (n_windows, 100, 4)
    y_test = data["y_test"]        # (n_windows,)

    label_map_keys = data["label_map_keys"]
    label_map_values = data["label_map_values"]
    label_map = {
        int(k): str(v) for k, v in zip(label_map_keys, label_map_values)
    }

    n_windows = X_test.shape[0]
    print(f"Test windows: {n_windows}")

    # --------------------------------------------------------
    # Run both pipelines on every window
    # --------------------------------------------------------

    mismatches = 0
    max_score_error = 0.0
    mismatch_examples = []

    for i in range(n_windows):

        window_np = X_test[i]              # (100, 4) numpy
        window_list = window_np.tolist()   # list of [ch1..ch4] lists

        # --- sklearn side ---
        feat_np = extract_td_features_numpy(window_np)
        feat_np = feat_np.reshape(1, -1)
        sklearn_pred = int(lda.predict(feat_np)[0])
        sklearn_scores = lda.decision_function(feat_np)[0]

        # --- ported side ---
        feat_pure = td_features_pure.extract_td_features(window_list)
        ported_pred, ported_scores = lda_infer.lda_predict(feat_pure)

        if sklearn_pred != ported_pred:
            mismatches += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    (i, sklearn_pred, ported_pred, int(y_test[i]))
                )

        # Compare scores in the same class order (CLASS_IDS order
        # should equal lda.classes_ order, both ascending)
        for a, b in zip(sklearn_scores, ported_scores):
            err = abs(float(a) - float(b))
            if err > max_score_error:
                max_score_error = err

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Mismatched predictions: {mismatches} / {n_windows}")
    print(f"Max decision-score error: {max_score_error:.2e}")

    if mismatches == 0:
        print()
        print("PASS: ported pipeline matches sklearn exactly.")
        print("Safe to copy td_features_pure.py, lda_infer.py, and")
        print("lda_weights.py onto the ESP32.")
    else:
        print()
        print("FAIL: ported pipeline disagrees with sklearn on "
              f"{mismatches} window(s).")
        print("Examples (index, sklearn_pred, ported_pred, true_label):")
        for idx, sk, pt, true in mismatch_examples:
            sk_name = label_map.get(sk, str(sk))
            pt_name = label_map.get(pt, str(pt))
            true_name = label_map.get(true, str(true))
            print(f"  window {idx}: sklearn={sk_name}  ported={pt_name}  "
                  f"true={true_name}")
        print()
        print("Do not trust the ESP32 firmware until this is fixed -")
        print("check that CLASS_IDS order in lda_weights.py matches")
        print("lda.classes_, and that td_features_pure.py's ZC/SSC")
        print("thresholds match train_lda_own.py exactly.")

    print("=" * 70)


if __name__ == "__main__":
    main()

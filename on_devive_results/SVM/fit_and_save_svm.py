"""
fit_and_save_svm.py

Fits the linear SVM model for the own-arm dataset.

Configuration:
    Window size = 150 ms
    Overlap     = 75%
    Training    = cycles 1-3
    Testing     = cycle 4

Outputs:
    svm_model.joblib
    svm_parameters.npz
"""

import joblib
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from my_data_loader import (
    build_windowed_dataset,
    train_test_split_by_cycle
)

from train_lda_own import (
    extract_features,
    TRAIN_CYCLES,
    TEST_CYCLES
)

# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_MS = 150
OVERLAP = 0.75

C = 0.1
MAX_ITER = 10000

MODEL_PATH = "svm_model.joblib"
PARAMETERS_PATH = "svm_parameters.npz"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRAINING LINEAR SVM")
    print("=" * 70)

    print(f"Window size : {WINDOW_MS} ms")
    print(f"Overlap     : {OVERLAP * 100:.0f}%")
    print(f"SVM         : LinearSVC")
    print(f"C           : {C}")
    print()

    # --------------------------------------------------------
    # Load windowed dataset
    # --------------------------------------------------------

    windows = build_windowed_dataset(
        window_ms=WINDOW_MS,
        overlap=OVERLAP
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        label_map
    ) = train_test_split_by_cycle(
        windows,
        train_cycles=TRAIN_CYCLES,
        test_cycles=TEST_CYCLES
    )

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # --------------------------------------------------------
    # Extract TD features
    # --------------------------------------------------------

    X_train_feat, train_feature_time = extract_features(X_train)
    X_test_feat, test_feature_time = extract_features(X_test)

    print("X_train features:", X_train_feat.shape)
    print("X_test features :", X_test_feat.shape)

    # --------------------------------------------------------
    # Train linear SVM
    # --------------------------------------------------------

    svm = LinearSVC(C=C, max_iter=MAX_ITER)
    svm.fit(X_train_feat, y_train)

    print()
    print("SVM classes   :", svm.classes_)
    print("Coef shape    :", svm.coef_.shape)
    print("Intercept shape:", svm.intercept_.shape)

    # --------------------------------------------------------
    # Test on cycle 4
    # --------------------------------------------------------

    y_pred = svm.predict(X_test_feat)
    accuracy = accuracy_score(y_test, y_pred)

    # Recover string labels from label_map {string: int}
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [label_map[c] for c in svm.classes_]

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Overall accuracy: {accuracy * 100:.2f}%")

    # --------------------------------------------------------
    # Per-class report
    # --------------------------------------------------------

    print()
    print("Per-class results:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
    col_w = max(len(n) for n in class_names) + 2

    print("Confusion matrix (rows = true, cols = predicted):")
    print()

    # Header
    header = " " * col_w + "".join(n.rjust(col_w) for n in class_names)
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        row_str = class_names[i].rjust(col_w)
        row_str += "".join(str(v).rjust(col_w) for v in row)
        print(row_str)

    print()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    joblib.dump(svm, MODEL_PATH)
    print(f"[saved] {MODEL_PATH}")

    np.savez(
        PARAMETERS_PATH,
        coef=svm.coef_,
        intercept=svm.intercept_,
        classes=svm.classes_
    )
    print(f"[saved] {PARAMETERS_PATH}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Window size        : {WINDOW_MS} ms")
    print(f"Overlap            : {OVERLAP * 100:.0f}%")
    print(f"Window samples     : {X_train.shape[1]}")
    print(f"Number of features : {X_train_feat.shape[1]}")
    print(f"Training windows   : {len(y_train)}")
    print(f"Testing windows    : {len(y_test)}")
    print(f"Test accuracy      : {accuracy * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
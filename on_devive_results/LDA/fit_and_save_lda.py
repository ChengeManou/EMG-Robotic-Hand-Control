"""
fit_and_save_lda.py

Trains the LDA model on the full training set (cycles 1-3)
using the selected 150 ms / 75% overlap window configuration
and the 16-dimensional time-domain feature pipeline.

Saves:

    lda_model.joblib
        Fitted sklearn LDA model.

    test_windows.npz
        Raw test windows from cycle 4, test labels and label map.

    lda_parameters.npz
        LDA coefficients, intercepts and class labels for
        MicroPython/ESP32 deployment.

The saved test windows can be used by validate_lda.py to
confirm that the ported MicroPython feature extraction and
LDA inference match sklearn before testing on the ESP32.
"""

import joblib
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

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

MODEL_PATH = "lda_model.joblib"

TEST_WINDOWS_PATH = "test_windows.npz"

LDA_PARAMETERS_PATH = "lda_parameters.npz"


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FIT AND SAVE LDA")
    print("=" * 70)

    print()
    print(
        f"Window size : {WINDOW_MS} ms"
    )

    print(
        f"Overlap     : {OVERLAP * 100:.0f}%"
    )

    print(
        f"Training cycles: {TRAIN_CYCLES}"
    )

    print(
        f"Testing cycles : {TEST_CYCLES}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

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

    print()
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

    print()
    print("=" * 70)
    print("EXTRACTING FEATURES")
    print("=" * 70)

    X_train_feat, train_feature_time = (
        extract_features(X_train)
    )

    X_test_feat, test_feature_time = (
        extract_features(X_test)
    )

    print()
    print(
        "X_train features:",
        X_train_feat.shape
    )

    print(
        "X_test features :",
        X_test_feat.shape
    )

    # ========================================================
    # TRAIN LDA
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING LDA")
    print("=" * 70)

    lda = LinearDiscriminantAnalysis(
        solver="svd"
    )

    lda.fit(
        X_train_feat,
        y_train
    )

    # ========================================================
    # VERIFY MODEL
    # ========================================================

    y_pred = lda.predict(
        X_test_feat
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print()
    print(
        f"Test accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print("LDA MODEL")
    print("=" * 70)

    print(
        "Classes:"
    )

    print(
        lda.classes_
    )

    print()
    print(
        "Coefficient shape:",
        lda.coef_.shape
    )

    print(
        "Intercept shape:",
        lda.intercept_.shape
    )

    # ========================================================
    # SAVE SKLEARN MODEL
    # ========================================================

    joblib.dump(
        lda,
        MODEL_PATH
    )

    print()
    print(
        f"[saved] {MODEL_PATH}"
    )

    # ========================================================
    # SAVE LDA PARAMETERS FOR ESP32
    # ========================================================

    np.savez(
        LDA_PARAMETERS_PATH,

        coef=lda.coef_,

        intercept=lda.intercept_,

        classes=lda.classes_
    )

    print(
        f"[saved] {LDA_PARAMETERS_PATH}"
    )

    # ========================================================
    # SAVE RAW TEST WINDOWS
    # ========================================================

    label_map_keys = np.array(
        sorted(label_map.keys()),
        dtype=np.int64
    )

    label_map_values = np.array(
        [
            label_map[k]
            for k in label_map_keys
        ]
    )

    np.savez(
        TEST_WINDOWS_PATH,

        X_test=X_test,

        y_test=y_test,

        label_map_keys=label_map_keys,

        label_map_values=label_map_values
    )

    print(
        f"[saved] {TEST_WINDOWS_PATH}"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Window size       : "
        f"{WINDOW_MS} ms"
    )

    print(
        f"Overlap            : "
        f"{OVERLAP * 100:.0f}%"
    )

    print(
        f"Window samples     : "
        f"{X_train.shape[1]}"
    )

    print(
        f"TD features        : "
        f"{X_train_feat.shape[1]}"
    )

    print(
        f"Training windows   : "
        f"{len(y_train)}"
    )

    print(
        f"Testing windows    : "
        f"{len(y_test)}"
    )

    print(
        f"Number of classes  : "
        f"{len(lda.classes_)}"
    )

    print(
        f"Test accuracy      : "
        f"{accuracy * 100:.2f}%"
    )

    print()
    print("Saved files:")

    print(
        f"  {MODEL_PATH}"
    )

    print(
        f"  {TEST_WINDOWS_PATH}"
    )

    print(
        f"  {LDA_PARAMETERS_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
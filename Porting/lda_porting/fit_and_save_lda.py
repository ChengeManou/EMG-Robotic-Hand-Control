"""
fit_and_save_lda.py

Trains the LDA model on the full training set (cycles 1-3) using the
same 16-dim TD feature pipeline as train_lda_own.py, then saves:

    lda_model.joblib     - the fitted sklearn LDA model
    test_windows.npz     - raw (un-featurized) test windows (cycle 4),
                            y_test, and label_map, for use by
                            validate_lda.py to confirm the ported
                            MicroPython inference matches sklearn
                            exactly before trusting it on the ESP32.

Run this after training/evaluating with train_lda_own.py and being
happy with the result (87.18% on the 7-class set).
"""

import joblib
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from my_data_loader import (
    build_windowed_dataset,
    train_test_split_by_cycle
)
from train_lda_own import extract_features, TRAIN_CYCLES, TEST_CYCLES


MODEL_PATH = "lda_model.joblib"
TEST_WINDOWS_PATH = "test_windows.npz"


def main():

    print()
    print("=" * 70)
    print("FIT AND SAVE LDA")
    print("=" * 70)

    windows = build_windowed_dataset()

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

    # --------------------------------------------------------
    # Extract features (train only needed to fit the model)
    # --------------------------------------------------------

    X_train_feat, _ = extract_features(X_train)

    # --------------------------------------------------------
    # Train LDA
    # --------------------------------------------------------

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_feat, y_train)

    print()
    print("coef_ shape     :", lda.coef_.shape)
    print("intercept_ shape:", lda.intercept_.shape)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(lda, MODEL_PATH)
    print()
    print(f"[saved] {MODEL_PATH}")

    # --------------------------------------------------------
    # Save RAW test windows (not features) + labels + label_map,
    # so validate_lda.py can run the full ported pipeline
    # (feature extraction + inference) end to end and compare
    # against sklearn on the exact same data.
    # --------------------------------------------------------

    label_map_keys = np.array(sorted(label_map.keys()))
    label_map_values = np.array([label_map[k] for k in label_map_keys])

    np.savez(
        TEST_WINDOWS_PATH,
        X_test=X_test,
        y_test=y_test,
        label_map_keys=label_map_keys,
        label_map_values=label_map_values
    )
    print(f"[saved] {TEST_WINDOWS_PATH}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

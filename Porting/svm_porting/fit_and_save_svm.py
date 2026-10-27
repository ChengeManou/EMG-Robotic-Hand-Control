"""
fit_and_save_svm.py

Trains a linear SVM (LinearSVC, one-vs-rest) on the same 16-dim TD
feature pipeline as fit_and_save_lda.py / train_lda_own.py, on the
same cycle 1-3 / cycle 4 split, so results are directly comparable
to the LDA model. Saves:

    svm_model.joblib      - the fitted sklearn LinearSVC model

test_windows.npz is NOT regenerated here: it holds raw (un-featurized)
cycle-4 windows + y_test + label_map, which are classifier-agnostic.
The one saved by fit_and_save_lda.py is reused as-is by validate_svm.py.
If it isn't on disk yet, run fit_and_save_lda.py first (or lift the
save block from it) before running this.

Run this after being happy with an equivalent train/eval pass using
the same TRAIN_CYCLES/TEST_CYCLES split as train_lda_own.py.
"""

import joblib
import numpy as np

from sklearn.svm import LinearSVC

from my_data_loader import (
    build_windowed_dataset,
    train_test_split_by_cycle
)
from train_lda_own import extract_features, TRAIN_CYCLES, TEST_CYCLES


MODEL_PATH = "svm_model.joblib"

# LinearSVC's only real hyperparameter. Start at 1.0; if train/test gap
# is large, sweep on a log grid (0.01, 0.1, 1, 10, 100) using the same
# cycle-based split discipline. Don't reuse WyoFlex RBF hyperparameters
# (C=100, gamma=0.001) -- those were tuned for a different kernel and
# subject and won't transfer.
C = 1.0
MAX_ITER = 10_000


def main():

    print()
    print("=" * 70)
    print("FIT AND SAVE SVM")
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
    # Train linear SVM
    # --------------------------------------------------------

    svm = LinearSVC(C=C, max_iter=MAX_ITER, dual="auto")
    svm.fit(X_train_feat, y_train)

    print()
    print("classes_        :", svm.classes_)
    print("coef_ shape     :", svm.coef_.shape)
    print("intercept_ shape:", svm.intercept_.shape)

    # Sanity check: a class dropping out here would silently produce a
    # coef_ with fewer rows than label_map has entries -- exactly what
    # happened with thumb_up during the LDA export. Catch it now.
    if len(svm.classes_) != len(label_map):
        raise RuntimeError(
            f"Class count mismatch: fitted on {len(svm.classes_)} classes "
            f"but label_map has {len(label_map)}. Check y_train counts "
            f"before .fit() to find which class dropped."
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(svm, MODEL_PATH)
    print()
    print(f"[saved] {MODEL_PATH}")
    print()
    print("Reusing existing test_windows.npz (raw windows are classifier-"
          "agnostic) -- run validate_svm.py next.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
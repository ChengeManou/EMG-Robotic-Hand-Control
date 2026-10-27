"""
fit_and_save_knn.py

Trains a KNeighborsClassifier (k=8) on the same 16-dim TD feature
pipeline as fit_and_save_lda.py / fit_and_save_svm.py / fit_and_save_rf.py,
on the same cycle 1-3 / cycle 4 split, for direct comparison. Saves:

    knn_model.joblib     - the fitted sklearn model (for host-side
                            reference/re-eval only)

    knn_train_data.npz  - X_train_feat (already TD-extracted, float32),
                           y_train, label_map -- this is what
                           export_knn_weights.py actually needs, since
                           KNN's "model" IS the training set. Not
                           relying on sklearn's private _fit_X/_fit_y
                           attributes, which aren't a stable API.

test_windows.npz is reused as-is (classifier-agnostic raw windows),
same as SVM/RF.
"""

import joblib
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from my_data_loader import (
    build_windowed_dataset,
    train_test_split_by_cycle
)
from train_lda_own import extract_features, TRAIN_CYCLES, TEST_CYCLES


MODEL_PATH = "knn_model.joblib"
TRAIN_DATA_PATH = "knn_train_data.npz"

# Selected window configuration
WINDOW_MS = 150
OVERLAP = 0.75

K = 8


def main():

    print()
    print("=" * 70)
    print("FIT AND SAVE KNN")
    print("=" * 70)

    print()
    print(f"Window size : {WINDOW_MS} ms")
    print(f"Overlap     : {OVERLAP * 100:.0f}%")
    print(f"K           : {K}")

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

    X_train_feat, _ = extract_features(X_train)
    X_test_feat, _ = extract_features(X_test)

    knn = KNeighborsClassifier(
        n_neighbors=K
    )

    knn.fit(
        X_train_feat,
        y_train
    )

    train_acc = knn.score(
        X_train_feat,
        y_train
    )

    test_acc = knn.score(
        X_test_feat,
        y_test
    )

    print(
        f"Train accuracy: {train_acc:.4f}"
    )

    print(
        f"Test accuracy:  {test_acc:.4f}"
    )

    if len(knn.classes_) != len(label_map):
        raise RuntimeError(
            f"Fitted on {len(knn.classes_)} classes but label_map has "
            f"{len(label_map)} entries -- a class was dropped upstream."
        )

    joblib.dump(
        knn,
        MODEL_PATH
    )

    print(
        f"[saved] {MODEL_PATH}"
    )

    label_map_keys = np.array(
        sorted(label_map.keys())
    )

    label_map_values = np.array(
        [label_map[k] for k in label_map_keys]
    )

    np.savez(
        TRAIN_DATA_PATH,
        X_train_feat=X_train_feat.astype(np.float32),
        y_train=y_train.astype(int),
        label_map_keys=label_map_keys,
        label_map_values=label_map_values,
    )

    print(
        f"[saved] {TRAIN_DATA_PATH}"
    )

    print()

    print(
        f"Training set size for export: "
        f"{X_train_feat.shape[0]} samples "
        f"x {X_train_feat.shape[1]} features = "
        f"{X_train_feat.size * 4 / 1024:.1f} KB as float32"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
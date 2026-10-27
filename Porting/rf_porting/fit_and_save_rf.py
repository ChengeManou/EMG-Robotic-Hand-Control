"""
fit_and_save_rf.py

Trains a RandomForestClassifier (small/embedded-friendly config: 20
trees, max_depth=6 -- NOT sklearn defaults, which produced an
unworkably large export) on the same 16-dim TD feature pipeline as
fit_and_save_lda.py / fit_and_save_svm.py, on the same cycle 1-3 /
cycle 4 split, for direct comparison. Saves:

    rf_model.joblib

test_windows.npz is reused as-is (classifier-agnostic raw windows),
same as the SVM port.

IMPORTANT: prints total node count across all 100 trees and an
estimated export size BEFORE you commit to running export_rf_weights.py
-- unlimited-depth RF on this data volume can easily produce hundreds
of thousands of nodes, which may not fit ESP32-S3 flash alongside
everything else already deployed. Check this number before proceeding.
"""

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from my_data_loader import (
    build_windowed_dataset,
    train_test_split_by_cycle
)
from train_lda_own import extract_features, TRAIN_CYCLES, TEST_CYCLES


MODEL_PATH = "rf_model.joblib"

N_ESTIMATORS = 20   # small, embedded-friendly (was 100)
MAX_DEPTH = 6        # capped (was None/unlimited) -- this is what actually
                      # controls export size; node count per tree is bounded
                      # by ~2^(MAX_DEPTH+1)-1 = 127 nodes max at depth 6
RANDOM_STATE = 42   # for reproducibility; not an sklearn default but recommended

BYTES_PER_NODE_ESTIMATE = 44  # 3x int32 + threshold f32 + 7x class-proba f32


def main():

    print()
    print("=" * 70)
    print("FIT AND SAVE RANDOM FOREST")
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

    X_train_feat, _ = extract_features(X_train)

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train_feat, y_train)

    print()
    print("classes_:", rf.classes_)

    if len(rf.classes_) != len(label_map):
        raise RuntimeError(
            f"Fitted on {len(rf.classes_)} classes but label_map has "
            f"{len(label_map)} entries -- a class was dropped upstream."
        )

    # --------------------------------------------------------
    # Report actual tree size BEFORE export -- this is the number
    # that determines whether this fits on the ESP32 at all.
    # --------------------------------------------------------
    total_nodes = sum(tree.tree_.node_count for tree in rf.estimators_)
    node_counts = [tree.tree_.node_count for tree in rf.estimators_]
    est_bytes = total_nodes * BYTES_PER_NODE_ESTIMATE

    print()
    print(f"Trees: {len(rf.estimators_)}")
    print(f"Nodes per tree: min={min(node_counts)} max={max(node_counts)} "
          f"mean={sum(node_counts)/len(node_counts):.0f}")
    print(f"TOTAL NODES: {total_nodes}")
    print(f"Estimated export size: {est_bytes/1024:.1f} KB "
          f"({est_bytes/1024/1024:.2f} MB) at ~{BYTES_PER_NODE_ESTIMATE} bytes/node")
    print()
    print("Check this against your ESP32-S3's available flash before "
          "running export_rf_weights.py. If it's too large, re-run this "
          "script with a smaller N_ESTIMATORS and/or a MAX_DEPTH cap.")

    joblib.dump(rf, MODEL_PATH)
    print()
    print(f"[saved] {MODEL_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()

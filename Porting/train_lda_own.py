"""
train_lda_own.py

Train and evaluate LDA on the user's own EMG dataset.

Pipeline:

    CSV cycles
        ↓
    remove rest
        ↓
    150ms windows / 75% step
        ↓
    16 TD features
        ↓
    LDA
        ↓
    classification results
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from my_data_loader import (
    build_windowed_dataset,
    train_test_split_by_cycle
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CYCLES = (1, 2, 3)
TEST_CYCLES = (6,)


# ============================================================
# TIME-DOMAIN FEATURES
# ============================================================

def mav(x):
    return np.mean(np.abs(x))


def waveform_length(x):
    return np.sum(np.abs(np.diff(x)))


def zero_crossings(x, threshold=10):

    x1 = x[:-1]
    x2 = x[1:]

    crossings = (
        ((x1 >= 0) & (x2 < 0)) |
        ((x1 < 0) & (x2 >= 0))
    )

    amplitude_difference = np.abs(x1 - x2)

    crossings &= (
        amplitude_difference >= threshold
    )

    return np.sum(crossings)


def slope_sign_changes(x, threshold=10):

    d1 = x[1:-1] - x[:-2]
    d2 = x[2:] - x[1:-1]

    changes = (
        ((d1 > 0) & (d2 < 0)) |
        ((d1 < 0) & (d2 > 0))
    )

    amplitude_difference = np.abs(
        x[2:] - x[:-2]
    )

    changes &= (
        amplitude_difference >= threshold
    )

    return np.sum(changes)


def extract_td_features(window):

    features = []

    for ch in range(window.shape[1]):

        x = window[:, ch].astype(np.float64)

        # Mean removal
        x = x - np.mean(x)

        # MAV
        features.append(
            mav(x)
        )

        # ZC
        features.append(
            zero_crossings(x)
        )

        # SSC
        features.append(
            slope_sign_changes(x)
        )

        # WL
        features.append(
            waveform_length(x)
        )

    return np.asarray(
        features,
        dtype=np.float64
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(X):

    features = []

    start = time.perf_counter()

    for window in X:

        features.append(
            extract_td_features(window)
        )

    elapsed = (
        time.perf_counter() - start
    )

    features = np.asarray(
        features,
        dtype=np.float64
    )

    return features, elapsed


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def plot_confusion_matrices(y_true, y_pred, labels, names, accuracy,
                            title_prefix="LDA",
                            save_path="lda_confusion_matrix.png"):

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Left: raw counts
    cm_counts = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    disp1 = ConfusionMatrixDisplay(
        cm_counts,
        display_labels=names
    )

    disp1.plot(
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
        xticks_rotation=45,
        values_format="d"
    )

    axes[0].set_title("Confusion matrix (counts)")

    # Right: row-normalised
    cm_norm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true"
    )

    disp2 = ConfusionMatrixDisplay(
        cm_norm,
        display_labels=names
    )

    disp2.plot(
        ax=axes[1],
        cmap="Blues",
        colorbar=False,
        xticks_rotation=45,
        values_format=".2f"
    )

    axes[1].set_title("Confusion matrix (row-normalised)")

    fig.suptitle(
        f"{title_prefix} — accuracy {accuracy * 100:.2f}%"
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("OWN EMG DATASET — LDA")
    print("=" * 70)

    print()
    print("Training cycles:", TRAIN_CYCLES)
    print("Testing cycles :", TEST_CYCLES)

    # ========================================================
    # LOAD DATA
    # ========================================================

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

    # ========================================================
    # LABEL MAP
    # ========================================================

    print()
    print("=" * 70)
    print("LABEL MAP")
    print("=" * 70)

    print(label_map)

    # ========================================================
    # CHECK LABEL COUNTS
    # ========================================================

    print()
    print("Training label counts:")

    for label in sorted(label_map):

        print(
            f"  {label}: "
            f"{label_map[label]:20s} "
            f"{np.sum(y_train == label)} samples"
        )

    print()
    print("Testing label counts:")

    for label in sorted(label_map):

        print(
            f"  {label}: "
            f"{label_map[label]:20s} "
            f"{np.sum(y_test == label)} samples"
        )

    # ========================================================
    # WINDOW DATA
    # ========================================================

    print()
    print("=" * 70)
    print("WINDOW DATA")
    print("=" * 70)

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

    print()
    print("=" * 70)
    print("EXTRACTING 16 TD FEATURES")
    print("=" * 70)

    X_train_feat, train_feat_time = (
        extract_features(X_train)
    )

    X_test_feat, test_feat_time = (
        extract_features(X_test)
    )

    print(
        "Training features:",
        X_train_feat.shape
    )

    print(
        "Testing features :",
        X_test_feat.shape
    )

    print(
        f"Training feature extraction: "
        f"{train_feat_time:.3f} s"
    )

    print(
        f"Testing feature extraction: "
        f"{test_feat_time:.3f} s"
    )

    # ========================================================
    # TRAIN LDA
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING LDA")
    print("=" * 70)

    lda = LinearDiscriminantAnalysis()

    train_start = time.perf_counter()

    lda.fit(
        X_train_feat,
        y_train
    )

    train_time = (
        time.perf_counter()
        - train_start
    )

    print(
        f"LDA training time: "
        f"{train_time:.6f} s"
    )

    # ========================================================
    # TEST
    # ========================================================

    print()
    print("=" * 70)
    print("TESTING LDA")
    print("=" * 70)

    inference_start = time.perf_counter()

    y_pred = lda.predict(
        X_test_feat
    )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print()
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    # ========================================================
    # TIMING
    # ========================================================

    n_test = len(y_test)

    print()
    print("Timing")
    print("-" * 40)

    print(
        f"Total LDA inference: "
        f"{inference_time:.6f} s"
    )

    print(
        f"Average LDA inference: "
        f"{inference_time / n_test * 1e6:.2f} us/window"
    )

    print(
        f"Average feature extraction: "
        f"{test_feat_time / n_test * 1e6:.2f} us/window"
    )

    print(
        f"Average total: "
        f"{(inference_time + test_feat_time) / n_test * 1e6:.2f} us/window"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report_labels = sorted(
        label_map.keys()
    )

    target_names = [
        label_map[i]
        for i in report_labels
    ]

    print(
        classification_report(
            y_test,
            y_pred,
            labels=report_labels,
            target_names=target_names,
            zero_division=0
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    # Numeric matrix in the terminal (remove if not needed)
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=report_labels
        )
    )

    # Plot: counts + row-normalised, saved to lda_confusion_matrix.png
    plot_confusion_matrices(
        y_test,
        y_pred,
        labels=report_labels,
        names=target_names,
        accuracy=accuracy
    )

    # ========================================================
    # PER-CLASS ACCURACY
    # ========================================================

    print()
    print("=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

    for i in report_labels:

        mask = (
            y_test == i
        )

        if np.sum(mask) == 0:
            continue

        class_accuracy = np.mean(
            y_pred[mask] == i
        )

        print(
            f"{label_map[i]:25s} "
            f"{class_accuracy * 100:6.2f}% "
            f"({np.sum(mask)} samples)"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Number of TD features : "
        f"{X_train_feat.shape[1]}"
    )

    print(
        f"Training samples      : "
        f"{len(y_train)}"
    )

    print(
        f"Testing samples       : "
        f"{len(y_test)}"
    )

    print(
        f"LDA accuracy          : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average inference     : "
        f"{inference_time / n_test * 1e6:.2f} us"
    )

    print(
        f"Average feature time  : "
        f"{test_feat_time / n_test * 1e6:.2f} us"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
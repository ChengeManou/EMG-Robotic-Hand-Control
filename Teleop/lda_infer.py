"""
lda_infer.py

Pure-Python (no numpy) LDA inference using weights exported by
export_lda_weights.py (lda_weights.py must be importable alongside
this file).

sklearn's LDA (default 'svd' solver, decision_function) computes:

    scores[k] = x . coef_[k] + intercept_[k]

and predicts the class with the highest score. This module
reproduces that exactly with plain Python lists, so it runs on the
ESP32-S3 in MicroPython with no numpy dependency.
"""

from lda_weights import COEF, INTERCEPT, CLASS_IDS, N_CLASSES, N_FEATURES, LABELS


def _dot(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def lda_scores(features):
    """
    features: flat list of N_FEATURES floats (from td_features_pure).

    Returns a list of N_CLASSES raw decision scores, in the same
    order as CLASS_IDS.
    """
    if len(features) != N_FEATURES:
        raise ValueError(
            "expected {} features, got {}".format(N_FEATURES, len(features))
        )

    scores = []
    for k in range(N_CLASSES):
        scores.append(_dot(features, COEF[k]) + INTERCEPT[k])
    return scores


def lda_predict(features):
    """
    Returns (class_id, scores) - class_id matches the original
    sklearn training labels (CLASS_IDS values, not row index).
    """
    scores = lda_scores(features)

    best_idx = 0
    best_score = scores[0]
    for k in range(1, N_CLASSES):
        if scores[k] > best_score:
            best_score = scores[k]
            best_idx = k

    return CLASS_IDS[best_idx], scores


def lda_predict_label(features):
    """
    Same as lda_predict, but returns (gesture_name, class_id, scores)
    using LABELS from lda_weights.py - convenient for driving the
    servo mapping directly by gesture name.
    """
    class_id, scores = lda_predict(features)
    return LABELS[class_id], class_id, scores

"""
svm_infer.py

Pure-Python (no numpy) inference for the ported linear SVM. Structurally
identical to lda_infer.py: decision_function is coef @ x + intercept per
class, prediction is argmax. Intended for MicroPython on ESP32 alongside
td_features_pure.py, which supplies the 16-dim TD feature vector `x`.

LinearSVC's decision_function has no probability interpretation like
LDA's does -- it's a raw signed distance to each OvR hyperplane. argmax
is still the correct decision rule, but don't treat the margin as a
confidence score without calibration.
"""

from svm_weights import COEF, INTERCEPT, CLASS_IDS, LABELS, N_CLASSES, N_FEATURES


def decision_function(x):
    """x: list of N_FEATURES floats. Returns list of N_CLASSES scores."""
    scores = []
    for c in range(N_CLASSES):
        row = COEF[c]
        s = INTERCEPT[c]
        for i in range(N_FEATURES):
            s += row[i] * x[i]
        scores.append(s)
    return scores


def predict(x):
    """Returns (predicted_class_id, predicted_label, scores)."""
    scores = decision_function(x)
    best_i = 0
    best_s = scores[0]
    for i in range(1, N_CLASSES):
        if scores[i] > best_s:
            best_s = scores[i]
            best_i = i
    return CLASS_IDS[best_i], LABELS[best_i], scores

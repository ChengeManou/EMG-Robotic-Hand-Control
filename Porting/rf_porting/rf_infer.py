"""
rf_infer.py

Pure-Python (no numpy) inference for the ported Random Forest.
Because the exported tree data (rf_nodes.bin) may be several MB --
too large to load wholesale into ESP32 RAM -- nodes are read one at a
time via file seek, exactly like test_data.py's get_window(). The file
is opened once at import and kept open for the life of the program.

predict_proba(x) traverses every tree from its root (TREE_STARTS[t])
down to a leaf, following child_left/child_right based on
x[feature] <= threshold, and accumulates that leaf's normalized class
distribution. The final probability vector is the average across all
trees -- this exactly matches
RandomForestClassifier.predict_proba()/.predict(), so it stays
comparable bit-for-bit against sklearn in validate_rf.py.
"""

import ustruct as struct  # MicroPython; falls back to struct on desktop
try:
    import ustruct as struct
except ImportError:
    import struct

from rf_weights import (
    N_TREES, N_CLASSES, N_FEATURES, CLASS_IDS, LABELS,
    TREE_STARTS, RECORD_FMT, RECORD_SIZE, BIN_PATH
)

_f = open(BIN_PATH, 'rb')  # kept open for the life of the program


def _read_node(idx):
    _f.seek(idx * RECORD_SIZE)
    buf = _f.read(RECORD_SIZE)
    fields = struct.unpack(RECORD_FMT, buf)
    child_left, child_right, feature, threshold = fields[0], fields[1], fields[2], fields[3]
    proba = fields[4:4 + N_CLASSES]
    return child_left, child_right, feature, threshold, proba


def predict_proba(x):
    """x: list of N_FEATURES floats. Returns list of N_CLASSES averaged probabilities."""
    total = [0.0] * N_CLASSES
    for t in range(N_TREES):
        node = TREE_STARTS[t]
        child_left, child_right, feature, threshold, proba = _read_node(node)
        while child_left != -1:
            if x[feature] <= threshold:
                node = child_left
            else:
                node = child_right
            child_left, child_right, feature, threshold, proba = _read_node(node)
        for c in range(N_CLASSES):
            total[c] += proba[c]
    for c in range(N_CLASSES):
        total[c] /= N_TREES
    return total


def rf_predict(x):
    """Returns (predicted_class_id, predicted_label, proba) -- 3-tuple,
    matching svm_infer's predict(), NOT lda_predict's 2-tuple. Check
    which arity your caller expects."""
    proba = predict_proba(x)
    best_i = 0
    for i in range(1, N_CLASSES):
        if proba[i] > proba[best_i]:
            best_i = i
    return CLASS_IDS[best_i], LABELS[best_i], proba

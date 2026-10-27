"""
knn_infer.py

Pure-Python (no numpy) KNN inference. Because KNN's "model" IS the
training set, this reads knn_train.bin one training sample at a time
via file seek (like rf_infer.py/test_data.py), rather than loading
all 3,276 x 16 floats into RAM at once.

For each query, computes squared Euclidean distance to every training
sample (sqrt is skipped -- monotonic, doesn't change ranking), keeps a
running top-K (K=8) via manual insertion (no numpy, no heapq
dependency), then majority-votes the K nearest labels.

TIE-BREAKING: on a vote tie, this picks RANDOMLY among the tied
classes (not sklearn's deterministic behavior). This means the port
is NON-DETERMINISTIC on tied inputs -- the same window can predict
different gestures across runs -- and validate_knn.py will show
mismatches on tied rows even when the port is working correctly,
since sklearn's tie-break is deterministic and this one isn't.

This is expected to be the SLOWEST of the four ports: O(N_TRAIN) work
per query with no shortcuts, vs LDA/SVM's single matmul or RF's
tree-depth traversal.
"""

from array import array
try:
    import urandom as random  # MicroPython
except ImportError:
    import random  # desktop, for validate_knn.py

from knn_weights import (
    K, N_TRAIN, N_FEATURES, N_CLASSES, CLASS_IDS, LABELS,
    BIN_PATH, RECORD_SIZE, Y_TRAIN
)


def _random_index(n):
    """Random int in [0, n). Uses getrandbits, since randrange() isn't
    guaranteed present across all MicroPython ports' random module."""
    if n <= 1:
        return 0
    bits = 1
    while (1 << bits) < n:
        bits += 1
    while True:
        r = random.getrandbits(bits)
        if r < n:
            return r

_f = open(BIN_PATH, 'rb')  # kept open for the life of the program


def knn_predict(x):
    """x: list of N_FEATURES floats. Returns (class_id, label, vote_counts)."""
    best_d = []  # ascending squared distances, len <= K
    best_l = []  # corresponding labels, same order

    for i in range(N_TRAIN):
        _f.seek(i * RECORD_SIZE)
        buf = _f.read(RECORD_SIZE)
        sample = array('f', buf)

        d = 0.0
        for j in range(N_FEATURES):
            diff = x[j] - sample[j]
            d += diff * diff

        label = Y_TRAIN[i]

        if len(best_d) < K:
            pos = len(best_d)
            while pos > 0 and best_d[pos - 1] > d:
                pos -= 1
            best_d.insert(pos, d)
            best_l.insert(pos, label)
        elif d < best_d[-1]:
            pos = K - 1
            while pos > 0 and best_d[pos - 1] > d:
                pos -= 1
            best_d.insert(pos, d)
            best_l.insert(pos, label)
            best_d.pop()
            best_l.pop()

    # majority vote, ties broken RANDOMLY among tied classes
    counts = [0] * N_CLASSES
    for label in best_l:
        idx = CLASS_IDS.index(label)
        counts[idx] += 1

    max_count = max(counts)
    tied = [i for i in range(N_CLASSES) if counts[i] == max_count]
    best_i = tied[0] if len(tied) == 1 else tied[_random_index(len(tied))]

    return CLASS_IDS[best_i], LABELS[best_i], counts

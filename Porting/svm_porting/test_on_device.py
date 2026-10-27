"""
test_on_device.py

Runs ON THE ESP32 (MicroPython). Loops over the exported test_data.py
windows, runs the full pipeline (TD feature extraction + classifier
inference) per window exactly as production does, and reports:

  - accuracy against Y_TRUE
  - per-stage timing: feature extraction, inference, total
  - min / mean / median / max for each stage (ms)

Flip CLASSIFIER between "lda" and "svm" to test either port without
duplicating the script -- both expose the same predict(x) -> (class_id,
label, scores) interface, so this file doesn't otherwise change.

Copy alongside on the device:
    test_data.py            (from export_test_windows.py)
    td_features_pure.py
    lda_infer.py + lda_weights.py   (if CLASSIFIER == "lda")
    svm_infer.py + svm_weights.py   (if CLASSIFIER == "svm")

Usage (on-device, e.g. via Thonny):
    import test_on_device
"""

import utime

from test_data import N_WINDOWS, Y_TRUE, get_window
from td_features_pure import extract_features_pure  # adjust name if yours differs

CLASSIFIER = "svm"  # "svm" or "lda"

if CLASSIFIER == "svm":
    from svm_infer import predict
elif CLASSIFIER == "lda":
    from lda_infer import predict
else:
    raise ValueError("CLASSIFIER must be 'svm' or 'lda'")


def stats_ms(samples_us):
    """samples_us: list of microsecond durations -> dict of ms stats."""
    ms = sorted(s / 1000 for s in samples_us)
    n = len(ms)
    mean = sum(ms) / n
    median = ms[n // 2] if n % 2 else (ms[n // 2 - 1] + ms[n // 2]) / 2
    return {"min": ms[0], "mean": mean, "median": median, "max": ms[-1]}


def main():
    correct = 0
    feat_times = []
    infer_times = []
    total_times = []

    print("=" * 60)
    print(f"ON-DEVICE TEST: {CLASSIFIER.upper()}, {N_WINDOWS} windows")
    print("=" * 60)

    for i in range(N_WINDOWS):
        window = get_window(i)
        y_true = Y_TRUE[i]

        t0 = utime.ticks_us()
        feat = extract_features_pure(window)
        t1 = utime.ticks_us()
        pred_class, pred_label, scores = predict(feat)
        t2 = utime.ticks_us()

        feat_times.append(utime.ticks_diff(t1, t0))
        infer_times.append(utime.ticks_diff(t2, t1))
        total_times.append(utime.ticks_diff(t2, t0))

        if pred_class == y_true:
            correct += 1

        if i % 100 == 0:
            print(f"  [{i}/{N_WINDOWS}] pred={pred_class} true={y_true}")

    acc = correct / N_WINDOWS
    feat_s = stats_ms(feat_times)
    infer_s = stats_ms(infer_times)
    total_s = stats_ms(total_times)

    print()
    print(f"Accuracy: {correct}/{N_WINDOWS} = {acc*100:.2f}%")
    print()
    print(f"{'stage':<10} {'min':>8} {'mean':>8} {'median':>8} {'max':>8}  (ms)")
    for name, s in [("feature", feat_s), ("infer", infer_s), ("total", total_s)]:
        print(f"{name:<10} {s['min']:>8.2f} {s['mean']:>8.2f} {s['median']:>8.2f} {s['max']:>8.2f}")

    print()
    budget_ms = 100  # adjust to your real-time budget
    if total_s["mean"] > budget_ms:
        print(f"Mean total latency ({total_s['mean']:.1f}ms) exceeds the "
              f"{budget_ms}ms real-time budget.")
    else:
        print(f"Mean total latency ({total_s['mean']:.1f}ms) is within the "
              f"{budget_ms}ms real-time budget.")


if __name__ == "__main__":
    main()

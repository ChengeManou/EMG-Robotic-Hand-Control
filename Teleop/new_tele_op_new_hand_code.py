"""
main_emg_teleop.py

Real-time EMG teleoperation main loop, built on top of the existing
hand_functions / per-finger servo modules (already handling hand
initialisation and actuation).

Pipeline (all on-device):

    4x MyoWare RAW ADC1 channels @ 1kHz (GPIO1-4, matches readEMG.py)
        -> 100-sample sliding window (100ms), 75-sample step (75ms)
        -> td_features_pure.extract_td_features()  (16-dim TD)
        -> lda_infer.lda_predict_label()             (4-class LDA)
        -> pairwise switch-margin check
        -> run-length debounce (per class)
        -> matching hand_functions.* call

Classes (from lda_weights.LABELS):
    0 hand_close, 1 index_up, 2 pinch, 3 rocknroll
    (+ "rest", produced by the MAV rest gate, not by the classifier)

Requires, all in the same directory as this file:
    td_features_pure.py
    lda_infer.py
    lda_weights.py       (from export_lda_weights.py, with LABELS)
    initialise.py, pinky.py, ring.py, middle.py, index.py, thumb.py
    hand_functions.py
"""

import time
from machine import ADC, Pin

import td_features_pure
import lda_infer
import lda_weights
import hand_functions  # importing this also runs initialise.initialize_hand()

# ============================================================
# CONFIG - must match my_data_loader.py / train_lda_own.py
# ============================================================

SAMPLE_RATE_HZ = 1000
PERIOD_US = int(1_000_000 / SAMPLE_RATE_HZ)

WINDOW_SAMPLES = 150   # 100ms window
STEP_SAMPLES = 38     # 75ms step (25% overlap)

# ------------------------------------------------------------
# DEBOUNCE WINDOW (consecutive windows required, per class)
#
# Chosen from the offline confusion matrix:
#   pinch is the attractor (precision 0.67) but has recall 0.98, so
#   a long streak costs little latency -> 4.
#   hand_close (precision 1.00) and rocknroll (recall 0.78, streaks
#   get broken) -> 2.
# At 75ms/window: 2 -> 150ms, 3 -> 225ms, 4 -> 300ms.
# ------------------------------------------------------------
DEBOUNCE_WINDOWS = 3

PER_CLASS_DEBOUNCE = {
    "hand_close": 4,
    "index_up":   3,
    "pinch":      4,
    "rocknroll":  2,
    "rest":       3,
}

# ------------------------------------------------------------
# PAIRWISE MIN SWITCH MARGIN (LDA decision-score units)
#
# MIN_SWITCH_MARGIN[committed][new] = how far `new` must out-score
# `committed` in a window before a switch is considered. The score
# difference is the log-odds between those two classes.
#
# Based on the confusion matrix (row = true/committed, col = predicted):
#   - almost all errors are X -> pinch, so entering pinch is strict
#   - leaving pinch is easy (escape a false pinch lock)
#   - hand_close is never falsely predicted, so entering it is cheap
#   - rocknroll has the weakest recall, so entering it is loose
#
# If committed is rest/None, the reference is the runner-up class
# in that window. Missing pairs fall back to DEFAULT_SWITCH_MARGIN.
# Tune from the PRINT_MARGIN output.
# ------------------------------------------------------------
DEFAULT_SWITCH_MARGIN = 0.5

MIN_SWITCH_MARGIN = {
    "hand_close": {"index_up": 0.0, "pinch": 1.5, "rocknroll": 0.5},
    "index_up":   {"hand_close": 0.8, "pinch": 1.5, "rocknroll": 0.2},
    "pinch":      {"hand_close": 0.25, "index_up": 0.5, "rocknroll": 0.25},
    "rocknroll":  {"hand_close": 0.25, "index_up": 0.8, "pinch": 2.0},
}

# ------------------------------------------------------------
# REST GATE
#
# The training data has NO rest class (REMOVE_REST=True in
# my_data_loader.py strips rest samples before windowing), so the
# classifier always picks one of the 4 active gestures - even at
# rest. This gate avoids acting on any prediction unless average
# MAV across all 4 channels clears REST_MAV_THRESHOLD.
#
# HOW TO CALIBRATE:
#   1. Set PRINT_PREDICTIONS=True and PRINT_MAV=True below.
#   2. Rest your arm for a few seconds, note the avg_mav values.
#   3. Perform a clear gesture (e.g. hand_close), note avg_mav values.
#   4. Set REST_MAV_THRESHOLD roughly halfway between the two, with
#      a bit of margin above the resting noise floor.
# ------------------------------------------------------------
REST_MAV_THRESHOLD = 3.0  # PLACEHOLDER - calibrate on your own arm
PRINT_MAV = True
PRINT_MARGIN = True

PRINT_PREDICTIONS = True  # set False once everything is working

# ============================================================
# GESTURE LABEL -> hand_functions FUNCTION NAME
# ============================================================

GESTURE_TO_FUNCTION = {
    "hand_close": "close_hand",
    "index_up": "index_extension",
    "pinch": "pinch",
    "rocknroll": "rocknroll",
    "rest": "open_hand",
}

# class_id -> row index in the scores list (robust to non-contiguous ids)
CLASS_ROW = {cid: row for row, cid in enumerate(lda_weights.CLASS_IDS)}
# label -> row index, and back
LABEL_TO_ROW = {name: CLASS_ROW[cid] for cid, name in lda_weights.LABELS.items()}
ROW_TO_LABEL = {row: name for name, row in LABEL_TO_ROW.items()}


def resolve_functions(mapping):
    """Turn {label: function_name} into {label: callable}, failing fast."""
    resolved = {}
    missing = []
    for label, name in mapping.items():
        fn = getattr(hand_functions, name, None)
        if fn is None:
            missing.append((label, name))
        else:
            resolved[label] = fn
    if missing:
        print("MISSING in hand_functions:", missing)
        print("Available:", [n for n in dir(hand_functions) if not n.startswith("_")])
        raise SystemExit
    return resolved


# ============================================================
# ADC SETUP - same pins/config as readEMG.py
# ============================================================

emg1 = ADC(Pin(1))
emg1.atten(ADC.ATTN_11DB)
emg1.width(ADC.WIDTH_12BIT)

emg2 = ADC(Pin(2))
emg2.atten(ADC.ATTN_11DB)
emg2.width(ADC.WIDTH_12BIT)

emg3 = ADC(Pin(3))
emg3.atten(ADC.ATTN_11DB)
emg3.width(ADC.WIDTH_12BIT)

emg4 = ADC(Pin(4))
emg4.atten(ADC.ATTN_11DB)
emg4.width(ADC.WIDTH_12BIT)


def read_sample():
    """Returns one 4-channel sample as a plain list [ch1,ch2,ch3,ch4]."""
    return [emg1.read(), emg2.read(), emg3.read(), emg4.read()]


# ============================================================
# PAIRWISE SWITCH MARGIN
# ============================================================

def runner_up_row(scores, exclude_row):
    """Row index of the highest-scoring class other than exclude_row."""
    best = None
    for i in range(len(scores)):
        if i == exclude_row:
            continue
        if best is None or scores[i] > scores[best]:
            best = i
    return best


def margin_needed(from_label, to_label):
    """Threshold for switching from_label -> to_label."""
    row = MIN_SWITCH_MARGIN.get(from_label)
    if row is None:
        return DEFAULT_SWITCH_MARGIN
    return row.get(to_label, DEFAULT_SWITCH_MARGIN)


# ============================================================
# DEBOUNCE
# ============================================================

class Debouncer:
    """
    Run-length debounce: only reports a new gesture once the same
    prediction has held for the required number of consecutive
    windows. `per_class_required` sets a streak length per label.
    """

    def __init__(self, required, per_class_required=None):
        self.required = required
        self.per_class_required = per_class_required or {}
        self.last_label = None
        self.streak = 0
        self.committed_label = None

    def update(self, label):
        if label == self.last_label:
            self.streak += 1
        else:
            self.last_label = label
            self.streak = 1

        needed = self.per_class_required.get(label, self.required)
        if self.streak >= needed:
            self.committed_label = label

        return self.committed_label


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    print()
    print("=" * 60)
    print("EMG TELEOP - LDA 4-class, real-time")
    print("=" * 60)

    gesture_fns = resolve_functions(GESTURE_TO_FUNCTION)

    debouncer = Debouncer(DEBOUNCE_WINDOWS, PER_CLASS_DEBOUNCE)

    # Rolling buffer of the most recent WINDOW_SAMPLES samples
    buffer = []

    step_counter = 0
    next_sample_time = time.ticks_us()
    last_executed = None  # last gesture actually sent to the hand

    print("Warming up buffer...")

    while True:

        # ------------------------------------------------
        # Pace sampling to SAMPLE_RATE_HZ
        # ------------------------------------------------
        now = time.ticks_us()
        wait = time.ticks_diff(next_sample_time, now)
        if wait > 0:
            time.sleep_us(wait)
        next_sample_time = time.ticks_add(next_sample_time, PERIOD_US)

        # ------------------------------------------------
        # Read one sample, push into rolling buffer
        # ------------------------------------------------
        sample = read_sample()
        buffer.append(sample)
        if len(buffer) > WINDOW_SAMPLES:
            buffer.pop(0)

        step_counter += 1

        # ------------------------------------------------
        # Every STEP_SAMPLES, classify if the buffer is full
        # ------------------------------------------------
        if step_counter >= STEP_SAMPLES and len(buffer) == WINDOW_SAMPLES:
            step_counter = 0

            features = td_features_pure.extract_td_features(buffer)

            # Rest gate: MAV features are at indices 0,4,8,12
            # (one per channel, in channel order - see
            # td_features_pure.extract_td_features).
            avg_mav = (features[0] + features[4] + features[8] + features[12]) / 4.0

            if PRINT_MAV:
                print("avg_mav:", avg_mav)

            margin = None
            ref_label = None
            if avg_mav < REST_MAV_THRESHOLD:
                label = "rest"
            else:
                label, class_id, scores = lda_infer.lda_predict_label(features)

                # Pairwise switch-margin check, only when the prediction
                # differs from what is currently committed.
                current = debouncer.committed_label
                if label != current:
                    new_row = CLASS_ROW[class_id]
                    if current in LABEL_TO_ROW:
                        ref_label = current
                        ref_row = LABEL_TO_ROW[current]
                    else:
                        # committed is rest/None: compare to runner-up
                        ref_row = runner_up_row(scores, new_row)
                        ref_label = ROW_TO_LABEL[ref_row]

                    margin = scores[new_row] - scores[ref_row]
                    needed = margin_needed(ref_label, label)
                    if margin < needed:
                        # Not convincing enough: hold the current state.
                        # (At startup, hold means "rest".)
                        label = current if current is not None else "rest"

            committed = debouncer.update(label)

            # Only drive the hand when the committed gesture changes,
            # so blocking servo calls don't stall the sampling loop.
            if committed is not None and committed != last_executed:
                fn = gesture_fns.get(committed)
                if fn is not None:
                    fn()
                last_executed = committed

            if PRINT_PREDICTIONS:
                if PRINT_MARGIN and margin is not None:
                    print("raw:", label, " vs:", ref_label, " margin:", margin,
                          " streak:", debouncer.streak,
                          " committed:", committed)
                else:
                    print("raw:", label, " streak:", debouncer.streak,
                          " committed:", committed)


if __name__ == "__main__":
    main()
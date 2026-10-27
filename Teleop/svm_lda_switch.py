"""
main_emg_teleop_new.py

Real-time EMG teleoperation using a 5-class LDA model.
Uses the consolidated hand.py (hand.do / hand.initialize).

Gesture set:
    0 -> hand_close
    1 -> hand_open
    2 -> pinch
    3 -> rocnroll
    4 -> thumb_up
"""

import time
from machine import ADC, Pin

import td_features_pure
import hand

# ============================================================
# MODEL SELECTION  ("lda" or "svm")
# ============================================================

MODEL = "lda"

# Import only the chosen model so the other's weights don't use RAM.
if MODEL == "lda":
    import lda_infer

    def predict_label(features):
        return lda_infer.lda_predict_label(features)   # (label, class_id, scores)

elif MODEL == "svm":
    import svm_infer

    def predict_label(features):
        class_id, label, scores = svm_infer.predict(features)
        return label, class_id, scores                 # same order as LDA

else:
    raise ValueError("MODEL must be 'lda' or 'svm'")

# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE_HZ = 1000
PERIOD_US = int(1_000_000 / SAMPLE_RATE_HZ)

WINDOW_SAMPLES = 150
STEP_SAMPLES = 38

DEBOUNCE_WINDOWS = 3

INIT_ON_START = True
INIT_DELAY = 0.5   # seconds per finger (old default was 2)


# ============================================================
# PER-CLASS DEBOUNCE
# ============================================================

PER_CLASS_DEBOUNCE = {
    "hand_close": 3,
    "hand_open": 3,
    "pinch": 3,
    "rocnroll": 3,
    "thumb_up": 3,
}


# ============================================================
# SWITCH MARGIN
# ============================================================

MIN_SWITCH_MARGIN_BY_MODEL = {
    "lda": 3.0,
    "svm": 0.3,    # PLACEHOLDER - calibrate on your arm
}
MIN_SWITCH_MARGIN = MIN_SWITCH_MARGIN_BY_MODEL[MODEL]

PER_CLASS_MARGIN_BY_MODEL = {
    "lda": {
        "hand_close": 3.0,
        "hand_open": 3.0,
        "pinch": 0.5,
        "rocnroll": 3.0,
        "thumb_up": 3.0,
    },
    "svm": {
        "hand_close": MIN_SWITCH_MARGIN,
        "hand_open": MIN_SWITCH_MARGIN,
        "pinch": MIN_SWITCH_MARGIN,
        "rocnroll": MIN_SWITCH_MARGIN,
        "thumb_up": MIN_SWITCH_MARGIN,
    },
}
PER_CLASS_MARGIN = PER_CLASS_MARGIN_BY_MODEL[MODEL]


# ============================================================
# REST GATE
# ============================================================

REST_MAV_THRESHOLD = 3.0

PRINT_MAV = True
PRINT_PREDICTIONS = True


# ============================================================
# GESTURE LABEL -> POSE NAME (in hand.POSES)
# ============================================================

GESTURE_TO_POSE = {
    "hand_close": "close",
    "hand_open": "open",
    "pinch": "pinch",
    "rocnroll": "rocknroll",
    "thumb_up": "thumb_extension",

    # Rest is not a trained class.
    # It is created by the MAV threshold gate.
    "rest": "open",
}

for _name in GESTURE_TO_POSE.values():
    if _name not in hand.POSES:
        print("WARNING: pose '" + _name + "' not in hand.POSES")


# ============================================================
# ADC SETUP
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
    """Returns one 4-channel sample."""
    return [
        emg1.read(),
        emg2.read(),
        emg3.read(),
        emg4.read()
    ]


# ============================================================
# DEBOUNCER
# ============================================================

class Debouncer:

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

        needed = self.per_class_required.get(
            label,
            self.required
        )

        if self.streak >= needed:
            self.committed_label = label

        return self.committed_label


# ============================================================
# SCORE MARGIN
# ============================================================

def top_margin(scores):
    """
    Difference between the highest LDA score and
    the second-highest score.
    """

    ranked = sorted(scores, reverse=True)

    if len(ranked) < 2:
        return ranked[0] if ranked else 0.0

    return ranked[0] - ranked[1]


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    print()
    print("=" * 60)
    print("EMG TELEOP - " + MODEL.upper() + " 5-class, real-time")
    print("=" * 60)

    print("Classes:")
    print("  0 = hand_close")
    print("  1 = hand_open")
    print("  2 = pinch")
    print("  3 = rocnroll")
    print("  4 = thumb_up")
    print("=" * 60)

    if INIT_ON_START:
        hand.initialize(INIT_DELAY)

    last_executed = None

    debouncer = Debouncer(
        DEBOUNCE_WINDOWS,
        PER_CLASS_DEBOUNCE
    )

    # Rolling buffer
    buffer = []

    step_counter = 0

    next_sample_time = time.ticks_us()

    print("Warming up buffer...")

    while True:

        # ----------------------------------------------------
        # Pace sampling at 1 kHz
        # ----------------------------------------------------

        now = time.ticks_us()

        wait = time.ticks_diff(
            next_sample_time,
            now
        )

        if wait > 0:
            time.sleep_us(wait)

        next_sample_time = time.ticks_add(
            next_sample_time,
            PERIOD_US
        )


        # ----------------------------------------------------
        # Read sample
        # ----------------------------------------------------

        sample = read_sample()

        buffer.append(sample)

        if len(buffer) > WINDOW_SAMPLES:
            buffer.pop(0)

        step_counter += 1


        # ----------------------------------------------------
        # Classify every STEP_SAMPLES
        # ----------------------------------------------------

        if (
            step_counter >= STEP_SAMPLES
            and len(buffer) == WINDOW_SAMPLES
        ):

            step_counter = 0

            # Extract 16 TD features
            features = td_features_pure.extract_td_features(
                buffer
            )


            # ------------------------------------------------
            # REST GATE
            # ------------------------------------------------

            # MAV features:
            #   CH1 -> index 0
            #   CH2 -> index 4
            #   CH3 -> index 8
            #   CH4 -> index 12

            avg_mav = (
                features[0]
                + features[4]
                + features[8]
                + features[12]
            ) / 4.0

            if PRINT_MAV:
                print("avg_mav:", avg_mav)


            # ------------------------------------------------
            # REST
            # ------------------------------------------------

            if avg_mav < REST_MAV_THRESHOLD:

                label = "rest"

            else:

                # --------------------------------------------
                # LDA prediction
                # --------------------------------------------

                label, class_id, scores = predict_label(features)


                # --------------------------------------------
                # Safety check
                # --------------------------------------------

                if label not in GESTURE_TO_POSE:

                    if PRINT_PREDICTIONS:
                        print(
                            "raw:",
                            label,
                            "(ignored - unmapped class)"
                        )

                    continue


                # --------------------------------------------
                # Switch-margin gate
                # --------------------------------------------

                held = debouncer.committed_label

                if (
                    held is not None
                    and label != held
                ):

                    margin = top_margin(scores)

                    needed_margin = PER_CLASS_MARGIN.get(
                        label,
                        MIN_SWITCH_MARGIN
                    )

                    if margin < needed_margin:

                        if PRINT_PREDICTIONS:
                            print(
                                "raw:",
                                label,
                                "margin:",
                                margin,
                                "(below switch threshold, held:",
                                held,
                                ")"
                            )

                        continue


            # ------------------------------------------------
            # Debounce
            # ------------------------------------------------

            committed = debouncer.update(label)


            # ------------------------------------------------
            # Execute hand movement (only on change)
            # ------------------------------------------------

            if committed is not None and committed != last_executed:
                hand.do(GESTURE_TO_POSE[committed])
                last_executed = committed


            # ------------------------------------------------
            # Debug output
            # ------------------------------------------------

            if PRINT_PREDICTIONS:

                print(
                    "raw:",
                    label,
                    "streak:",
                    debouncer.streak,
                    "committed:",
                    committed
                )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
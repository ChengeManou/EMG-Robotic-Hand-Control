"""
plot_emg.py

Plot recorded 4-channel EMG data and detect active regions.

Usage:
    python plot_emg.py data/manou_cycle1.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

FS = 1000  # Hz

CHANNELS = [
    "ch1",
    "ch2",
    "ch3",
    "ch4"
]

# Envelope smoothing
ENVELOPE_WINDOW = 50       # 50 ms at 1 kHz

# Adaptive double threshold
HIGH_THRESHOLD_MAD = 4.0
LOW_THRESHOLD_MAD = 2.0

# Remove very short detections
MIN_ACTIVITY_MS = 100

# Join small gaps
MAX_GAP_MS = 100


# ============================================================
# MOVING AVERAGE
# ============================================================

def moving_average(x, window):

    if window <= 1:
        return x.copy()

    kernel = np.ones(window) / window

    return np.convolve(
        x,
        kernel,
        mode="same"
    )


# ============================================================
# ACTIVITY DETECTION
# ============================================================

def detect_activity(emg, fs=1000):

    # --------------------------------------------------------
    # Remove DC offset from each channel
    # --------------------------------------------------------

    centered = emg - np.median(
        emg,
        axis=0,
        keepdims=True
    )

    # --------------------------------------------------------
    # Rectify
    # --------------------------------------------------------

    rectified = np.abs(centered)

    # --------------------------------------------------------
    # Combine the four channels
    # --------------------------------------------------------

    activity_signal = np.sqrt(
        np.mean(
            rectified ** 2,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Smooth envelope
    # --------------------------------------------------------

    envelope = moving_average(
        activity_signal,
        ENVELOPE_WINDOW
    )

    # --------------------------------------------------------
    # Robust statistics
    # --------------------------------------------------------

    median_env = np.median(envelope)

    mad = np.median(
        np.abs(
            envelope - median_env
        )
    )

    # Prevent zero MAD
    if mad < 1e-9:
        mad = np.std(envelope)

    if mad < 1e-9:
        return (
            np.zeros(
                len(envelope),
                dtype=bool
            ),
            envelope,
            median_env,
            mad
        )

    # --------------------------------------------------------
    # Adaptive thresholds
    # --------------------------------------------------------

    high_threshold = (
        median_env +
        HIGH_THRESHOLD_MAD * mad
    )

    low_threshold = (
        median_env +
        LOW_THRESHOLD_MAD * mad
    )

    print()
    print("======================================")
    print(" ACTIVITY DETECTION")
    print("======================================")

    print(
        f"Envelope median : {median_env:.3f}"
    )

    print(
        f"MAD             : {mad:.3f}"
    )

    print(
        f"High threshold  : {high_threshold:.3f}"
    )

    print(
        f"Low threshold   : {low_threshold:.3f}"
    )

    # --------------------------------------------------------
    # Double-threshold hysteresis
    # --------------------------------------------------------

    active = np.zeros(
        len(envelope),
        dtype=bool
    )

    in_activity = False

    for i in range(len(envelope)):

        if not in_activity:

            if envelope[i] >= high_threshold:

                in_activity = True
                active[i] = True

        else:

            if envelope[i] >= low_threshold:

                active[i] = True

            else:

                in_activity = False

    # --------------------------------------------------------
    # Fill short gaps
    # --------------------------------------------------------

    max_gap_samples = int(
        MAX_GAP_MS * fs / 1000
    )

    i = 0

    while i < len(active):

        if active[i]:
            i += 1
            continue

        start = i

        while (
            i < len(active)
            and not active[i]
        ):
            i += 1

        end = i

        gap_length = end - start

        if (
            start > 0
            and end < len(active)
            and gap_length <= max_gap_samples
        ):
            active[start:end] = True

    # --------------------------------------------------------
    # Remove short activity regions
    # --------------------------------------------------------

    min_activity_samples = int(
        MIN_ACTIVITY_MS * fs / 1000
    )

    i = 0

    while i < len(active):

        if not active[i]:
            i += 1
            continue

        start = i

        while (
            i < len(active)
            and active[i]
        ):
            i += 1

        end = i

        duration = end - start

        if duration < min_activity_samples:

            active[
                start:end
            ] = False

    return (
        active,
        envelope,
        median_env,
        mad
    )


# ============================================================
# GET ACTIVITY REGIONS
# ============================================================

def get_activity_regions(
    active,
    time_s
):

    regions = []

    i = 0

    while i < len(active):

        if not active[i]:
            i += 1
            continue

        start = i

        while (
            i < len(active)
            and active[i]
        ):
            i += 1

        end = i - 1

        regions.append(
            (
                time_s[start],
                time_s[end]
            )
        )

    return regions


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python plot_emg.py <csv_file>"
        )

        return

    filename = sys.argv[1]

    # ========================================================
    # LOAD CSV
    # ========================================================

    df = pd.read_csv(filename)

    required = [
        "sample_idx",
        "timestamp_ms",
        "ch1",
        "ch2",
        "ch3",
        "ch4"
    ]

    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"Missing required column: {col}"
            )

    # ========================================================
    # CLEAN NUMERIC DATA
    # ========================================================

    for ch in CHANNELS:

        df[ch] = pd.to_numeric(
            df[ch],
            errors="coerce"
        )

    df["sample_idx"] = pd.to_numeric(
        df["sample_idx"],
        errors="coerce"
    )

    df["timestamp_ms"] = pd.to_numeric(
        df["timestamp_ms"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "sample_idx",
            "timestamp_ms",
            *CHANNELS
        ]
    )

    # ========================================================
    # TIME AXIS
    # ========================================================
    #
    # IMPORTANT:
    #
    # sample_idx is safer than timestamp_ms here because
    # the ESP32 microsecond counter can wrap after ~71.6 min.
    #
    # At 1000 Hz:
    #
    #       time = sample_idx / 1000
    #
    # ========================================================

    time_s = (
        df["sample_idx"].to_numpy()
        / FS
    )

    # ========================================================
    # EMG
    # ========================================================

    emg = df[
        CHANNELS
    ].to_numpy(
        dtype=np.float64
    )

    # ========================================================
    # RECORDING INFORMATION
    # ========================================================

    duration = time_s[-1]

    print()
    print("======================================")
    print(" EMG RECORDING")
    print("======================================")

    print(
        f"File       : {filename}"
    )

    print(
        f"Samples    : {len(df)}"
    )

    print(
        f"Duration   : {duration:.2f} s"
    )

    print(
        f"Duration   : {duration / 60:.2f} min"
    )

    # ========================================================
    # CHECK SAMPLING TIMING
    # ========================================================

    timestamps = (
        df["timestamp_ms"]
        .to_numpy(
            dtype=np.float64
        )
    )

    timestamp_diff = np.diff(
        timestamps
    )

    # Only look at positive differences.
    # Negative differences can occur because of timer rollover.

    positive_diffs = timestamp_diff[
        timestamp_diff > 0
    ]

    if len(positive_diffs) > 0:

        print()
        print("Timestamp statistics:")

        print(
            f"Median sample interval : "
            f"{np.median(positive_diffs):.1f} us"
        )

        print(
            f"Mean sample interval   : "
            f"{np.mean(positive_diffs):.1f} us"
        )

    # ========================================================
    # ACTIVITY DETECTION
    # ========================================================

    (
        active,
        envelope,
        median_env,
        mad
    ) = detect_activity(
        emg,
        fs=FS
    )

    regions = get_activity_regions(
        active,
        time_s
    )

    # ========================================================
    # PRINT ACTIVITY
    # ========================================================

    print()
    print("======================================")
    print(" DETECTED ACTIVITY")
    print("======================================")

    print(
        f"Number of regions: {len(regions)}"
    )

    for i, (start, end) in enumerate(
        regions,
        1
    ):

        print(
            f"{i:02d}: "
            f"{start:.2f} s -> "
            f"{end:.2f} s "
            f"({end - start:.2f} s)"
        )

    print()

    print(
        f"Total active time: "
        f"{np.sum(active) / FS:.2f} s"
    )

    # ========================================================
    # PLOT
    # ========================================================

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 10),
        sharex=True
    )

    for i, ch in enumerate(
        CHANNELS
    ):

        axes[i].plot(
            time_s,
            emg[:, i],
            linewidth=0.7
        )

        axes[i].set_ylabel(
            ch.upper()
        )

        axes[i].grid(
            True,
            alpha=0.3
        )

        # ----------------------------------------------------
        # Shade detected activity
        # ----------------------------------------------------

        for start, end in regions:

            axes[i].axvspan(
                start,
                end,
                alpha=0.18
            )

    # ========================================================
    # X AXIS
    # ========================================================

    axes[-1].set_xlabel(
        "Time (s)"
    )

    axes[-1].set_xlim(
        0,
        duration
    )

    # ========================================================
    # TITLE
    # ========================================================

    fig.suptitle(
        f"4-Channel Raw EMG with Activity Detection "
        f"— {duration:.1f} s",
        fontsize=14
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
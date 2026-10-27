"""Plot four EMG channels and windowed Hudgins TD features from a recording.

Dependencies: pip install numpy pandas matplotlib
Change CSV_PATH, GESTURE, REP and output settings as needed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path('manou_cycle1(1).csv')
FS = 1000                 # Hz
WINDOW = 150              # samples = 150 ms
STEP = 38                 # samples = 38 ms (~75% overlap)
THRESHOLD = 10            # ADC counts for ZC and SSC
GESTURE = 'index_up'      # Set to None for the full recording
REP = 1                   # Set to None for all repetitions
SAVE = True

CHANNELS = ['ch1', 'ch2', 'ch3', 'ch4']
FEATURES = ['MAV', 'WL', 'ZC', 'SSC']


def td_features(window):
    """Return MAV, WL, ZC, SSC after subtracting window mean."""
    x = np.asarray(window, dtype=float)
    x = x - x.mean()

    mav = np.mean(np.abs(x))
    wl = np.sum(np.abs(np.diff(x)))
    zc = np.count_nonzero(
        (x[:-1] * x[1:] < 0) & (np.abs(np.diff(x)) >= THRESHOLD)
    )
    left = x[1:-1] - x[:-2]
    right = x[1:-1] - x[2:]
    ssc = np.count_nonzero(
        (left * right > 0)
        & ((np.abs(left) >= THRESHOLD) | (np.abs(right) >= THRESHOLD))
    )
    return [mav, wl, zc, ssc]


def main():
    df = pd.read_csv(CSV_PATH)
    if GESTURE is not None:
        df = df.loc[df['gesture'] == GESTURE]
    if REP is not None:
        df = df.loc[df['rep'] == REP]
    df = df.reset_index(drop=True)
    if len(df) < WINDOW:
        raise ValueError('Selection has fewer than 150 samples.')

    # Use sample number for a uniform 1 kHz time axis.
    t = np.arange(len(df)) / FS
    data = df[CHANNELS].to_numpy(dtype=float)
    starts = np.arange(0, len(df) - WINDOW + 1, STEP)
    feature_times = (starts + (WINDOW - 1) / 2) / FS
    values = np.empty((len(starts), len(CHANNELS), len(FEATURES)))
    for j, start in enumerate(starts):
        for ch in range(len(CHANNELS)):
            values[j, ch] = td_features(data[start:start + WINDOW, ch])

    fig1, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    for ch, ax in enumerate(axes):
        ax.plot(t, data[:, ch], lw=0.65)
        ax.set_ylabel(f'CH{ch + 1}\nADC counts')
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel('Time (s)')
    fig1.suptitle(f'Raw EMG — {GESTURE or "all gestures"}, repetition {REP or "all"}')
    fig1.tight_layout()

    fig2, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True)
    for ch in range(4):
        for k, feature in enumerate(FEATURES):
            ax = axes[ch, k]
            ax.plot(feature_times, values[:, ch, k], lw=1)
            if ch == 0:
                ax.set_title(feature)
            if k == 0:
                ax.set_ylabel(f'CH{ch + 1}')
            if ch == 3:
                ax.set_xlabel('Time (s)')
            ax.grid(alpha=0.25)
    fig2.suptitle(f'TD features — {WINDOW} ms window, {STEP} ms step, threshold {THRESHOLD}')
    fig2.tight_layout()

    if SAVE:
        fig1.savefig('emg_channels.png', dpi=300, bbox_inches='tight')
        fig2.savefig('emg_td_features.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()

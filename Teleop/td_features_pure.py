ZC_THRESHOLD = 10
SSC_THRESHOLD = 10


def _channel_values(window, ch):
    """Extract one channel's samples from a window as a plain list."""
    return [window[i][ch] for i in range(len(window))]


def _mean(values):
    return sum(values) / len(values)


def _mav(x):
    return sum(abs(v) for v in x) / len(x)


def _waveform_length(x):
    total = 0.0
    for i in range(1, len(x)):
        total += abs(x[i] - x[i - 1])
    return total


def _zero_crossings(x, threshold=ZC_THRESHOLD):
    count = 0
    for i in range(len(x) - 1):
        x1 = x[i]
        x2 = x[i + 1]
        sign_change = (x1 >= 0 and x2 < 0) or (x1 < 0 and x2 >= 0)
        if sign_change and abs(x1 - x2) >= threshold:
            count += 1
    return count


def _slope_sign_changes(x, threshold=SSC_THRESHOLD):
    count = 0
    for i in range(1, len(x) - 1):
        d1 = x[i] - x[i - 1]
        d2 = x[i + 1] - x[i]
        change = (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)
        if change and abs(x[i + 1] - x[i - 1]) >= threshold:
            count += 1
    return count


def extract_td_features(window):
    """
    window: sequence of samples, each sample a 4-element sequence
            [ch1, ch2, ch3, ch4]. Length is expected to be 100
            (100ms at 1kHz), but any length works.

    Returns a flat list of 16 floats: [MAV, ZC, SSC, WL] x 4 channels,
    in the same order as train_lda_own.py's extract_td_features().
    """
    n_channels = len(window[0])
    features = []

    for ch in range(n_channels):
        x = _channel_values(window, ch)

        # Mean removal (matches: x = x - np.mean(x))
        mean_x = _mean(x)
        x = [v - mean_x for v in x]

        features.append(_mav(x))
        features.append(_zero_crossings(x))
        features.append(_slope_sign_changes(x))
        features.append(_waveform_length(x))

    return features

"""Frame-verified video movement latency measurement.

Install:  pip install opencv-python numpy matplotlib
Run:      python video_latency_verified.py
          python video_latency_verified.py "videos/my trial.mp4"
          python video_latency_verified.py --scan-only "videos/my trial.mp4"

Place recordings in videos/ or pass paths explicitly. Select a USER-hand ROI and
then a PROSTHETIC-hand ROI. Boxes and manually confirmed frame pairs are saved
per video in latency_outputs/ and can be reopened.

INTERACTIVE KEYS (click video window before typing):
  a / d      previous / next frame (precise manual onset refinement)
  z / x      10 frames backward / forward
  j / l      one second backward / forward
  SPACE      play / pause (playback is for navigation, not precise timing)
  n / b      next / previous automatic USER onset suggestion
  m / k      next / previous automatic PROSTHETIC onset suggestion
  u          mark first visible USER movement frame
  h          mark first visible PROSTHETIC movement frame; save pair
  c          cancel pending user onset
  Backspace  undo last saved pair
  q or Esc   finish video; save reviewed pairs and diagnostic plot

IMPORTANT: Motion thresholds produce NAVIGATION HINTS ONLY. No suggested event
enters latency_results.csv unless you manually mark and verify both frames.
Choose the first frame where movement is visible, not the largest motion peak.
Compare the 'previous' and 'current' cropped frames to check movement onsets.
Match the same commanded gesture and reject ambiguous camera or hand motion.
The result is visible hand-to-prosthetic movement latency, NOT EMG-to-actuation
latency or a guaranteed physiological reaction time.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # diagnostic plots must not block OpenCV windows
import matplotlib.pyplot as plt
import numpy as np

# Automatic SUGGESTIONS only; measurement comes from manually marked frames.
K = 3.0
SMOOTH = 3
MIN_GAP_S = 0.45
MIN_ACTIVE_FRAMES = 2
MAX_LAG_S = 1.5  # manual pairs longer than this require increasing the limit
OUT_DIR = Path("latency_outputs")
WIN = "Latency verification - click here for keyboard controls"


def clips():
    return sorted([*Path("videos").glob("*.mp4"),
                   *Path("videos").glob("*.mov"),
                   *Path("videos").glob("*.avi")])


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width == 0 or height == 0:
        cap.release()
        raise RuntimeError(f"Missing FPS/dimensions: {path}")
    return cap, float(fps), width, height


def load_or_pick_rois(path, cap, width, height):
    roi_file = OUT_DIR / f"{path.stem}_rois.json"
    if roi_file.exists():
        data = json.loads(roi_file.read_text())
        if data.get("size") == [width, height]:
            print(f"Using saved ROIs: {roi_file} (delete to select again)")
            return [tuple(map(int, r)) for r in data["rois"]]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first video frame")
    print("Select the moving part of YOUR hand/arm, then press ENTER.")
    u = tuple(map(int, cv2.selectROI("1. USER hand/arm", frame, False, False)))
    cv2.destroyWindow("1. USER hand/arm")
    print("Select the moving fingers of the PROSTHETIC, then press ENTER.")
    h = tuple(map(int, cv2.selectROI("2. PROSTHETIC fingers", frame, False, False)))
    cv2.destroyWindow("2. PROSTHETIC fingers")
    for x, y, w, ht in (u, h):
        if not (0 <= x < width and 0 <= y < height and w > 2 and ht > 2
                and x + w <= width and y + ht <= height):
            raise RuntimeError("ROI selection cancelled/invalid")
    roi_file.write_text(json.dumps({"size": [width, height], "rois": [u, h]}, indent=2))
    return [u, h]


def scan_signals(path, rois, fps):
    cap = cv2.VideoCapture(str(path))
    signals = [[], []]
    pts = []
    old = [None, None]
    frame_number = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i, (x, y, w, h) in enumerate(rois):
            crop = cv2.GaussianBlur(gray[y:y+h, x:x+w], (5, 5), 0)
            val = 0.0 if old[i] is None else float(cv2.absdiff(crop, old[i]).mean())
            signals[i].append(val)
            old[i] = crop
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        frame_number += 1
    cap.release()
    if frame_number < 3:
        raise RuntimeError("Video too short")

    # Prefer video presentation timestamps if readable and strictly increasing.
    ts = np.asarray(pts, dtype=float)
    if (not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0)
            or ts[-1] < 0.8 * (frame_number - 1) / fps):
        print("No reliable presentation timestamps; using frame_number / FPS.")
        ts = np.arange(frame_number, dtype=float) / fps
    return np.asarray(signals, dtype=float), ts


def onset_suggestions(signal, times):
    smooth = np.convolve(signal, np.ones(SMOOTH) / SMOOTH, mode="same")
    med = np.median(smooth)
    mad = 1.4826 * np.median(np.abs(smooth - med))
    threshold = max(float(med + K * mad), 0.03)
    above = smooth > threshold
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    ends = np.flatnonzero(above & ~np.r_[above[1:], False]) + 1
    events = []
    for start, end in zip(starts, ends):
        if end - start < MIN_ACTIVE_FRAMES:
            continue
        if events and times[start] - times[events[-1]] < MIN_GAP_S:
            continue
        events.append(int(start))
    return events, smooth, threshold


def frame_at(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {index}")
    return frame


def shrink_to_fit(img, w, h):
    """Letterbox a crop to a consistent size without stretching it."""
    H, W = img.shape[:2]
    scale = min(w / W, h / H)
    resized = cv2.resize(img, (max(1, round(W*scale)), max(1, round(H*scale))))
    canvas = np.full((h, w, 3), 22, dtype=np.uint8)
    yy = (h - resized.shape[0]) // 2
    xx = (w - resized.shape[1]) // 2
    canvas[yy: yy+resized.shape[0], xx: xx+resized.shape[1]] = resized
    return canvas


def put(img, text, x, y, color=(240, 240, 240), size=0.54):
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 1, cv2.LINE_AA)


def preview(current, prev, rois, frame_idx, times, sig, limits,
            user_candidates, hand_candidates, pairs, pending, warning):
    H, W = current.shape[:2]
    left_h = min(740, H)
    left_w = round(W * left_h / H)
    view = current.copy()
    colors = [(80, 245, 80), (40, 190, 255)]
    for i, (x, y, w, h) in enumerate(rois):
        cv2.rectangle(view, (x, y), (x+w, y+h), colors[i], 3)
        cv2.putText(view, "USER" if i == 0 else "PROSTHETIC",
                    (x, max(22, y-8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colors[i], 2, cv2.LINE_AA)
    view = cv2.resize(view, (left_w, left_h))

    panel_w = 470
    canvas = np.full((max(left_h, 720), left_w + panel_w, 3), 28, dtype=np.uint8)
    canvas[:left_h, :left_w] = view
    ox = left_w + 12

    put(canvas, f"Frame {frame_idx} | {times[frame_idx]:.3f} s", ox, 24, size=0.64)
    put(canvas, "PREVIOUS vs CURRENT frame (zoom)", ox, 51)
    crop_w, crop_h = 215, 175
    for i, (x, y, w, h) in enumerate(rois):
        row_y = 77 + i * 235
        put(canvas, "USER" if i == 0 else "PROSTHETIC", ox, row_y, colors[i])
        a = shrink_to_fit(prev[y:y+h, x:x+w], crop_w, crop_h)
        b = shrink_to_fit(current[y:y+h, x:x+w], crop_w, crop_h)
        yy = row_y + 10
        canvas[yy: yy+crop_h, ox: ox+crop_w] = a
        canvas[yy: yy+crop_h, ox+230: ox+230+crop_w] = b
        put(canvas, "prev", ox+6, yy+crop_h+17)
        put(canvas, "current", ox+236, yy+crop_h+17)
        put(canvas, f"motion={sig[i, frame_idx]:.2f} / thresh={limits[i]:.2f}",
            ox, yy+crop_h+37)

    yy = 569
    put(canvas, f"Verified pairs: {len(pairs)}", ox, yy)
    if pending is None:
        put(canvas, "Pending user onset: none", ox, yy+25)
    else:
        put(canvas, f"Pending USER: frame {pending} ({times[pending]:.3f}s)",
            ox, yy+25, (80, 245, 80))
    put(canvas, "u: mark USER     h: mark PROSTHETIC", ox, yy+53)
    put(canvas, "a/d: 1 frame    z/x: 10 frames", ox, yy+74)
    put(canvas, "j/l: 1 sec       Space: play/pause", ox, yy+95)
    put(canvas, "n/b: user hint   m/k: hand hint", ox, yy+116)
    put(canvas, "Backspace: undo   c: clear   q: save", ox, yy+137)
    if warning:
        # Warning is also printed to the console if there is insufficient space.
        cv2.rectangle(canvas, (10, canvas.shape[0]-42),
                      (canvas.shape[1]-10, canvas.shape[0]-5), (25, 25, 115), -1)
        put(canvas, warning[:110], 20, canvas.shape[0]-18, (255, 255, 255))
    return canvas


def saved_pairs(path, times):
    fn = OUT_DIR / f"{path.stem}_verified.csv"
    out = []
    if fn.exists():
        with fn.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u, h = int(row["user_frame"]), int(row["hand_frame"])
                if 0 <= u < h < len(times):
                    out.append((u, h))
        print(f"Loaded {len(out)} previously reviewed pairs for {path.name}")
    return out


def save_pairs(path, pairs, times, fps):
    fn = OUT_DIR / f"{path.stem}_verified.csv"
    with fn.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "rep", "user_frame", "user_time_s", "hand_frame",
                         "hand_time_s", "latency_ms", "fps", "manually_verified"])
        for i, (u, h) in enumerate(sorted(pairs), 1):
            writer.writerow([str(path), i, u, f"{times[u]:.6f}", h,
                             f"{times[h]:.6f}", f"{1000*(times[h]-times[u]):.2f}",
                             f"{fps:.6f}", "yes"])
    return fn


def plot_diagnostic(path, times, smooth, thresholds, suggestions, pairs):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    for i, (ax, label) in enumerate(zip(axs, ["USER", "PROSTHETIC"])):
        ax.plot(times, smooth[i], linewidth=0.9)
        ax.axhline(thresholds[i], ls="--", color="tab:red", label="Auto threshold")
        for idx in suggestions[i]:
            ax.axvline(times[idx], ls=":", color="0.65", alpha=0.35)
        for u, h in pairs:
            ax.axvline(times[u if i == 0 else h], color="tab:green", alpha=0.8)
        ax.set_ylabel(label + " motion")
    axs[0].set_title("Gray dotted: onset suggestions. Green: manually verified frames.")
    axs[1].set_xlabel("Video time (s)")
    fig.tight_layout()
    dst = OUT_DIR / f"{path.stem}_diagnostic.png"
    fig.savefig(dst, dpi=140)
    plt.close(fig)
    return dst


def next_hint(candidates, current, forward):
    if forward:
        return next((i for i in candidates if i > current), current)
    return next((i for i in reversed(candidates) if i < current), current)


def verify(path, cap, fps, rois, times, smooth, thresholds, candidates):
    N = len(times)
    pairs = saved_pairs(path, times)
    pending = None
    index = 0
    old_index = -1
    warning = ""
    playing = False
    # Pygame/Tkinter not necessary: OpenCV lets you scrub and frame-step directly.
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1020, 740)
    cv2.createTrackbar("Frame", WIN, 0, N-1, lambda _: None)
    print("\nClick the video window for keyboard input; use u then h for each pair.")
    print("IMPORTANT: move frame by frame (a/d) before marking first movement.")
    while True:
        slider = cv2.getTrackbarPos("Frame", WIN)
        if slider != index:
            index = slider
            playing = False
        if index != old_index:
            current = frame_at(cap, index)
            previous = frame_at(cap, max(0, index-1))
            old_index = index
        img = preview(current, previous, rois, index, times, smooth,
                      thresholds, *candidates, pairs, pending, warning)
        cv2.imshow(WIN, img)
        key = cv2.waitKey(33) & 0xFF
        warning = ""
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            playing = not playing
        elif key == ord("a"):
            index -= 1
            playing = False
        elif key == ord("d"):
            index += 1
            playing = False
        elif key == ord("z"):
            index -= 10
            playing = False
        elif key == ord("x"):
            index += 10
            playing = False
        elif key == ord("j"):
            index -= round(fps)
            playing = False
        elif key == ord("l"):
            index += round(fps)
            playing = False
        elif key == ord("n"):
            index = next_hint(candidates[0], index, True)
            playing = False
        elif key == ord("b"):
            index = next_hint(candidates[0], index, False)
            playing = False
        elif key == ord("m"):
            index = next_hint(candidates[1], index, True)
            playing = False
        elif key == ord("k"):
            index = next_hint(candidates[1], index, False)
            playing = False
        elif key == ord("u"):
            pending = index
            playing = False
            print(f"USER onset: frame {pending}, time {times[pending]:.3f}s")
        elif key == ord("h"):
            playing = False
            if pending is None:
                warning = "First press u at the USER onset frame."
            elif index <= pending:
                warning = "Prosthetic onset must FOLLOW user onset."
            else:
                lag = times[index] - times[pending]
                if lag > MAX_LAG_S:
                    warning = f"Delay {lag:.2f}s exceeds MAX_LAG_S={MAX_LAG_S}s. Review event."
                elif (pending, index) in pairs:
                    warning = "This pair is already saved."
                    pending = None
                elif any(pending == u or index == h for u, h in pairs):
                    warning = "A frame is already used in another pair. Undo it first."
                else:
                    pairs.append((pending, index))
                    save_pairs(path, pairs, times, fps)  # keep a recovery copy immediately
                    print(f"VERIFIED: user {times[pending]:.3f}s -> "
                          f"prosthetic {times[index]:.3f}s = {lag*1000:.1f} ms")
                    pending = None
        elif key == ord("c"):
            pending = None
        elif key in (8, 127):
            if pairs:
                pair = pairs.pop()
                save_pairs(path, pairs, times, fps)
                print(f"Removed pair: user frame {pair[0]}, hand frame {pair[1]}")
            else:
                warning = "No pairs to undo."
        elif playing and index < N-1:
            index += 1
        elif playing:
            playing = False
        if warning:
            print("NOTE:", warning)
        index = max(0, min(N-1, index))
        cv2.setTrackbarPos("Frame", WIN, index)
    cv2.destroyWindow(WIN)
    pairs.sort()
    return pairs


def write_summary(all_data):
    output = Path("latency_results.csv")
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "rep", "user_frame", "user_time_s", "hand_frame",
                    "hand_time_s", "latency_ms", "fps", "manually_verified"])
        for video, pairs, times, fps in all_data:
            for j, (u, h) in enumerate(pairs, 1):
                w.writerow([str(video), j, u, f"{times[u]:.6f}", h,
                            f"{times[h]:.6f}", f"{(times[h]-times[u])*1000:.2f}",
                            f"{fps:.6f}", "yes"])
    latencies = np.asarray([1000*(times[h]-times[u])
                            for _, pairs, times, _ in all_data for u, h in pairs])
    if len(latencies):
        std = float(np.std(latencies, ddof=1)) if len(latencies) > 1 else float("nan")
        print(f"\nREVIEWED RESULTS (n={len(latencies)}): "
              f"mean={latencies.mean():.1f} ms, median={np.median(latencies):.1f} ms, "
              f"std={std:.1f} ms, range=[{latencies.min():.1f}, {latencies.max():.1f}] ms")
        print("Report video frame resolution and number of trials with the results.")
    else:
        print("\nNo manually verified movement pairs. No latency values reported.")
    print("Saved", output)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="*", type=Path)
    p.add_argument("--scan-only", action="store_true",
                   help="Save diagnostic plots of suggestions; no interactive verification")
    args = p.parse_args()
    paths = args.videos or clips()
    if not paths:
        p.error("No videos. Place videos in videos/ or pass one or more file paths.")
    OUT_DIR.mkdir(exist_ok=True)
    all_data = []
    for video in paths:
        cap = None
        try:
            print(f"\n==== {video} ====")
            cap, fps, width, height = video_info(video)
            roi = load_or_pick_rois(video, cap, width, height)
            signals, times = scan_signals(video, roi, fps)
            analysed = [onset_suggestions(s, times) for s in signals]
            candidates = [analysed[i][0] for i in range(2)]
            smooth = np.vstack([analysed[i][1] for i in range(2)])
            thresholds = [analysed[i][2] for i in range(2)]
            print(f"Frames: {len(times)}, FPS: {fps:.3f}, "
                  f"duration: {times[-1]:.2f}s")
            print(f"Automatic hints: {len(candidates[0])} user, "
                  f"{len(candidates[1])} prosthetic (NOT verified events)")
            if args.scan_only:
                pairs = []  # Plot only: NEVER write pseudo-verified data.
            else:
                pairs = verify(video, cap, fps, roi, times, smooth,
                               thresholds, candidates)
                fn = save_pairs(video, pairs, times, fps)
                print("Saved reviewed events:", fn)
                all_data.append((video, pairs, times, fps))
            png = plot_diagnostic(video, times, smooth, thresholds, candidates, pairs)
            print("Saved diagnostic:", png)
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
    if not args.scan_only:
        write_summary(all_data)


if __name__ == "__main__":
    main()

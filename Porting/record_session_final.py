"""
record_session.py  (robust version)

Host-side recording script for own-arm EMG data collection.
Run while esp32_raw_capture.py streams "timestamp_ms,ch1,ch2,ch3,ch4" lines
over serial. Disconnect Thonny first or the port will be busy.

Output (in --outdir, default data_4):
    <subject>_cycle<N>.csv        columns: sample_idx, timestamp_ms, ch1..ch4,
                                  cycle, gesture, rep, phase,
                                  t_in_phase_ms, wear_min
    <subject>_cycle<N>_meta.json  schedule, seed, pre-flight stats, per-phase
                                  stats (rate, gaps, rest baselines), fatigue
                                  ratings, warnings, completion status

The first 10 columns match the old format. phase is "rest" or "contract".
t_in_phase_ms lets you trim the onset later (e.g. drop the first 500 ms of
each contract). wear_min is minutes since electrodes went on.

Robustness features:
  * pre-flight signal check (flat channels, ADC rail, no data)
  * bulk serial reads, display updated at 10 Hz (no per-sample printing)
  * dropped-sample / rate / backlog warnings per phase
  * data written to .partial.csv and renamed only on completion, so an abort
    or crash never destroys a previous good file and keeps what was recorded
  * optional interleaved trial order to spread fatigue evenly over gestures
  * seed, order, rest-baselines and fatigue ratings saved for drift analysis

Usage:
    python record_session.py --subject manou --cycle 1 --port COM5
    python record_session.py --subject manou --cycle 2 --port COM5 \
        --wear-start 14:05 --order interleaved --fs 1500
"""

import argparse
import csv
import json
import os
import random
import statistics
import threading
import time
from datetime import datetime

import serial  # pip install pyserial

# Keep label spellings identical across sessions (older sets used
# "rocnroll" and "pinch_rest"; pick one and stick to it).
GESTURES = [
    "hand_close",
    "index_up",
    "pinch",
    "rocknroll",
]

REPS_PER_GESTURE = 5
REST_S = 3.0
CONTRACT_S = 3.0
BAUD_RATE = 115200  # ignored by native USB CDC, kept for UART boards
COLUMNS = ["sample_idx", "timestamp_ms", "ch1", "ch2", "ch3", "ch4",
           "cycle", "gesture", "rep", "phase", "t_in_phase_ms", "wear_min"]


def beep(freq=1000, ms=80):
    """Non-blocking audio cue so reading never stalls."""
    def _b():
        try:
            import winsound
            winsound.Beep(freq, ms)
        except ImportError:
            print("\a", end="", flush=True)
    threading.Thread(target=_b, daemon=True).start()


class SerialReader:
    """Bulk reader: drains the serial buffer and parses complete lines."""

    def __init__(self, ser):
        self.ser = ser
        self.buf = b""
        self.bad_lines = 0
        self.max_waiting = 0

    def read_samples(self):
        n = self.ser.in_waiting
        self.max_waiting = max(self.max_waiting, n)
        data = self.ser.read(n if n else 1)
        if not data:
            return []
        self.buf += data
        *lines, self.buf = self.buf.split(b"\n")
        out = []
        for ln in lines:
            parts = ln.strip().split(b",")
            if len(parts) != 5:
                self.bad_lines += 1
                continue
            try:
                out.append(tuple(int(p) for p in parts))
            except ValueError:
                self.bad_lines += 1
        return out


class State:
    def __init__(self, cycle, wear_t0):
        self.idx = 0
        self.cycle = cycle
        self.wear_t0 = wear_t0


def preflight(reader, seconds, adc_max, flat_std):
    print(f"\nPre-flight: relax the arm for {seconds:.1f}s ...")
    rows = []
    end = time.time() + seconds
    while time.time() < end:
        rows.extend(reader.read_samples())
    if not rows:
        return None, ["no data received (Thonny disconnected? "
                      "esp32_raw_capture.py running? correct port?)"]
    stats, warnings = [], []
    print(f"{'ch':<4}{'mean':>9}{'std':>9}{'min':>7}{'max':>7}")
    for c in range(4):
        v = [r[1 + c] for r in rows]
        m, s = statistics.fmean(v), statistics.pstdev(v)
        stats.append({"ch": c + 1, "mean": round(m, 1), "std": round(s, 2),
                      "min": min(v), "max": max(v)})
        print(f"{c + 1:<4}{m:9.1f}{s:9.2f}{min(v):7d}{max(v):7d}")
        if s < flat_std:
            warnings.append(f"ch{c + 1} looks flat (std {s:.2f}): "
                            "check lead / electrode contact")
        if min(v) <= 5 or max(v) >= adc_max - 5:
            warnings.append(f"ch{c + 1} touches the ADC rail: "
                            "check supply / gain / contact")
    return stats, warnings


def phase_stats(gesture, rep, phase, ts_list, chans, backlog):
    n = len(ts_list)
    s = {"gesture": gesture, "rep": rep, "phase": phase, "n": n,
         "max_backlog_bytes": backlog}
    if n > 1:
        span = ts_list[-1] - ts_list[0]
        s["rate_hz"] = round(1000.0 * (n - 1) / span, 1) if span > 0 else None
        s["max_gap_ms"] = max(b - a for a, b in zip(ts_list, ts_list[1:]))
    if phase == "rest" and n:
        s["mean"] = [round(statistics.fmean(c), 1) for c in chans]
        s["std"] = [round(statistics.pstdev(c), 2) for c in chans]
    return s


def check_phase(s, args):
    w = []
    if s["n"] == 0:
        w.append("no samples received")
    if s.get("max_gap_ms", 0) > args.gap_ms:
        w.append(f"{s['max_gap_ms']} ms gap (dropped samples?)")
    if args.fs and s.get("rate_hz") and abs(s["rate_hz"] - args.fs) > 0.05 * args.fs:
        w.append(f"rate {s['rate_hz']} Hz vs expected {args.fs}")
    if s["max_backlog_bytes"] > 16384:
        w.append("serial backlog: host falling behind, labels may lag")
    return w


def run_phase(reader, writer, st, gesture, rep, phase, duration):
    label = "REST" if phase == "rest" else "CONTRACT"
    if phase == "contract":
        print(f"\n>>> {gesture.upper()} <<<")
        beep(1200, 100)
    end = time.time() + duration
    last_disp = 0.0
    first_ts = None
    ts_list = []
    chans = ([], [], [], [])
    reader.max_waiting = 0

    while True:
        now = time.time()
        if now >= end:
            break
        if now - last_disp >= 0.1:
            print(f"\r{label:<9s}{end - now:4.1f}s ", end="", flush=True)
            last_disp = now
        batch = reader.read_samples()
        if not batch:
            continue
        wear = round((time.time() - st.wear_t0) / 60.0, 2)
        for ts, a, b, c, d in batch:
            if first_ts is None:
                first_ts = ts
            writer.writerow([st.idx, ts, a, b, c, d, st.cycle, gesture, rep,
                             phase, ts - first_ts, wear])
            st.idx += 1
            ts_list.append(ts)
            for lst, v in zip(chans, (a, b, c, d)):
                lst.append(v)

    if phase == "contract":
        beep(600, 80)
    print(f"\r{label:<9s}done    ", end="", flush=True)
    return phase_stats(gesture, rep, phase, ts_list, chans, reader.max_waiting)


def build_schedule(gestures, reps, order, rng):
    if order == "block":  # all reps of one gesture together (old protocol)
        g = gestures.copy()
        rng.shuffle(g)
        return [(x, r) for x in g for r in range(1, reps + 1)]
    sched = []  # interleaved: every round visits all gestures in new order
    for r in range(1, reps + 1):
        g = gestures.copy()
        rng.shuffle(g)
        sched += [(x, r) for x in g]
    return sched


def ask_fatigue(when):
    v = input(f"Perceived fatigue {when} (0-10, Enter to skip): ").strip()
    return int(v) if v.isdigit() and 0 <= int(v) <= 10 else None


def main():
    p = argparse.ArgumentParser(description="Record one EMG data collection cycle.")
    p.add_argument("--subject", required=True)
    p.add_argument("--cycle", required=True, type=int)
    p.add_argument("--port", required=True)
    p.add_argument("--reps", type=int, default=REPS_PER_GESTURE)
    p.add_argument("--rest", type=float, default=REST_S)
    p.add_argument("--contract", type=float, default=CONTRACT_S)
    p.add_argument("--outdir", default="data_final")
    p.add_argument("--order", choices=["block", "interleaved"], default="block",
                   help="block matches the old protocol; interleaved spreads "
                        "fatigue evenly across gestures")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--wear-start", default=None,
                   help="HH:MM when electrodes were applied (default: now)")
    p.add_argument("--fs", type=float, default=None,
                   help="expected sample rate in Hz, enables rate warnings")
    p.add_argument("--gap-ms", type=int, default=20,
                   help="warn if consecutive samples are further apart (ms)")
    p.add_argument("--adc-max", type=int, default=4095)
    p.add_argument("--flat-std", type=float, default=2.0)
    p.add_argument("--preflight-s", type=float, default=3.0)
    p.add_argument("--notes", default="", help="free text, e.g. electrode placement")
    args = p.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    schedule = build_schedule(GESTURES, args.reps, args.order, rng)

    now = time.time()
    if args.wear_start:
        t = datetime.strptime(args.wear_start, "%H:%M").time()
        wear_t0 = datetime.combine(datetime.now().date(), t).timestamp()
        if wear_t0 > now:
            raise SystemExit("--wear-start is in the future")
    else:
        wear_t0 = now

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, f"{args.subject}_cycle{args.cycle}")
    final_path, partial_path, meta_path = base + ".csv", base + ".partial.csv", base + "_meta.json"

    if os.path.exists(final_path):
        if input(f"{final_path} exists. Overwrite? [y/N] ").lower() != "y":
            print("Aborted.")
            return

    print(f"\nSubject: {args.subject}  |  Cycle: {args.cycle}  |  Order: {args.order}  |  Seed: {seed}")
    print(f"Trials: {len(schedule)}  |  Rest {args.rest}s / Contract {args.contract}s")
    print("Sequence:", ", ".join(f"{g}#{r}" for g, r in schedule))
    fatigue_start = ask_fatigue("now")
    input("\nPress Enter when electrodes are checked and you're ready...")

    ser = serial.Serial(args.port, BAUD_RATE, timeout=0.02)
    time.sleep(2)
    ser.reset_input_buffer()
    reader = SerialReader(ser)

    pre_stats, pre_warn = preflight(reader, args.preflight_s, args.adc_max, args.flat_std)
    for w in pre_warn:
        print(f"  ! {w}")
    if pre_warn and input("Continue anyway? [y/N] ").lower() != "y":
        ser.close()
        print("Aborted.")
        return

    st = State(args.cycle, wear_t0)
    phase_log, flags = [], []
    completed_trials, completed, error = 0, False, None
    f = open(partial_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(COLUMNS)

    try:
        for i, (gesture, rep) in enumerate(schedule, 1):
            print(f"\n\n=== Trial {i}/{len(schedule)}: {gesture} (rep {rep}) ===")
            for phase, dur in (("rest", args.rest), ("contract", args.contract),
                               ("rest", args.rest)):
                s = run_phase(reader, writer, st, gesture, rep, phase, dur)
                phase_log.append(s)
                for msg in check_phase(s, args):
                    print(f"\n  ! {gesture} rep {rep} {phase}: {msg}")
                    flags.append({"trial": i, "gesture": gesture, "rep": rep,
                                  "phase": phase, "warning": msg})
            f.flush()
            completed_trials = i
        completed = True
    except KeyboardInterrupt:
        print("\n\nInterrupted: keeping what was recorded so far.")
    except Exception as e:  # serial drop, etc.
        error = repr(e)
        print(f"\n\nError: {error}. Keeping what was recorded so far.")
    finally:
        f.close()
        ser.close()

    fatigue_end = ask_fatigue("now (end of cycle)") if completed else None
    if completed:
        os.replace(partial_path, final_path)
        out_path = final_path
    else:
        out_path = partial_path

    meta = {
        "subject": args.subject, "cycle": args.cycle, "args": vars(args),
        "seed": seed, "schedule": schedule,
        "start_iso": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
        "wear_start_iso": datetime.fromtimestamp(wear_t0).isoformat(timespec="seconds"),
        "fatigue_start": fatigue_start, "fatigue_end": fatigue_end,
        "preflight": pre_stats, "preflight_warnings": pre_warn,
        "phases": phase_log, "warnings": flags,
        "bad_serial_lines": reader.bad_lines,
        "completed": completed, "completed_trials": completed_trials,
        "error": error,
    }
    with open(meta_path, "w") as mf:
        json.dump(meta, mf, indent=2)

    print(f"\nSamples: {st.idx}  |  Trials completed: {completed_trials}/{len(schedule)}")
    print(f"Bad serial lines: {reader.bad_lines}  |  Phase warnings: {len(flags)}")
    print(f"Saved: {out_path}")
    print(f"Meta:  {meta_path}")
    if not completed:
        print("Cycle incomplete: do not use the .partial.csv as a full cycle.")


if __name__ == "__main__":
    main()
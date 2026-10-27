# hand.py
from machine import Pin, PWM
import time

# (pin, min_us, max_us) per finger. Fill in each finger's own pin and measured limits.
_CFG = (
    (9,  375, 2600),   # thumb
    (8,  375, 2600),   # index
    (6, 375, 2600),   # middle
    (7, 375, 2600),   # ring
    (5, 375, 2600),   # pinky
)
THUMB, INDEX, MIDDLE, RING, PINKY = range(5)

class Finger:
    __slots__ = ("pwm", "lo", "span")

    def __init__(self, pin, lo, hi):
        self.pwm = PWM(Pin(pin), freq=50)
        self.lo = lo
        self.span = hi - lo

    def move(self, pos):  # 0 = close, 100 = flex
        if pos < 0: pos = 0
        elif pos > 100: pos = 100
        self.pwm.duty_u16(int((self.lo + self.span * pos / 100) * 3.27675))  # 65535/20000

F = [Finger(*c) for c in _CFG]

# poses: (thumb, index, middle, ring, pinky), 0 = close, 100 = flex
POSES = {
    "open":             (100, 100, 100, 100, 100),
    "close":            (0,   0,   0,   0,   0),
    "pinch":            (0,   0,   100, 100, 100),
    "tripod":           (0,   0,   0,   50,  50),
    "thumb_extension":  (100, 0,   0,   0,   0),
    "index_extension":  (0,   100, 0,   0,   0),
    "middle_extension": (0,   0,   100, 0,   0),
    "ring_extension":   (0,   0,   0,   100, 0),
    "pinky_extension":  (0,   0,   0,   0,   100),
    "peace":            (0,   100, 100, 0,   0),
    "rocknroll":        (0,   100, 0,   0,   100),
}

def do(name):
    for f, p in zip(F, POSES[name]):
        f.move(p)

def initialize(delay=5000):
    for p in (0, 100, 0):          # close -> flex -> close, one finger at a time
        for f in F:
            f.move(p)
            time.sleep(delay)

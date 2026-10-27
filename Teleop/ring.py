from machine import Pin, PWM

SERVO_PIN = 7
PWM_FREQ = 50
SERVO_MIN_US = 375   # measured min
SERVO_MAX_US = 2600  # measured max

servo = PWM(Pin(SERVO_PIN), freq=PWM_FREQ)

def set_pulse_us(pulse_us):
    pulse_us = max(SERVO_MIN_US, min(SERVO_MAX_US, pulse_us))
    duty = int((pulse_us / 20000) * 65535)
    servo.duty_u16(duty)

def set_servo_angle(angle):
    # angle: 0 = min position, 100 = max position
    angle = max(0, min(100, angle))
    pulse_us = SERVO_MIN_US + (angle / 100) * (SERVO_MAX_US - SERVO_MIN_US)
    set_pulse_us(int(pulse_us))
    return angle
def flex():
    set_servo_angle(100)
    
def close():
    set_servo_angle(0)
    
def moveto(position):
    set_servo_angle(float(position))

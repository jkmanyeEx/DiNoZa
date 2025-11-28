import RPi.GPIO as GPIO
import time
import json


# ============================================================
# Load motor config (enables/disables)
# ============================================================
with open("config.json") as f:
    CONFIG = json.load(f)

ENABLE_ST1 = CONFIG["motors"]["stepper1_enabled"]
ENABLE_ST2 = CONFIG["motors"]["stepper2_enabled"]
ENABLE_DC  = CONFIG["motors"]["dc_enabled"]


# ============================================================
# L298N Stepper Driver
# ============================================================
class L298NStepper:
    FULL_STEP_SEQ = [
        [1,0,1,0],
        [0,1,1,0],
        [0,1,0,1],
        [1,0,0,1]
    ]

    def __init__(self, pins):
        self.pins = pins
        self.index = 0

        GPIO.setmode(GPIO.BCM)
        for p in pins:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, 0)

    def step(self, direction=1, delay=0.002):
        self.index = (self.index + direction) % 4
        seq = self.FULL_STEP_SEQ[self.index]

        for pin, val in zip(self.pins, seq):
            GPIO.output(pin, val)

        time.sleep(delay)

    def rotate(self, steps, delay=0.002):
        direction = 1 if steps > 0 else -1
        for _ in range(abs(steps)):
            self.step(direction, delay)

    def release(self):
        for p in self.pins:
            GPIO.output(p, 0)


# ============================================================
# MOSFET DC driver (1-pin)
# ============================================================
class MOSFET_DC:
    def __init__(self, pin):
        self.pin = pin
        self.state = False

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    def on(self):
        self.state = True
        GPIO.output(self.pin, GPIO.HIGH)

    def off(self):
        self.state = False
        GPIO.output(self.pin, GPIO.LOW)

    def toggle(self):
        self.state = not self.state
        GPIO.output(self.pin, GPIO.HIGH if self.state else GPIO.LOW)


# ============================================================
# Main Motor Controller
# ============================================================
class MotorControl:

    def __init__(self, pinmap):
        # Create or disable Stepper 1
        if ENABLE_ST1:
            self.stepper1 = L298NStepper(pinmap["stepper1"])
        else:
            self.stepper1 = None

        # Stepper 2
        if ENABLE_ST2:
            self.stepper2 = L298NStepper(pinmap["stepper2"])
        else:
            self.stepper2 = None

        # DC shooter
        if ENABLE_DC:
            self.dc = MOSFET_DC(pinmap["dc"]["pin"])
        else:
            self.dc = None

        # motion params
        self.PAN_STEP  = 5
        self.TILT_STEP = 5
        self.SPEED     = 0.002

    # ========================================================
    # Manual Control
    # ========================================================
    def manual_control(self, cmd):

        # ---- Stepper 1 (left/right) ----
        if self.stepper1 and cmd == "a":
            self.stepper1.rotate(-self.PAN_STEP, self.SPEED)

        elif self.stepper1 and cmd == "d":
            self.stepper1.rotate(+self.PAN_STEP, self.SPEED)

        # ---- Stepper 2 (up/down) ----
        if self.stepper2 and cmd == "w":
            self.stepper2.rotate(+self.TILT_STEP, self.SPEED)

        elif self.stepper2 and cmd == "s":
            self.stepper2.rotate(-self.TILT_STEP, self.SPEED)

        # ---- Shooter Toggle ----
        if self.dc and cmd == "f":
            self.dc.toggle()

    # ========================================================
    # Auto Control
    # ========================================================
    def auto_control(self, lr, ud):

        # pan
        if self.stepper1:
            if lr == "left":
                self.stepper1.rotate(-self.PAN_STEP, self.SPEED)
            elif lr == "right":
                self.stepper1.rotate(+self.PAN_STEP, self.SPEED)

        # tilt
        if self.stepper2:
            if ud == "up":
                self.stepper2.rotate(+self.TILT_STEP, self.SPEED)
            elif ud == "down":
                self.stepper2.rotate(-self.TILT_STEP, self.SPEED)

        # auto shooting only if DC enabled
        if self.dc:
            if lr == "center" and ud == "center":
                self.dc.on()
            else:
                self.dc.off()

    # ========================================================
    # Cleanup
    # ========================================================
    def cleanup(self):
        if self.stepper1:
            self.stepper1.release()

        if self.stepper2:
            self.stepper2.release()

        if self.dc:
            self.dc.off()

        GPIO.cleanup()

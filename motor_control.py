#
import RPi.GPIO as GPIO
import time


class L298NStepper:
    """
    Stepper motor controlled via L298N using 4 GPIO pins:
    pins = [IN1, IN2, IN3, IN4]
    FULL-STEP sequence (bipolar).
    """

    FULL_STEP_SEQ = [
        [1, 0, 1, 0],  # step 0
        [0, 1, 1, 0],  # step 1
        [0, 1, 0, 1],  # step 2
        [1, 0, 0, 1],  # step 3
    ]

    def __init__(self, pins):
        assert len(pins) == 4, "L298NStepper needs 4 pins [IN1, IN2, IN3, IN4]"
        self.pins = pins
        self.index = 0

        for p in self.pins:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, GPIO.LOW)

    def step(self, direction=1, delay=0.002):
        """
        direction: +1 for CW, -1 for CCW (depends on wiring).
        delay: time between microsteps.
        """
        self.index = (self.index + direction) % 4
        seq = self.FULL_STEP_SEQ[self.index]

        for pin, val in zip(self.pins, seq):
            GPIO.output(pin, GPIO.HIGH if val else GPIO.LOW)

        time.sleep(delay)

    def rotate(self, steps, delay=0.002):
        """
        Rotate by 'steps' full-steps. Positive = one way, negative = other.
        """
        direction = 1 if steps > 0 else -1
        steps = abs(steps)
        for _ in range(steps):
            self.step(direction, delay)

    def release(self):
        """De-energize coils."""
        for p in self.pins:
            GPIO.output(p, GPIO.LOW)


class MOSFET_DC:
    """
    DC motor via MOSFET (one GPIO pin to gate).
    HIGH = motor ON, LOW = motor OFF.
    """

    def __init__(self, pin):
        self.pin = pin
        self.state = False  # False=OFF, True=ON

        GPIO.setup(self.pin, GPIO.OUT)
        GPIO.output(self.pin, GPIO.LOW)

    def on(self):
        self.state = True
        GPIO.output(self.pin, GPIO.HIGH)

    def off(self):
        self.state = False
        GPIO.output(self.pin, GPIO.LOW)

    def toggle(self):
        self.state = not self.state
        GPIO.output(self.pin, GPIO.HIGH if self.state else GPIO.LOW)


class MotorControl:
    """
    High-level turret motor control:
      - stepper1: pan (L298N)
      - stepper2: tilt (L298N)
      - dc: shooter (MOSFET)
    """

    def __init__(self, pinmap):
        """
        pinmap = {
            "stepper1": [IN1, IN2, IN3, IN4],
            "stepper2": [IN1, IN2, IN3, IN4],
            "dc": {"pin": X}
        }
        """

        GPIO.setmode(GPIO.BCM)

        self.stepper1 = L298NStepper(pinmap["stepper1"])
        self.stepper2 = L298NStepper(pinmap["stepper2"])
        self.dc = MOSFET_DC(pinmap["dc"]["pin"])

        # Default movement tuning
        self.PAN_STEP = 5
        self.TILT_STEP = 5
        self.SPEED = 0.002  # seconds between microsteps

    # ---------------------------------------------------------
    # MANUAL MODE
    # ---------------------------------------------------------
    def manual_control(self, cmd: str):
        if cmd == "a":
            # pan left
            self.stepper1.rotate(-self.PAN_STEP, self.SPEED)
        elif cmd == "d":
            # pan right
            self.stepper1.rotate(self.PAN_STEP, self.SPEED)
        elif cmd == "w":
            # tilt up
            self.stepper2.rotate(self.TILT_STEP, self.SPEED)
        elif cmd == "s":
            # tilt down
            self.stepper2.rotate(-self.TILT_STEP, self.SPEED)
        elif cmd == "f":
            # toggle shooter
            self.dc.toggle()

    # ---------------------------------------------------------
    # AUTO MODE
    # ---------------------------------------------------------
    def auto_control(self, lr: str, ud: str):
        """
        lr: "left" / "right" / "center"
        ud: "up" / "down" / "center"
        """

        # horizontal control
        if lr == "left":
            self.stepper1.rotate(-self.PAN_STEP, self.SPEED)
        elif lr == "right":
            self.stepper1.rotate(self.PAN_STEP, self.SPEED)

        # vertical control
        if ud == "up":
            self.stepper2.rotate(self.TILT_STEP, self.SPEED)
        elif ud == "down":
            self.stepper2.rotate(-self.TILT_STEP, self.SPEED)

        # shooting control: auto ON only when fully centered
        if lr == "center" and ud == "center":
            self.dc.on()
        else:
            self.dc.off()

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------
    def cleanup(self):
        self.stepper1.release()
        self.stepper2.release()
        self.dc.off()
        GPIO.cleanup()

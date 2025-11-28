import RPi.GPIO as GPIO
import time


class MotorControl:
    def __init__(self, pinmap, step_count):
        """
        pinmap example:
        {
            "stepper1": {"dir": 5, "step": 6},
            "stepper2": {"dir": 13, "step": 19},
            "dc": {"pin": 20}   # <-- single pin DC motor
        }

        step_count example: 200 (full steps per rev)
        """
        self.pins = pinmap
        self.step_count = step_count

        GPIO.setmode(GPIO.BCM)

        # Setup stepper 1
        GPIO.setup(self.pins["stepper1"]["dir"], GPIO.OUT)
        GPIO.setup(self.pins["stepper1"]["step"], GPIO.OUT)

        # Setup stepper 2
        GPIO.setup(self.pins["stepper2"]["dir"], GPIO.OUT)
        GPIO.setup(self.pins["stepper2"]["step"], GPIO.OUT)

        # Setup DC motor (single pin)
        GPIO.setup(self.pins["dc"]["pin"], GPIO.OUT)

    #########################################################
    # UTILITIES
    #########################################################
    def _do_steps(self, dir_pin, step_pin, steps, delay):
        GPIO.output(dir_pin, GPIO.HIGH if steps > 0 else GPIO.LOW)
        steps = abs(steps)

        for _ in range(steps):
            GPIO.output(step_pin, GPIO.HIGH)
            time.sleep(delay)
            GPIO.output(step_pin, GPIO.LOW)
            time.sleep(delay)

    #########################################################
    # STEPPER 1
    #########################################################
    def rotate_stepper1(self, steps, speed=0.001):
        s = self.pins["stepper1"]
        self._do_steps(s["dir"], s["step"], steps, speed)

    #########################################################
    # STEPPER 2
    #########################################################
    def rotate_stepper2(self, steps, speed=0.001):
        s = self.pins["stepper2"]
        self._do_steps(s["dir"], s["step"], steps, speed)

    #########################################################
    # DC MOTOR CONTROL (single-pin)
    #########################################################
    def dc_on(self):
        """Turn DC motor ON"""
        GPIO.output(self.pins["dc"]["pin"], GPIO.HIGH)

    def dc_off(self):
        """Turn DC motor OFF"""
        GPIO.output(self.pins["dc"]["pin"], GPIO.LOW)

    #########################################################
    # CLEANUP
    #########################################################
    def cleanup(self):
        GPIO.cleanup()

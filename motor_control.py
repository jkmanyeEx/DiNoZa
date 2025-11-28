import RPi.GPIO as GPIO
import time


class MotorControl:
    def __init__(self, config: dict):
        """
        config example:
        {
            "use_stepper1": true,
            "use_stepper2": false,
            "use_dc": true,

            "stepper1": {"dir": 5, "step": 6},
            "stepper2": {"dir": 13, "step": 19},
            "dc": {"pin": 20},

            "step_size": 5,
            "speed": 0.001
        }
        """

        self.cfg = config
        GPIO.setmode(GPIO.BCM)

        if config["use_stepper1"]:
            GPIO.setup(config["stepper1"]["dir"], GPIO.OUT)
            GPIO.setup(config["stepper1"]["step"], GPIO.OUT)

        if config["use_stepper2"]:
            GPIO.setup(config["stepper2"]["dir"], GPIO.OUT)
            GPIO.setup(config["stepper2"]["step"], GPIO.OUT)

        if config["use_dc"]:
            GPIO.setup(config["dc"]["pin"], GPIO.OUT)

    # ----------------------------
    def _do_steps(self, pin_dir, pin_step, steps, delay):
        GPIO.output(pin_dir, GPIO.HIGH if steps > 0 else GPIO.LOW)
        steps = abs(steps)
        for _ in range(steps):
            GPIO.output(pin_step, 1)
            time.sleep(delay)
            GPIO.output(pin_step, 0)
            time.sleep(delay)

    # ----------------------------
    def stepper1(self, steps):
        if not self.cfg["use_stepper1"]:
            return
        c = self.cfg
        self._do_steps(c["stepper1"]["dir"], c["stepper1"]["step"],
                       steps, c["speed"])

    def stepper2(self, steps):
        if not self.cfg["use_stepper2"]:
            return
        c = self.cfg
        self._do_steps(c["stepper2"]["dir"], c["stepper2"]["step"],
                       steps, c["speed"])

    # ----------------------------
    def dc_on(self):
        if not self.cfg["use_dc"]:
            return
        GPIO.output(self.cfg["dc"]["pin"], 1)

    def dc_off(self):
        if not self.cfg["use_dc"]:
            return
        GPIO.output(self.cfg["dc"]["pin"], 0)

    # ----------------------------
    def cleanup(self):
        GPIO.cleanup()

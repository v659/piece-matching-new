import time
import inspect
import json
import os

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

from pyfirmata import Arduino, util

# Two physically separate Arduinos, each on its own serial port (a USB hub does
# NOT merge them into one port — Firmata is one connection per board).
#   Board A (gantry):  the 4 STEP/DIR gantry steppers (X, Y, Z, A)
#   Board B (head):    the 28BYJ-48 rotation stepper + the 2 suction servos
PORT_A = "/dev/cu.usbserial-A5069RR4"   # e.g. "/dev/cu.usbmodem31401"
PORT_B = "/dev/cu.usbmodem1101"     # e.g. "/dev/cu.usbmodem31402"


def connect_board(name, port):
    """Try to open a board. On failure (e.g. USB not attached) return None and
    keep going, so the other board can still run."""
    try:
        board = Arduino(port)
        it = util.Iterator(board)
        it.start()
        print(f"{name} connected on {port}")
        return board
    except Exception as exc:
        print(f"WARNING: could not connect {name} on {port}: {exc}")
        print(f"  -> commands needing {name} will be skipped.")
        return None


print(f"Gantry board (A) port: {PORT_A}")
print(f"Head board   (B) port: {PORT_B}")

board_a = connect_board("Board A (gantry)", PORT_A)
board_b = connect_board("Board B (head)", PORT_B)

time.sleep(1)

# ---------------------------------------------------------------------------
# Board B — rotation stepper (28BYJ-48) + suction servos
# ---------------------------------------------------------------------------
# 28BYJ-48 specs
steps_per_revolution = 2048
rpm = 20

IN1 = 8
IN2 = 9
IN3 = 10
IN4 = 11

pins = []
servo0 = None
servo1 = None
if board_b is not None:
    pins = [
        board_b.digital[IN1],
        board_b.digital[IN2],
        board_b.digital[IN3],
        board_b.digital[IN4]
    ]
    for p in pins:
        p.mode = 1

    servo0 = board_b.get_pin('d:3:s')
    servo1 = board_b.get_pin('d:2:s')

    servo0.write(0)
    servo1.write(0)

# ---------------------------------------------------------------------------
# Board A — gantry STEP/DIR steppers
# ---------------------------------------------------------------------------
STEP_X_PIN, DIR_X_PIN = 2, 5
STEP_Y_PIN, DIR_Y_PIN = 3, 6
STEP_Z_PIN, DIR_Z_PIN = 4, 7
STEP_A_PIN, DIR_A_PIN = 12, 13

# Calibration constants — tune these for your hardware
STEPS_PER_PIXEL = 1.0          # JSON 'move' values are in image pixels
STEP_PULSE_DELAY = 0.001       # seconds between HIGH/LOW edges per step
PICKUP_Z_STEPS = 400           # arbitrary distance to lower head for pickup
DROP_OFFSET_STEPS = 20         # drop a few mm ABOVE the pickup level
DROP_Z_STEPS = PICKUP_Z_STEPS - DROP_OFFSET_STEPS

# Direction conventions — flip if your wiring is reversed
DIR_X_POS = 1
DIR_Y_POS = 1
DIR_Z_DOWN = 0  # which dir-pin level lowers the Z gantry

step_x = dir_x = step_y = dir_y = step_z = dir_z = step_a = dir_a = None
stepper_pins = []
if board_a is not None:
    step_x = board_a.digital[STEP_X_PIN]
    dir_x = board_a.digital[DIR_X_PIN]
    step_y = board_a.digital[STEP_Y_PIN]
    dir_y = board_a.digital[DIR_Y_PIN]
    step_z = board_a.digital[STEP_Z_PIN]
    dir_z = board_a.digital[DIR_Z_PIN]
    step_a = board_a.digital[STEP_A_PIN]
    dir_a = board_a.digital[DIR_A_PIN]

    stepper_pins = [step_x, dir_x, step_y, dir_y, step_z, dir_z, step_a, dir_a]
    for p in stepper_pins:
        p.mode = 1
        p.write(0)


def _pulse(step_pin):
    step_pin.write(1)
    time.sleep(STEP_PULSE_DELAY)
    step_pin.write(0)
    time.sleep(STEP_PULSE_DELAY)


def step_axis(step_pin, dir_pin, steps, dir_pos=1):
    if steps == 0:
        return
    dir_pin.write(dir_pos if steps > 0 else 1 - dir_pos)
    for _ in range(abs(steps)):
        _pulse(step_pin)


def step_z_parallel(steps):
    """Drive Z and A motors together — both belong to the vertical axis.
    Positive `steps` raises the head, negative lowers it."""
    if board_a is None:
        print("  (gantry board A unavailable — skipping Z move)")
        return
    if steps == 0:
        return
    # negative = down
    level = DIR_Z_DOWN if steps < 0 else 1 - DIR_Z_DOWN
    dir_z.write(level)
    dir_a.write(level)
    for _ in range(abs(steps)):
        step_z.write(1)
        step_a.write(1)
        time.sleep(STEP_PULSE_DELAY)
        step_z.write(0)
        step_a.write(0)
        time.sleep(STEP_PULSE_DELAY)


def move_xy(x_pixels, y_pixels):
    if board_a is None:
        print("  (gantry board A unavailable — skipping move)")
        return
    x_steps = int(round(x_pixels * STEPS_PER_PIXEL))
    y_steps = int(round(y_pixels * STEPS_PER_PIXEL))
    print(f"MOVE → x: {x_pixels:.2f}px ({x_steps} steps), y: {y_pixels:.2f}px ({y_steps} steps)")
    step_axis(step_x, dir_x, x_steps, dir_pos=DIR_X_POS)
    step_axis(step_y, dir_y, y_steps, dir_pos=DIR_Y_POS)

step_sequence = [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
]

step_delay = 60.0 / (rpm * steps_per_revolution)


def step_motor(steps):
    if board_b is None:
        print("  (head board B unavailable — skipping rotation)")
        return
    direction = 1 if steps > 0 else -1
    steps = abs(steps)

    for _ in range(steps):
        sequence = step_sequence if direction == 1 else list(reversed(step_sequence))

        for step in sequence:
            for pin, val in zip(pins, step):
                pin.write(val)
            time.sleep(step_delay)

    for pin in pins:
        pin.write(0)


def pick_up():
    print("PICK UP")
    step_z_parallel(-PICKUP_Z_STEPS)  # lower head to piece (board A)

    if servo0 is not None and servo1 is not None:
        servo0.write(180)
        servo1.write(0)
        time.sleep(1.5)

        servo0.write(0)
        time.sleep(0.08)

        servo1.write(180)
        time.sleep(3)
    else:
        print("  (suction servos on board B unavailable — skipping grip)")

    step_z_parallel(PICKUP_Z_STEPS)   # raise head back (board A)


def drop():
    print("DROP")
    step_z_parallel(-DROP_Z_STEPS)    # lower head to a few mm above pickup level (board A)

    if servo1 is not None:
        servo1.write(0)
        time.sleep(0.8)
    else:
        print("  (suction servo on board B unavailable — skipping release)")

    step_z_parallel(DROP_Z_STEPS)     # raise head back (board A)


def rotate_degrees(deg):
    steps = int((deg / 360.0) * steps_per_revolution)
    print(f"ROTATE {deg:.2f}° → {steps} steps")
    step_motor(steps)


if board_a is None and board_b is None:
    print("ERROR: neither board connected — nothing to drive. Check USB ports.")

base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "assembly_steps.json")

with open(json_path, "r") as f:
    data = json.load(f)

steps_list = data["steps"]

try:
    for step in steps_list:
        cmd = step["command"]

        if cmd == "pickup piece":
            pick_up()

        elif cmd == "drop piece":
            drop()

        elif cmd == "rotate":
            rotate_degrees(step["degrees"])

        elif cmd == "move":
            x = step["x"]
            y = step["y"]
            move_xy(x, y)

        else:
            print(f"UNKNOWN COMMAND: {cmd}")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    if board_a is not None:
        for p in stepper_pins:
            p.write(0)
        board_a.exit()
    if board_b is not None:
        for p in pins:
            p.write(0)
        board_b.exit()

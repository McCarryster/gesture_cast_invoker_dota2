# ──────────────────────────────────────────────────────────────────
# CONFIG  — edit these to match your setup
# ──────────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0       # try 1 or 2 if wrong camera opens
FLIP_CAMERA         = True    # True = mirror (natural selfie view)
MASKS_FOLDER = r"D:\very_big_work\ML_projects\hand_gesture_cast_invoker_dota2\invoker\masks"

# Invoker sphere keys
KEY_QUAS   = 'q'
KEY_WEX    = 'w'
KEY_EXORT  = 'e'
KEY_INVOKE = 'r'


# Spell slot keys
KEY_SLOT1  = 'd'   # tilt left  → slot 1
KEY_SLOT2  = 'f'   # tilt right → slot 2


# Gesture confirmation: how many consecutive frames must agree
GESTURE_HOLD_FRAMES = 10     # ~0.27s at 30fps


# Head tilt: angle from vertical (degrees) to trigger a cast
TILT_THRESHOLD_DEG  = 18    # tilt at least 18° left or right
TILT_HOLD_FRAMES    = 6     # frames head must stay tilted to confirm


# Cooldown between casts (prevents double-fire on same tilt)
CAST_COOLDOWN_SEC = 0.8
GESTURE_TRIGGER_THRESHOLD = 0.8
GESTURE_RESET_THRESHOLD = 0.4

# ──────────────────────────────────────────────────────────────────
# SPELL TABLE
# Format: spell_number → (name, [sphere, sphere, sphere])
# Q=quas  W=wex  E=exort
# ──────────────────────────────────────────────────────────────────
Q, W, E = KEY_QUAS, KEY_WEX, KEY_EXORT
SPELLS = {
    1:  ("Sun Strike",       [E, E, E]),
    2:  ("Ghost Walk",      [Q, Q, W]),
    3:  ("Ice Wall",        [Q, Q, E]),
    4:  ("EMP",             [W, W, W]),
    5:  ("Tornado",         [W, W, Q]),
    6:  ("Alacrity",        [W, W, E]),
    7:  ("Cold Snap",      [Q, Q, Q]),
    8:  ("Forge Spirit",    [E, E, Q]),
    9:  ("Deafening Blast",    [Q, W, E]),
    10: ("Chaos Meteor", [E, E, W]),
}

# UI accent colors per spell (BGR)
SPELL_COLORS = {
    1:  (220, 180,  60),
    2:  (180, 180, 180),
    3:  (200, 220, 100),
    4:  ( 60, 210, 255),
    5:  ( 80, 200, 120),
    6:  ( 50, 230, 180),
    7:  (  0,  90, 255),
    8:  (  0, 160, 210),
    9:  (  0,  50, 220),
    10: (180,  50, 240),
}
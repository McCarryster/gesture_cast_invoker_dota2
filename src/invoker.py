import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from collections import deque
from pynput.keyboard import Controller, Key
import cfg as cfg


# ──────────────────────────────────────────────────────────────────
# SHARED STATE  (read/written from main loop, thread-safe via lock)
# ──────────────────────────────────────────────────────────────────
state_lock       = threading.Lock()
current_spell    = None       # spell number currently invoked (or None)
last_cast_label  = ""         # display string for "just cast" banner
last_cast_ts     = 0.0        # timestamp for banner fade
keyboard = Controller()


# ──────────────────────────────────────────────────────────────────
# Key presser
# ──────────────────────────────────────────────────────────────────
def press_keys(keys) -> None:
    for key in keys:
        keyboard.press(key)
        keyboard.release(key)
        # time.sleep(delay)


# ──────────────────────────────────────────────────────────────────
# Gestures classifier
# ──────────────────────────────────────────────────────────────────
def get_finger_states(landmarks, handedness: str) -> list[int]:
    """
    Returns [thumb, index, middle, ring, pinky] as 1 (open) or 0 (closed).
    """
    tips = [4, 8, 12, 16, 20]
    pip = [3, 6, 10, 14, 18]
    states = []

    # Thumb: Horizontal comparison. 
    # If Right hand (mirrored), thumb is open if tip.x < pip.x
    # If Left hand (mirrored), thumb is open if tip.x > pip.x
    if handedness == "Right":
        states.append(1 if landmarks[tips[0]].x < landmarks[pip[0]].x else 0)
    else:
        states.append(1 if landmarks[tips[0]].x > landmarks[pip[0]].x else 0)

    # Other fingers: Vertical comparison (tip above PIP)
    for i in range(1, 5):
        states.append(1 if landmarks[tips[i]].y < landmarks[pip[i]].y else 0)

    return states

def classify_gesture(landmarks, handedness: str) -> int | None:
    """
    Maps finger states to Invoker spell numbers (1-10).
    """
    states = get_finger_states(landmarks, handedness)
    thumb, index, middle, ring, pinky = states
    total_extended = sum(states)

    # 1: Index up (Cold Snap)
    if index == 1 and total_extended == 1:
        return 1
    # 2: Peace (Ghost Walk)
    if index == 1 and middle == 1 and total_extended == 2:
        return 2
    # 3: Three fingers (Ice Wall)
    if index == 1 and middle == 1 and ring == 1 and total_extended == 3:
        return 3
    # 4: Four fingers (EMP)
    if index == 1 and middle == 1 and ring == 1 and pinky == 1 and total_extended == 4:
        return 4
    # 5: Open palm (Tornado)
    if total_extended == 5:
        return 5
    # 6: Shaka / Thumb + Pinky (Alacrity)
    if thumb == 1 and pinky == 1 and total_extended == 2:
        return 6
    # 7: Thumb only (Sun Strike)
    if thumb == 1 and total_extended == 1:
        return 7
    # 8: L shape / Thumb + Index (Forge Spirit)
    if thumb == 1 and index == 1 and total_extended == 2:
        return 8
    # 9: Rock / Horns (Chaos Meteor)
    if index == 1 and pinky == 1 and middle == 0 and ring == 0:
        return 9
    # 10: Fist (Deafening Blast)
    if total_extended == 0:
        return 10

    return None


# ──────────────────────────────────────────────────────────────────
# Head tilt classifiier
# ──────────────────────────────────────────────────────────────────
def head_tilt_direction(face_lm) -> str:
    """
    Returns 'LEFT', 'RIGHT', or 'NEUTRAL'.
    Uses the line between left eye (33) and right eye (263) landmarks.
    Positive angle = head tilted to user's left (after mirror flip).
    """
    left_eye  = face_lm[33]   # left eye outer corner
    right_eye = face_lm[263]  # right eye outer corner

    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y

    # angle of the eye line relative to horizontal
    angle_deg = np.degrees(np.arctan2(dy, dx))

    # After flip: positive angle = right eye lower = head tilted to user's LEFT
    if angle_deg > cfg.TILT_THRESHOLD_DEG:
        return "LEFT"
    elif angle_deg < -cfg.TILT_THRESHOLD_DEG:
        return "RIGHT"
    return "NEUTRAL"


# ──────────────────────────────────────────────────────────────────
# OVERLAY DRAWING
# ──────────────────────────────────────────────────────────────────
PANEL_W = 240

def draw_overlay(frame, detected_gesture, confirmed_gesture, tilt_dir, tilt_progress, current_spell_num, cast_label, cast_ts):
    h, w = frame.shape[:2]

    # ── Right panel background ───────────────────────────────────
    panel_x = w - PANEL_W
    overlay  = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (8, 8, 16), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    px = panel_x + 12

    # Title
    cv2.putText(frame, "INVOKER CTRL", (px, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 200, 255), 1)
    cv2.line(frame, (panel_x, 38), (w, 38), (40, 40, 60), 1)

    # ── Spell list ───────────────────────────────────────────────
    for num in range(1, 11):
        name, spheres = cfg.SPELLS[num]
        sy   = 46 + (num - 1) * 47
        col  = cfg.SPELL_COLORS[num]
        is_detected  = (num == detected_gesture)
        is_confirmed = (num == confirmed_gesture)
        is_active    = (num == current_spell_num)

        bg = (30, 30, 50)
        if is_confirmed or is_active:
            bg = tuple(min(int(c * 0.4), 255) for c in col)
        elif is_detected:
            bg = (38, 38, 62)

        cv2.rectangle(frame, (panel_x + 2, sy), (w - 2, sy + 40), bg, -1)
        if is_detected or is_confirmed or is_active:
            cv2.rectangle(frame, (panel_x + 2, sy), (w - 2, sy + 40), col, 1)

        # Gesture number badge
        badge_col = col if (is_detected or is_active) else (55, 55, 75)
        cv2.circle(frame, (px + 12, sy + 20), 12, badge_col, -1)
        num_str = str(num)
        nx = px + 7 if num < 10 else px + 5
        cv2.putText(frame, num_str, (nx, sy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0) if (is_detected or is_active) else (130, 130, 150), 1)

        # Spell name
        txt_col = col if (is_detected or is_active) else (170, 170, 190)
        cv2.putText(frame, name, (px + 28, sy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1)

        # Sphere indicators (small coloured dots)
        sx0 = px + 28
        for s in spheres:
            sc = (80, 100, 220) if s == cfg.Q else \
                 (80, 200,  80) if s == cfg.W else \
                 (30,  80, 220)
            cv2.circle(frame, (sx0, sy + 30), 5, sc, -1)
            sx0 += 13

        # Active label
        if is_active:
            cv2.putText(frame, "INVOKED", (panel_x + PANEL_W - 70, sy + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

    # ── Gesture hold progress bar ─────────────────────────────────
    bar_y = h - 100
    if detected_gesture:
        bar_w = int((panel_x - 20))
        # col   = SPELL_COLORS.get(detected_gesture, (100, 200, 100))
        col = cfg.SPELL_COLORS.get(detected_gesture, (100, 200, 100)) if detected_gesture in cfg.SPELL_COLORS else (100, 100, 100)
        cv2.rectangle(frame, (10, bar_y), (panel_x - 10, bar_y + 16), (25, 25, 40), -1)
        cv2.rectangle(frame, (10, bar_y), (10 + bar_w,   bar_y + 16), col, -1)
        cv2.putText(frame, f"Hold gesture {detected_gesture}...",
                    (12, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 220), 1)

    # ── Tilt indicator ────────────────────────────────────────────
    tilt_y  = h - 70
    tilt_cx = (panel_x - 10) // 2   # center of camera area
    tilt_w  = 120

    cv2.rectangle(frame, (tilt_cx - tilt_w, tilt_y),
                          (tilt_cx + tilt_w, tilt_y + 20), (25, 25, 40), -1)

    if tilt_dir == "LEFT":
        bar_px = int(tilt_w * tilt_progress)
        cv2.rectangle(frame, (tilt_cx - bar_px, tilt_y),
                              (tilt_cx,          tilt_y + 20), (80, 200, 120), -1)
        cv2.putText(frame, "D", (tilt_cx - tilt_w - 20, tilt_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 120), 2)
    elif tilt_dir == "RIGHT":
        bar_px = int(tilt_w * tilt_progress)
        cv2.rectangle(frame, (tilt_cx,        tilt_y),
                              (tilt_cx + bar_px, tilt_y + 20), (80, 140, 240), -1)
        cv2.putText(frame, "F", (tilt_cx + tilt_w + 6, tilt_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 140, 240), 2)

    cv2.line(frame, (tilt_cx, tilt_y - 2), (tilt_cx, tilt_y + 22), (80, 80, 100), 1)
    cv2.putText(frame, "TILT", (tilt_cx - 16, tilt_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 140), 1)

    # ── Current gesture big number ────────────────────────────────
    if detected_gesture:
        col = cfg.SPELL_COLORS.get(detected_gesture, (255, 255, 255))
        cv2.putText(frame, str(detected_gesture), (10, 75),
                    cv2.FONT_HERSHEY_COMPLEX, 2.8, col, 4)
        cv2.putText(frame, cfg.SPELLS[detected_gesture][0], (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

    # ── Last cast banner ──────────────────────────────────────────
    if cast_label and time.time() - cast_ts < 2.0:
        cv2.putText(frame, f"CAST  {cast_label}", (10, h - 16),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 180), 2)

    return frame

def draw_hand_landmarks(frame, hand_lm, mp_drawing, mp_hands):
    mp_drawing.draw_landmarks(
        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 220, 255), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0, 100, 180), thickness=2),
    )

def draw_face_landmarks(frame, face_lm, mp_drawing, mp_face):
    mp_drawing.draw_landmarks(
        frame, face_lm,
        mp_face.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(60, 60, 100), thickness=1, circle_radius=0),
    )


# ──────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────
def main():
    last_tilt_time = 0.0
    gesture_ready = True

    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {cfg.CAMERA_INDEX}. "
               "Try changing CAMERA_INDEX at the top of the script.")
        return

    print("=" * 50)
    print(" Invoker Gesture Controller  —  running")
    print(" Focus Dota 2 window, then use gestures + head tilt")
    print(" Press Q or ESC in the camera window to quit")
    print("=" * 50)

    # Rolling history buffers
    gesture_history = deque(maxlen=cfg.GESTURE_HOLD_FRAMES)
    tilt_history    = deque(maxlen=cfg.TILT_HOLD_FRAMES)
    

    confirmed_gesture = None   # gesture that just fired an invoke
    tilt_confirmed    = False  # True for the frame a tilt was confirmed

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.6, max_num_hands=1) as hands, \
         mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.6, min_tracking_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Frame capture failed.")
                break

            if cfg.FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)
            rgb.flags.writeable = True


            # ── Hand gesture detection ────────────────────────────
            detected_gesture = None
            gesture_confirmed = None
            if hand_results.multi_hand_landmarks:
                hand_lm   = hand_results.multi_hand_landmarks[0]
                hand_info = hand_results.multi_handedness[0]
                label     = hand_info.classification[0].label
                draw_hand_landmarks(frame, hand_lm, mp_drawing, mp_hands)
                detected_gesture = classify_gesture(hand_lm.landmark, label)
            gesture_history.append(detected_gesture)

            if gesture_history.count(None) == len(gesture_history):
                gesture_ready = True

            # Confirm tilt when GESTURE_HOLD_FRAMES agree
            if detected_gesture is not None and len(gesture_history) == cfg.GESTURE_HOLD_FRAMES:
                count = gesture_history.count(detected_gesture)
                gesture_progress = count / cfg.GESTURE_HOLD_FRAMES

                # trigger
                if gesture_progress >= cfg.GESTURE_TRIGGER_THRESHOLD and gesture_ready:
                    gesture_confirmed = detected_gesture
                    gesture_ready = False

                # re-arm (IMPORTANT)
                elif gesture_progress <= cfg.GESTURE_RESET_THRESHOLD:
                    gesture_ready = True
            

            # ── Head tilt detection ───────────────────────────────
            tilt_dir  = "NEUTRAL"
            tilt_confirmed = False
            if face_results.multi_face_landmarks:
                face_lm = face_results.multi_face_landmarks[0]
                tilt_dir = head_tilt_direction(face_lm.landmark) # progressbar for head tilt
            tilt_history.append(tilt_dir)

            # Confirm tilt when TILT_HOLD_FRAMES agree on LEFT or RIGHT
            tilt_progress = 0.0
            if tilt_dir in ("LEFT", "RIGHT") and len(tilt_history) == cfg.TILT_HOLD_FRAMES:
                count = tilt_history.count(tilt_dir)
                tilt_progress = count / cfg.TILT_HOLD_FRAMES
                if tilt_progress >= 0.85:
                    tilt_confirmed = True


            # ── Dota integration ──────────────────────────────────
            if gesture_confirmed:
                keys = list(cfg.SPELLS[gesture_confirmed][1])
                keys.append(cfg.KEY_INVOKE)
                press_keys(keys)

            now = time.time()
            if tilt_confirmed and tilt_dir in ("LEFT", "RIGHT") and now - last_tilt_time > cfg.CAST_COOLDOWN_SEC:
                if tilt_dir == "LEFT":
                    press_keys([cfg.KEY_SLOT2])
                elif tilt_dir == "RIGHT":
                    press_keys([cfg.KEY_SLOT1])
                last_tilt_time = now


            # ── Draw overlay ──────────────────────────────────────
            with state_lock:
                _spell   = current_spell
                _label   = last_cast_label
                _cast_ts = last_cast_ts

            frame = draw_overlay(
                frame,
                detected_gesture,
                confirmed_gesture,
                tilt_dir,
                tilt_progress,
                _spell,
                _label,
                _cast_ts,
            )

            cv2.imshow("Invoker Controller  —  ESC / Q to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nInvoker Controller closed.")


if __name__ == "__main__":
    main()

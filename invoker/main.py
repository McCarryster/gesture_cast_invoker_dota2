import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import os
import glob
from collections import deque
from pynput.keyboard import Controller
import cfg


# ──────────────────────────────────────────────────────────────────
# SHARED STATE
# ──────────────────────────────────────────────────────────────────
state_lock      = threading.Lock()
current_spell   = None
last_cast_label = ""
last_cast_ts    = 0.0
keyboard        = Controller()


# ──────────────────────────────────────────────────────────────────
# KEY PRESSER
# ──────────────────────────────────────────────────────────────────
def press_keys(keys, delay: float = 0.05) -> None:
    for key in keys:
        keyboard.press(key)
        keyboard.release(key)
        time.sleep(delay)


# ──────────────────────────────────────────────────────────────────
# GESTURE CLASSIFIER
# ──────────────────────────────────────────────────────────────────
def get_finger_states(landmarks, handedness: str) -> list[int]:
    tips = [4, 8, 12, 16, 20]
    pip  = [3, 6, 10, 14, 18]
    states = []
    if handedness == "Right":
        states.append(1 if landmarks[tips[0]].x < landmarks[pip[0]].x else 0)
    else:
        states.append(1 if landmarks[tips[0]].x > landmarks[pip[0]].x else 0)
    for i in range(1, 5):
        states.append(1 if landmarks[tips[i]].y < landmarks[pip[i]].y else 0)
    return states

def classify_gesture(landmarks, handedness: str) -> int | None:
    states = get_finger_states(landmarks, handedness)
    thumb, index, middle, ring, pinky = states
    total_extended = sum(states)

    if index == 1 and total_extended == 1:                                              return 1
    if index == 1 and middle == 1 and total_extended == 2:                              return 2
    if index == 1 and middle == 1 and ring == 1 and total_extended == 3:                return 3
    if index == 1 and middle == 1 and ring == 1 and pinky == 1 and total_extended == 4: return 4
    if total_extended == 5:                                                              return 5
    if thumb == 1 and pinky == 1 and total_extended == 2:                               return 6
    if thumb == 1 and total_extended == 1:                                               return 7
    if thumb == 1 and index == 1 and total_extended == 2:                               return 8
    if index == 1 and pinky == 1 and middle == 0 and ring == 0:                         return 9
    if total_extended == 0:                                                              return 10
    return None


# ──────────────────────────────────────────────────────────────────
# HEAD TILT
# ──────────────────────────────────────────────────────────────────
def head_tilt_direction(face_lm) -> str:
    left_eye  = face_lm[33]
    right_eye = face_lm[263]
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    angle_deg = np.degrees(np.arctan2(dy, dx))
    if   angle_deg >  cfg.TILT_THRESHOLD_DEG: return "LEFT"
    elif angle_deg < -cfg.TILT_THRESHOLD_DEG: return "RIGHT"
    return "NEUTRAL"


# ──────────────────────────────────────────────────────────────────
# FACE MASK  — full 478-point Delaunay warp
# ──────────────────────────────────────────────────────────────────
def load_masks_from_folder(folder):
    masks = []

    files = glob.glob(os.path.join(folder, "*.png"))

    for path in files:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None or img.shape[2] != 4:
            continue  # must have alpha channel

        name = os.path.splitext(os.path.basename(path))[0]

        masks.append({
            "name": name,
            "img": img
        })

    return masks

def overlay_png(
    frame: np.ndarray,
    png: np.ndarray,
    x: int,
    y: int
) -> np.ndarray:

    fh, fw = frame.shape[:2]
    ph, pw = png.shape[:2]

    # Clip to screen
    if x >= fw or y >= fh or x + pw <= 0 or y + ph <= 0:
        return frame

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + pw, fw)
    y2 = min(y + ph, fh)

    px1 = x1 - x
    py1 = y1 - y
    px2 = px1 + (x2 - x1)
    py2 = py1 + (y2 - y1)

    alpha = png[py1:py2, px1:px2, 3] / 255.0
    alpha = alpha[..., None]

    frame[y1:y2, x1:x2] = (
        alpha * png[py1:py2, px1:px2, :3] +
        (1 - alpha) * frame[y1:y2, x1:x2]
    ).astype(np.uint8)

    return frame

def apply_face_mask(
    frame: np.ndarray,
    mask_img: np.ndarray,
    face_landmarks
) -> np.ndarray:

    h, w = frame.shape[:2]

    # Key landmarks
    left_eye  = face_landmarks[33]
    right_eye = face_landmarks[263]

    left  = np.array([int(left_eye.x * w), int(left_eye.y * h)])
    right = np.array([int(right_eye.x * w), int(right_eye.y * h)])

    # --- CENTER FIX ---
    eye_center = (left + right) // 2
    eye_dist = np.linalg.norm(right - left)

    # Move slightly down from eyes to center face
    center = eye_center + np.array([0, int(eye_dist * 0.15)])

    # --- SCALE ---
    mask_h, mask_w = mask_img.shape[:2]
    scale = eye_dist / mask_w * 2.2

    new_w = int(mask_w * scale)
    new_h = int(mask_h * scale)

    resized = cv2.resize(mask_img, (new_w, new_h))

    # --- ROTATION FIX ---
    dx = right[0] - left[0]
    dy = right[1] - left[1]

    angle = -np.degrees(np.arctan2(dy, dx))  # <-- FIXED

    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)

    rotated = cv2.warpAffine(
        resized,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # --- POSITION ---
    x = int(center[0] - new_w / 2)
    y = int(center[1] - new_h / 2)

    return overlay_png(frame, rotated, x, y)


# ──────────────────────────────────────────────────────────────────
# OVERLAY DRAWING
# ──────────────────────────────────────────────────────────────────
PANEL_W = 240

def draw_overlay(frame, detected_gesture, gesture_confirmed, tilt_dir,
                 tilt_progress, current_spell_num, cast_label, cast_ts,
                 mask_name):
    h, w = frame.shape[:2]
    panel_x = w - PANEL_W

    bg = frame.copy()
    cv2.rectangle(bg, (panel_x, 0), (w, h), (8, 8, 16), -1)
    cv2.addWeighted(bg, 0.78, frame, 0.22, 0, frame)

    px = panel_x + 12
    cv2.putText(frame, "INVOKER CTRL", (px, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 200, 255), 1)
    cv2.line(frame, (panel_x, 38), (w, 38), (40, 40, 60), 1)

    for num in range(1, 11):
        name, spheres = cfg.SPELLS[num]
        sy  = 46 + (num - 1) * 47
        col = cfg.SPELL_COLORS[num]
        is_detected  = (num == detected_gesture)
        is_confirmed = (num == gesture_confirmed)
        is_active    = (num == current_spell_num)

        row_bg = (30, 30, 50)
        if is_confirmed or is_active:
            row_bg = tuple(min(int(c * 0.4), 255) for c in col)
        elif is_detected:
            row_bg = (38, 38, 62)

        cv2.rectangle(frame, (panel_x + 2, sy), (w - 2, sy + 40), row_bg, -1)
        if is_detected or is_confirmed or is_active:
            cv2.rectangle(frame, (panel_x + 2, sy), (w - 2, sy + 40), col, 1)

        badge_col = col if (is_detected or is_active) else (55, 55, 75)
        cv2.circle(frame, (px + 12, sy + 20), 12, badge_col, -1)
        nx = px + 7 if num < 10 else px + 5
        cv2.putText(frame, str(num), (nx, sy + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0) if (is_detected or is_active) else (130, 130, 150), 1)

        txt_col = col if (is_detected or is_active) else (170, 170, 190)
        cv2.putText(frame, name, (px + 28, sy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1)

        sx0 = px + 28
        for s in spheres:
            sc = (80, 100, 220) if s == cfg.Q else \
                 (80, 200,  80) if s == cfg.W else \
                 (30,  80, 220)
            cv2.circle(frame, (sx0, sy + 30), 5, sc, -1)
            sx0 += 13

        if is_active:
            cv2.putText(frame, "INVOKED", (panel_x + PANEL_W - 70, sy + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

    # Gesture bar
    bar_y = h - 100
    if detected_gesture:
        col   = cfg.SPELL_COLORS.get(detected_gesture, (100, 100, 100))
        bar_w = panel_x - 20
        cv2.rectangle(frame, (10, bar_y), (panel_x - 10, bar_y + 16), (25, 25, 40), -1)
        cv2.rectangle(frame, (10, bar_y), (10 + bar_w,   bar_y + 16), col, -1)
        cv2.putText(frame, f"Gesture {detected_gesture}  —  {cfg.SPELLS[detected_gesture][0]}",
                    (12, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 220), 1)

    # Tilt bar
    tilt_y  = h - 70
    tilt_cx = (panel_x - 10) // 2
    tilt_hw = 120

    cv2.rectangle(frame, (tilt_cx - tilt_hw, tilt_y),
                          (tilt_cx + tilt_hw, tilt_y + 20), (25, 25, 40), -1)
    cv2.line(frame, (tilt_cx, tilt_y - 2), (tilt_cx, tilt_y + 22), (80, 80, 100), 1)

    if tilt_dir == "LEFT":
        bar_px = int(tilt_hw * tilt_progress)
        cv2.rectangle(frame, (tilt_cx - bar_px, tilt_y),
                              (tilt_cx,           tilt_y + 20), (80, 200, 120), -1)
        cv2.putText(frame, "D", (tilt_cx - tilt_hw - 20, tilt_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 120), 2)
    elif tilt_dir == "RIGHT":
        bar_px = int(tilt_hw * tilt_progress)
        cv2.rectangle(frame, (tilt_cx,            tilt_y),
                              (tilt_cx + bar_px,   tilt_y + 20), (80, 140, 240), -1)
        cv2.putText(frame, "F", (tilt_cx + tilt_hw + 6, tilt_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 140, 240), 2)

    cv2.putText(frame, "TILT", (tilt_cx - 16, tilt_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 140), 1)

    if detected_gesture:
        col = cfg.SPELL_COLORS.get(detected_gesture, (255, 255, 255))
        cv2.putText(frame, str(detected_gesture), (10, 75),
                    cv2.FONT_HERSHEY_COMPLEX, 2.8, col, 4)
        cv2.putText(frame, cfg.SPELLS[detected_gesture][0], (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

    if cast_label and time.time() - cast_ts < 2.0:
        cv2.putText(frame, f"CAST  {cast_label}", (10, h - 16),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 180), 2)

    # Active mask name bottom-right of camera area
    if mask_name != "none":
        cv2.putText(frame, f"mask: {mask_name}", (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 160), 1)

    return frame

def draw_hand_landmarks(frame, hand_lm, mp_drawing, mp_hands):
    mp_drawing.draw_landmarks(
        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 220, 255), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0, 100, 180), thickness=2),
    )


# ──────────────────────────────────────────────────────────────────
# PIP WINDOW
# ──────────────────────────────────────────────────────────────────
PIP_W, PIP_H = 360, 270
PIP_NAME     = "cam"

def build_pip(frame, tilt_dir, current_spell_num, mask_name):
    pip = cv2.resize(frame, (PIP_W, PIP_H))
    cv2.rectangle(pip, (0, PIP_H - 28), (PIP_W, PIP_H), (8, 8, 16), -1)

    spell_label = cfg.SPELLS[current_spell_num][0] if current_spell_num else "—"
    cv2.putText(pip, spell_label, (8, PIP_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    tilt_sym = {"LEFT": "<  D", "RIGHT": "F  >", "NEUTRAL": ""}[tilt_dir]
    if tilt_sym:
        col = (80, 200, 120) if tilt_dir == "LEFT" else (80, 140, 240)
        cv2.putText(pip, tilt_sym, (PIP_W - 70, PIP_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    cv2.putText(pip, mask_name, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    return pip


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
def main():
    global current_spell, last_cast_label, last_cast_ts

    last_tilt_time = 0.0
    gesture_ready  = True
    mask_idx       = 0

    mp_hands     = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing   = mp.solutions.drawing_utils

    masks = load_masks_from_folder("masks")
    mask_names = ["none"] + [m["name"] for m in masks]
    print(f"[MASK] {len(masks)} mask(s) available. Press M to cycle.")

    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {cfg.CAMERA_INDEX}.")
        return

    cv2.namedWindow(PIP_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PIP_NAME, PIP_W, PIP_H)
    cv2.setWindowProperty(PIP_NAME, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(PIP_NAME, 1520, 20)

    print("=" * 50)
    print(" Invoker Controller running")
    print(" M = cycle mask  |  Q / ESC = quit")
    print("=" * 50)

    gesture_history   = deque(maxlen=cfg.GESTURE_HOLD_FRAMES)
    tilt_history      = deque(maxlen=cfg.TILT_HOLD_FRAMES)
    gesture_confirmed = None

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            max_num_hands=1) as hands, \
         mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Frame capture failed.")
                break

            if cfg.FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            # fh, fw = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            # ── Face mask ─────────────────────────────────────────
            active_mask_name = "none"

            if mask_idx > 0:
                active_mask_name = mask_names[mask_idx]

            if face_results.multi_face_landmarks:
                face_lm_all = face_results.multi_face_landmarks[0].landmark

                if mask_idx > 0:
                    mask_data = masks[mask_idx - 1]
                    frame = apply_face_mask(frame, mask_data["img"], face_lm_all)

            # ── Hand gesture ──────────────────────────────────────
            detected_gesture  = None
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

            if detected_gesture is not None and \
               len(gesture_history) == cfg.GESTURE_HOLD_FRAMES:
                count            = gesture_history.count(detected_gesture)
                gesture_progress = count / cfg.GESTURE_HOLD_FRAMES

                if gesture_progress >= cfg.GESTURE_TRIGGER_THRESHOLD and gesture_ready:
                    gesture_confirmed = detected_gesture
                    gesture_ready     = False
                elif gesture_progress <= cfg.GESTURE_RESET_THRESHOLD:
                    gesture_ready = True

            # ── Head tilt ─────────────────────────────────────────
            tilt_dir       = "NEUTRAL"
            tilt_progress  = 0.0
            tilt_confirmed = False

            if face_results.multi_face_landmarks:
                face_lm_all = face_results.multi_face_landmarks[0].landmark
                tilt_dir    = head_tilt_direction(face_lm_all)

            tilt_history.append(tilt_dir)

            if tilt_dir in ("LEFT", "RIGHT") and \
               len(tilt_history) == cfg.TILT_HOLD_FRAMES:
                count         = tilt_history.count(tilt_dir)
                tilt_progress = count / cfg.TILT_HOLD_FRAMES
                if tilt_progress >= 0.85:
                    tilt_confirmed = True

            # ── Dota integration ──────────────────────────────────
            if gesture_confirmed:
                keys = list(cfg.SPELLS[gesture_confirmed][1]) + [cfg.KEY_INVOKE]
                threading.Thread(target=press_keys, args=(keys,), daemon=True).start()
                with state_lock:
                    current_spell   = gesture_confirmed
                    last_cast_label = cfg.SPELLS[gesture_confirmed][0]
                    last_cast_ts    = time.time()
                print(f"[INVOKE]  {cfg.SPELLS[gesture_confirmed][0]}")

            now = time.time()
            if tilt_confirmed and now - last_tilt_time > cfg.CAST_COOLDOWN_SEC:
                slot_key  = cfg.KEY_SLOT2 if tilt_dir == "LEFT"  else cfg.KEY_SLOT1
                slot_name = "D"           if tilt_dir == "LEFT"  else "F"
                press_keys([slot_key])
                last_tilt_time = now
                with state_lock:
                    spell_name      = cfg.SPELLS[current_spell][0] if current_spell else "?"
                    last_cast_label = f"{spell_name} → {slot_name}"
                    last_cast_ts    = now
                print(f"[CAST]    {spell_name}  →  {slot_name}")

            # ── Draw ──────────────────────────────────────────────
            with state_lock:
                _spell   = current_spell
                _label   = last_cast_label
                _cast_ts = last_cast_ts

            frame = draw_overlay(
                frame, detected_gesture, gesture_confirmed,
                tilt_dir, tilt_progress, _spell, _label, _cast_ts,
                active_mask_name,
            )
            cv2.imshow("Invoker Controller  —  ESC / Q to quit", frame)

            pip = build_pip(frame, tilt_dir, _spell, active_mask_name)
            cv2.imshow(PIP_NAME, pip)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('m'), ord('M')):
                mask_idx = (mask_idx + 1) % len(mask_names)
                print(f"[MASK] Active: {mask_names[mask_idx]}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()

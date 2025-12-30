"""
YOLOv8 navigation + obstacle‑avoidance demo
=========================================

Adds **three major behaviours** on top of the basic “approach highest‑confidence
object” logic:

1. **Visual track overlay** – two converging lines (guiding corridor) are
   rendered on the live preview so operators can see the path the robot tries
   to follow.
2. **Reactive obstacle stop/avoid** – if any non‑target object gets too close,
   the robot: (a) brakes, (b) backs up a short distance, (c) turns away and
   skirts around the obstacle before resuming target pursuit.
3. **Find‑back** – if the target disappears from view, the robot remembers on
   which side it was last seen and slowly pans that way until it reacquires
   the target.

Only the computation of linear (**v**) and angular (**w**) velocities has
changed; the UDP message format to the embedded controller remains
`!{v:.2f}@{w:.2f}#`.
"""

from __future__ import annotations

import sys
import time
import socket
from typing import List, Tuple
from enum import Enum, auto

import cv2
import torch
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
DEFAULT_CLASS = "chair"
CAM_INDEX = 0
JETSON_IP = "10.5.144.120"
JETSON_PORT = 8888

# Velocity limits (tune to your platform)
MAX_V_FWD = 0.30   # m/s   forward
MAX_V_BACK = -0.12 # m/s   reverse
MAX_W = 0.16       # rad/s yaw
MAX_W_AVOID = 0.08 # rad/s 

# Detection heuristics
GROUND_Y_FRAC = 0.5        # consider lower‑half objects as ground obstacles
OBST_NEAR_FRAC = 0.10      # obstacle “near” if area ≥ img_area×OBST_NEAR_FRAC
OBST_STOP_FRAC = 0.33      # obstacle “too close” (immediate stop)
TARGET_FAR_FRAC = 0.10     # target small ⇒ far ⇒ fast
TARGET_CLOSE_FRAC = 0.33   # target large ⇒ close ⇒ slow

# Behaviour timing (frames at ~30 FPS)
BACKUP_FRAMES = 60         # frames to reverse on obstacle contact
AVOID_FRAMES = 100          # frames to turn/creep around obstacle
LOST_FRAMES_THRESH = 60    # frames w/o target before find‑back engages

# Visual guide line anchors (fractions of width/height)
GUIDE_LEFT_BASE_X  = 0.15  # bottom–centre slightly left
GUIDE_RIGHT_BASE_X = 0.85
GUIDE_TOP_X_OFFSET = 0.05  # converge toward horizon centre
GUIDE_TOP_Y = 0.3          # relative height of vanishing point

# ──────────────────────────────────────────────────────────────────────
# Helper types
# ──────────────────────────────────────────────────────────────────────
BBox = Tuple[float, float, float, float, float, str]  # (x1,y1,x2,y2,conf,label)

class Mode(Enum):
    SEEK = auto()         # pursue target
    BACKUP = auto()       # reverse a bit after impact
    AVOID = auto()        # sidestep obstacle
    FIND = auto()         # rotate to rediscover target

# ──────────────────────────────────────────────────────────────────────
# State machine logic
# ──────────────────────────────────────────────────────────────────────
class Navigator:
    def __init__(self):
        self.mode = Mode.SEEK
        self.timer = 0         # generic frame counter for timed states
        self.turn_dir = 0      # −1 left, +1 right
        self.last_target_side = 1  # remember last seen side: default to left
        self.lost_frames = 0

    # ---------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------
    def _in_ground_band(self, box: BBox, img_h: int) -> bool:
        _, y1, _, y2, _, _ = box
        centre_y = (y1 + y2) / 2
        return centre_y >= img_h * GROUND_Y_FRAC

    def _box_area_frac(self, box: BBox, img_area: int) -> float:
        x1, y1, x2, y2, _, _ = box
        return ((x2 - x1) * (y2 - y1)) / img_area

    # ---------------------------------------------------------------
    # Main interface
    # ---------------------------------------------------------------
    def step(self, target_boxes: List[BBox], obstacles: List[BBox],
             img_w: int, img_h: int) -> Tuple[float, float]:
        """Return (v, w) each frame based on detections + current mode."""
        img_area = img_w * img_h
        v = 0.0
        w = 0.0

        # ------------------------------------------------------- state transitions
        if self.mode == Mode.SEEK:
            # if any obstacle dangerously close ⇒ BACKUP
            danger = self._closest_obstacle(obstacles, img_area, img_h)
            if danger and self._box_area_frac(danger, img_area) >= OBST_STOP_FRAC:
                self.mode = Mode.BACKUP
                self.timer = BACKUP_FRAMES
                # Turn opposite to obstacle side later
                ox1, _, ox2, _, _, _ = danger
                self.turn_dir = 1 if (ox1 + ox2) / 2 < img_w / 2 else -1
            else:
                # pursue target normally
                v, w = self._seek_control(target_boxes, img_w, img_h, img_area)
                # target bookkeeping
                if target_boxes:
                    centre_x = (target_boxes[0][0] + target_boxes[0][2]) / 2
                    self.last_target_side = -1 if centre_x < img_w / 2 else 1
                    self.lost_frames = 0
                else:
                    self.lost_frames += 1
                    if self.lost_frames > LOST_FRAMES_THRESH:
                        self.mode = Mode.FIND
        elif self.mode == Mode.BACKUP:
            v = MAX_V_BACK
            w = 0.0
            self.timer -= 1
            if self.timer <= 0:
                self.mode = Mode.AVOID
                self.timer = AVOID_FRAMES
        elif self.mode == Mode.AVOID:
            v = 0.10  # creep forward gently
            w = self.turn_dir * MAX_W_AVOID
            if target_boxes:
                centre_x = (target_boxes[0][0] + target_boxes[0][2]) / 2
                self.last_target_side = -1 if centre_x < img_w / 2 else 1
            # If timer expires or obstacle not near any more ⇒ SEEK
            self.timer -= 1
            if self.timer <= 0:
                self.mode = Mode.SEEK
        elif self.mode == Mode.FIND:
            v = 0.10
            dir_ = self.last_target_side  # default rotate right: 1
            w = dir_ * 0.08
            if target_boxes:
                self.mode = Mode.SEEK
                self.lost_frames = 0

        # Clamp final outputs
        v = max(MAX_V_BACK, min(v, MAX_V_FWD))
        w = max(-MAX_W, min(w, MAX_W))
        return v, w

    # ---------------------------------------------------------------
    # Internal controls
    # ---------------------------------------------------------------
    def _closest_obstacle(self, obstacles: List[BBox], img_area: int, img_h: int) -> BBox | None:
        """Return largest ground obstacle (area proxy)."""
        ground_obs = [b for b in obstacles if self._in_ground_band(b, img_h)]
        if not ground_obs:
            return None
        return max(ground_obs, key=lambda b: self._box_area_frac(b, img_area))

    def _near_obstacle(self, obstacles: List[BBox], img_area: int, img_h: int) -> bool:
        obs = self._closest_obstacle(obstacles, img_area, img_h)
        return bool(obs and self._box_area_frac(obs, img_area) >= OBST_NEAR_FRAC)

    def _seek_control(self, target_boxes: List[BBox], img_w: int, img_h: int, img_area: int) -> Tuple[float, float]:
        """Attractive‑only P‑controller toward best target."""
        if not target_boxes:
            return 0.0, 0.0

        x1, y1, x2, y2, conf, _ = target_boxes[0]
        centre_x = (x1 + x2) / 2
        offset_x = centre_x - img_w / 2
        w = (offset_x / (img_w / 2)) * MAX_W
        w = max(-MAX_W, min(w, MAX_W))

        area_frac = ((x2 - x1) * (y2 - y1)) / img_area
        if area_frac < TARGET_FAR_FRAC:
            v = MAX_V_FWD
        elif area_frac < TARGET_CLOSE_FRAC:
            v = 0.10
        else:
            v = 0.0
        return v, w

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Parse CLI arg
    target_class = DEFAULT_CLASS
    for arg in sys.argv[1:]:
        if arg.lower().startswith("class="):
            target_class = arg.split("=", 1)[1].strip().lower()

    model = YOLO("yolov8m.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    coco_names = {n.lower() for n in model.names.values()}
    if target_class not in coco_names:
        print(f"[WARN] class '{target_class}' not in COCO. Using '{DEFAULT_CLASS}'.")
        target_class = DEFAULT_CLASS

    nav = Navigator()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_h, img_w = frame.shape[:2]

            res = model.predict(frame, conf=0.25, device=device, verbose=False)[0]
            target_boxes: List[BBox] = []
            obstacle_boxes: List[BBox] = []
            for b in sorted(res.boxes, key=lambda bb: float(bb.conf[0]), reverse=True):
                cls_id = int(b.cls[0])
                label = res.names[cls_id].lower()
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(float, b.xyxy[0])
                record: BBox = (x1, y1, x2, y2, conf, label)
                if label == target_class:
                    target_boxes.append(record)
                else:
                    obstacle_boxes.append(record)

            # navigator decides what to do
            v, w = nav.step(target_boxes, obstacle_boxes, img_w, img_h)
            cmd = f"!{v:.2f}@{w:.2f}#"
            sock.sendto(cmd.encode(), (JETSON_IP, JETSON_PORT))
            print(f"Mode={nav.mode.name} v={v:.2f} w={w:.2f}", end="\r")

            # ───────── visualisation ─────────
            # draw guide track (two converging lines)
            l_base = (int(img_w * GUIDE_LEFT_BASE_X), img_h)
            r_base = (int(img_w * GUIDE_RIGHT_BASE_X), img_h)
            l_van_pt = (int(img_w * 0.35), int(img_h * GUIDE_TOP_Y))
            r_van_pt = (int(img_w * 0.65), int(img_h * GUIDE_TOP_Y))
            cv2.line(frame, l_base, l_van_pt, (255, 255, 0), 2)
            cv2.line(frame, r_base, r_van_pt, (255, 255, 0), 2)

            # targets / obstacles boxes
            for (x1, y1, x2, y2, conf, _) in target_boxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            for (x1, y1, x2, y2, conf, _) in obstacle_boxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

            cv2.putText(frame, f"v={v:.2f} w={w:.2f} mode={nav.mode.name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Nav Preview", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC quits
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

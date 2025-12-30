"""
yolov8m_class=.py
Run:  python yolov8m_class=.py               # default class = chair
       python yolov8m_class=.py class=human  # track “person/human”
"""

import sys
import cv2
import torch
import socket
from ultralytics import YOLO

# check cuda availability
print("CUDA available:", torch.cuda.is_available())

# Connect to Jetson Nano ip through wifi
jetson_ip = '10.5.144.120'
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------------------------------------
# 1) Parse command–line argument  ------------------
# -------------------------------------------------
default_class = "chair"
target_class = default_class

# Accept syntax   class=human
for arg in sys.argv[1:]:
    if arg.lower().startswith("class="):
        target_class = arg.split("=", 1)[1].strip().lower()

# -------------------------------------------------
# 2) Load model & verify the requested class  -----
# -------------------------------------------------
model = YOLO("yolov8m.pt")                 # COCO-pretrained (80 classes)
model.to("cuda")                          # move model to cuda
coco_names = {name.lower() for name in model.names.values()}  # {'person', 'car', ...}

if target_class not in coco_names:
    print(
        f"[WARN] '{target_class}' is not in YOLOv8-COCO's 80 classes.\n"
        f"       Falling back to '{default_class}'.\n"
        f"       Valid choices include: {', '.join(sorted(coco_names))}"
    )
    target_class = default_class

print(f"[INFO] Tracking class ──▶  {target_class}")

# -------------------------------------------------
# 3) Helper: map detections to (v, w)  ------------
# -------------------------------------------------
def compute_control_signals(bboxes, img_w, img_h):
    if not bboxes:
        return 0.0, 0.0

    x1, y1, x2, y2, conf, label = bboxes[0]  # use highest-conf box (already sorted)
    bx_center_x = (x1 + x2) / 2
    offset_x    = bx_center_x - (img_w / 2)

    # Angular velocity  w  ∈ [-0.3, 0.3]
    max_w = 0.16
    w = (offset_x / (img_w / 2)) * max_w
    w = max(-max_w, min(w, max_w))

    # Linear velocity  v  ∈ [0, 0.3] (slow when object is close/large)
    max_v = 0.3
    box_area     = (x2 - x1) * (y2 - y1)
    area_thresh_maxspeed  = (img_w * img_h) / 10  # heuristic
    area_thresh_midspeed = (img_w * img_h) / 3
    if box_area < area_thresh_maxspeed:
        v = 0.3
    elif box_area < area_thresh_midspeed:
        v = 0.1
    else:
        v = 0
    v = max(0.0, min(v, max_v))

    return v, w

# -------------------------------------------------
# 4) Main loop  -----------------------------------
# -------------------------------------------------
cap = cv2.VideoCapture(0)   # 0 = default webcam; or path to video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model.predict(frame, conf=0.25, verbose=False, device='cuda')[0]

    # Keep detections matching the requested class
    bboxes = []
    for box in sorted(results.boxes, key=lambda b: float(b.conf[0]), reverse=True):
        cls_id = int(box.cls[0])
        label  = results.names[cls_id].lower()
        conf   = float(box.conf[0])

        if label == target_class:
            x1, y1, x2, y2 = box.xyxy[0]
            bboxes.append((x1, y1, x2, y2, conf, label))

    # Compute motion commands
    img_h, img_w = frame.shape[:2]
    v, w = compute_control_signals(bboxes, img_w, img_h)
    print(f"v = {v:.2f}  |  w = {w:.2f}", end="\r")

    # Send v and w to Jetson Nano
    command = f"!{v:.2f}@{w:.2f}#"
    print(command)
    sock.sendto(command.encode(), (jetson_ip, 8888))
    

    # Draw detections
    for (x1, y1, x2, y2, conf, label) in bboxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(frame, f"v={v:.2f}  w={w:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("YOLOv8 Robot Demo", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

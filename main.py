#!/usr/bin/env python3
"""
Real-time object tracking with YOLOv8 + OpenCV
Author: Neal Rao (slicedtorso)
Dependencies: opencv-python, opencv-contrib-python, ultralytics, numpy
Run:
    pip install opencv-python opencv-contrib-python ultralytics numpy
    python main.py --source 0
"""

import argparse
import os
import time
from collections import deque

import cv2
import numpy as np

# Try importing YOLO
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("[WARN] Ultralytics not available. Install with: pip install ultralytics")

# -------------------------- Utilities --------------------------

def iou_xyxy(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0: return 0.0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / (areaA + areaB - interArea + 1e-6)

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [int(x), int(y), int(x+w), int(y+h)]

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(x2-x1), int(y2-y1)]

def clamp_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    return [x1, y1, x2, y2]

def make_color(seed):
    np.random.seed(seed)
    return tuple(int(x) for x in np.random.randint(64, 255, size=3))

def create_tracker(tracker_name='CSRT'):
    name = tracker_name.upper()
    if name == 'CSRT':
        return cv2.legacy.TrackerCSRT_create()
    if name == 'KCF':
        return cv2.legacy.TrackerKCF_create()
    if name == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()
    raise ValueError(f"Unknown tracker: {tracker_name}")

# -------------------------- Track Class --------------------------

class Track:
    def __init__(self, tid, label, bbox_xyxy, tracker_type='CSRT', color=None):
        self.id = tid
        self.label = label
        self.tracker_type = tracker_type
        self.tracker = create_tracker(tracker_type)
        self.init_ok = False
        self.color = color if color else make_color(tid)
        self.history = deque(maxlen=20)
        self.bbox_xyxy = bbox_xyxy

    def init(self, frame):
        x, y, w, h = xyxy_to_xywh(self.bbox_xyxy)
        self.init_ok = self.tracker.init(frame, (x, y, w, h))
        return self.init_ok

    def update(self, frame):
        ok, box = self.tracker.update(frame)
        if not ok:
            return False, self.bbox_xyxy
        x, y, w, h = [int(v) for v in box]
        self.bbox_xyxy = [x, y, x+w, y+h]
        cx, cy = x + w//2, y + h//2
        self.history.append((cx, cy))
        return True, self.bbox_xyxy

# -------------------------- Multi Object Tracker --------------------------

class MultiObjectTracker:
    def __init__(self, tracker_type='CSRT', iou_match_thresh=0.3, max_age=30):
        self.tracker_type = tracker_type
        self.tracks = {}
        self.next_id = 1
        self.iou_match_thresh = iou_match_thresh
        self.ages = {}
        self.max_age = max_age

    def reset(self):
        self.tracks.clear()
        self.ages.clear()
        self.next_id = 1

    def _new_track(self, frame, label, bbox_xyxy):
        tid = self.next_id
        self.next_id += 1
        tr = Track(tid, label, bbox_xyxy, tracker_type=self.tracker_type)
        tr.init(frame)
        self.tracks[tid] = tr
        self.ages[tid] = 0

    def update_with_trackers(self, frame):
        dead = []
        for tid, tr in self.tracks.items():
            ok, _ = tr.update(frame)
            self.ages[tid] = 0 if ok else self.ages[tid]+1
            if self.ages[tid] > self.max_age:
                dead.append(tid)
        for tid in dead:
            self.tracks.pop(tid, None)
            self.ages.pop(tid, None)

    def _match_dets_to_tracks(self, dets_xyxy, det_labels):
        matches = []
        unmatched_dets = list(range(len(dets_xyxy)))
        unmatched_tracks = list(self.tracks.keys())
        if len(unmatched_dets) == 0 or len(unmatched_tracks) == 0:
            return matches, unmatched_dets, unmatched_tracks

        iou_matrix = np.zeros((len(unmatched_dets), len(unmatched_tracks)), dtype=np.float32)
        for i, d_idx in enumerate(unmatched_dets):
            d_box = dets_xyxy[d_idx]
            for j, t_id in enumerate(unmatched_tracks):
                t_box = self.tracks[t_id].bbox_xyxy
                iou_matrix[i, j] = iou_xyxy(d_box, t_box)

        used_dets, used_tracks = set(), set()
        while True:
            if iou_matrix.size == 0: break
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[i, j] < self.iou_match_thresh: break
            d_idx, t_id = unmatched_dets[i], unmatched_tracks[j]
            if d_idx in used_dets or t_id in used_tracks:
                iou_matrix[i, j] = -1
                continue
            matches.append((d_idx, t_id))
            used_dets.add(d_idx)
            used_tracks.add(t_id)
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        unmatched_dets_final = [d for d in unmatched_dets if d not in used_dets]
        unmatched_tracks_final = [t for t in unmatched_tracks if t not in used_tracks]
        return matches, unmatched_dets_final, unmatched_tracks_final

    def update_with_detections(self, frame, dets_xyxy, det_labels):
        matches, unmatched_dets, unmatched_tracks = self._match_dets_to_tracks(dets_xyxy, det_labels)
        for d_idx, t_id in matches:
            tr = self.tracks[t_id]
            tr.label = det_labels[d_idx]
            tr.bbox_xyxy = dets_xyxy[d_idx]
            tr.tracker = create_tracker(self.tracker_type)
            tr.init(frame)
            self.ages[t_id] = 0
        for d_idx in unmatched_dets:
            self._new_track(frame, det_labels[d_idx], dets_xyxy[d_idx])
        for t_id in unmatched_tracks:
            self.ages[t_id] += 1

    def draw(self, frame, show_trails=True):
        for tid, tr in self.tracks.items():
            x1, y1, x2, y2 = tr.bbox_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), tr.color, 2)
            label = f"ID {tid}: {tr.label}"
            cv2.putText(frame, label, (x1, max(0, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tr.color, 2)
            if show_trails and len(tr.history) >= 2:
                pts = np.array(tr.history, dtype=np.int32).reshape(-1,1,2)
                cv2.polylines(frame, [pts], isClosed=False, color=tr.color, thickness=2)

# -------------------------- YOLO Detection --------------------------

def yolo_detect(model, frame, score_thr=0.6):
    results = model(frame, verbose=False)
    dets, labels = [], []
    if len(results) == 0 or results[0].boxes is None: return dets, labels
    for b in results[0].boxes:
        conf = float(b.conf[0]) if b.conf is not None else 1.0
        if conf < score_thr: continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls = int(b.cls[0])
        label = model.names.get(cls, str(cls)) if hasattr(model, "names") else str(cls)
        dets.append([x1, y1, x2, y2])
        labels.append(label)
    return dets, labels

# -------------------------- Manual ROI --------------------------

def manual_roi_track(frame, mot):
    bbox = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyWindow("Select ROI")
    if bbox == (0,0,0,0): return
    x, y, w, h = bbox
    mot._new_track(frame, "manual", [int(x), int(y), int(x+w), int(y+h)])

# -------------------------- Main Loop --------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--tracker", type=str, default="CSRT", choices=["CSRT","KCF","MOSSE"])
    return ap.parse_args()

def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): return print("[ERR] Could not open video source")

    if YOLO is None: return print("[ERR] YOLO not installed")

    model = YOLO(args.model)
    mot = MultiObjectTracker(tracker_type=args.tracker)

    frame_idx, auto_detect = 0, True
    snapshots_dir = "snapshots"
    os.makedirs(snapshots_dir, exist_ok=True)
    fps_time, fps = time.time(), 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break
        H, W = frame.shape[:2]

        if auto_detect and frame_idx % 20 == 0:
            # Resize frame for faster YOLO inference
            small_frame = cv2.resize(frame, (W//2, H//2))
            dets, labels = yolo_detect(model, small_frame)
            # Scale detections back to original size
            dets = [[b[0]*2, b[1]*2, b[2]*2, b[3]*2] for b in dets]
            # Limit number of detections to top 10
            if len(dets) > 10:
                dets = dets[:10]
                labels = labels[:10]
            if dets:
                dets = [clamp_box_xyxy(b,W,H) for b in dets]
                mot.update_with_detections(frame, dets, labels)

        mot.update_with_trackers(frame)
        mot.draw(frame)

        now = time.time()
        fps = 1.0/(now-fps_time) if now-fps_time>0 else fps
        fps_time = now

        cv2.putText(frame, f"FPS:{fps:.1f} Tracks:{len(mot.tracks)} Auto:{auto_detect}",
                    (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2)

        cv2.imshow("Track + Identify", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('a'): auto_detect = not auto_detect
        elif key == ord('r'): mot.reset()
        elif key == ord('c'): manual_roi_track(frame, mot)
        elif key == ord('s'):
            ts = int(time.time())
            out_path = os.path.join(snapshots_dir,f"frame_{ts}.jpg")
            cv2.imwrite(out_path, frame)
            print(f"[INFO] Saved snapshot: {out_path}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

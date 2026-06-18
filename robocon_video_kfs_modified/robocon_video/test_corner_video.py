"""
test_corner_video.py  –  same diagnostic as test_corner.py but loads
a single video frame instead of a PDF page.

Usage:
    python test_corner_video.py                          # uses defaults below
    python test_corner_video.py --video my.mp4 --ts 3.5 # specific timestamp
"""

import argparse
import cv2
import numpy as np
from run_kfs_video import (verify_corner_slopes, preprocess, detect_corners,
                           deduplicate, select_dominant_kfs, CFG)

parser = argparse.ArgumentParser()
parser.add_argument('--video', default=CFG['video_path'],
                    help='Path to video file')
parser.add_argument('--ts', type=float, default=0.0,
                    help='Timestamp (seconds) of the frame to inspect')
args = parser.parse_args()

print(f'Loading frame at t={args.ts:.2f}s from: {args.video}')
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f'Cannot open video: {args.video}')

native_fps = cap.get(cv2.CAP_PROP_FPS)
target_frame = int(round(args.ts * native_fps))
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
ret, bgr = cap.read()
cap.release()

if not ret:
    raise SystemExit(f'Could not read frame at t={args.ts:.2f}s '
                     f'(frame index {target_frame})')

gray, binary = preprocess(bgr, CFG['stroke_thresh'], CFG['close_kernel'])
groups = detect_corners(binary, CFG['angle_thresh'], CFG['min_area'], CFG['epsilon_frac'])
dominant = select_dominant_kfs(groups, bgr.shape, CFG.get('center_weight', 0.3))

print(f'Found {len(groups)} KFS contour(s) in frame.')
if dominant is None:
    raise SystemExit('No KFS above min_area in this frame.')
print(f'Dominant KFS bbox={dominant["bbox"]}  area={dominant["area"]:.0f}  '
      f'score={dominant.get("score", 0):.3f}')

corners = deduplicate(dominant['pts'], CFG['dedup_dist'])

print(f'Dominant KFS has {len(corners)} corner candidates — showing first 5:\n')
for cx, cy, ang in corners[:5]:
    wp, valid = verify_corner_slopes(
        binary, cx, cy, CFG['circle_radius'], CFG['n_samples'], CFG['angle_thresh'])
    print(f'  Point ({cx}, {cy})  angle={ang:.1f}°  white_pct={wp:.2f}  valid={valid}')

    n_samples = CFG['n_samples']
    angles_arr = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    xs = (cx + CFG['circle_radius'] * np.cos(angles_arr)).astype(int)
    ys = (cy + CFG['circle_radius'] * np.sin(angles_arr)).astype(int)
    vals = []
    for x, y in zip(xs, ys):
        if 0 <= x < binary.shape[1] and 0 <= y < binary.shape[0]:
            vals.append(binary[y, x])
    dark_indices = [i for i, v in enumerate(vals) if v <= 127]
    print(f'    Dark sample indices: {dark_indices}\n')

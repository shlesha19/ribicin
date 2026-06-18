"""
KFS Detection — video input edition.

Instead of a PDF, takes a video file, extracts frames at a configurable
sample rate (frames-per-second), then runs the same KFS corner/anchor
detection on each frame.

Usage
-----
    python run_kfs_video.py                          # uses CFG defaults
    python run_kfs_video.py --video path/to/vid.mp4
    python run_kfs_video.py --video vid.mp4 --fps 2 --out ./my_output
"""

import argparse
import os
import sys
import warnings

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    # --- VIDEO SETTINGS (replaces pdf_path / dpi) ---
    'video_path'    : 'input_video.mp4',   # path to your video file
    'sample_fps'    : 1.0,                 # frames to extract per second of video
                                           #   1.0  → 1 frame / second
                                           #   0.5  → 1 frame every 2 seconds
                                           #   5.0  → 5 frames / second
    'skip_first_n'  : 0,                   # skip the first N extracted frames
                                           #   (mirrors the "skip cover page" logic)

    # --- OUTPUT ---
    'out_dir'       : 'output_frames',

    # --- DETECTION (unchanged from PDF version) ---
    'stroke_thresh' : 80,
    'angle_thresh'  : 100,
    'epsilon_frac'  : 0.008,
    'circle_radius' : 20,
    'n_samples'     : 100,
    'white_pct_min' : 0.70,
    'white_pct_max' : 0.80,
    'count_thresh'  : 5,
    'min_area'      : 16000,
    'dedup_dist'    : 5,
    'close_kernel'  : 15,
    'center_weight' : 0.3,   # weight of frame-center proximity vs area when
                             # picking the dominant KFS (0 = pure area).
                             # On the real rig, replace area with depth.
}

# ── Video → frames ────────────────────────────────────────────────────────────

def extract_frames(video_path: str, sample_fps: float) -> list[tuple[int, float, np.ndarray]]:
    """
    Open *video_path* and return a list of (frame_index, timestamp_sec, bgr_array)
    tuples sampled at *sample_fps* frames per second of video duration.

    Parameters
    ----------
    video_path : str
        Path to any video format OpenCV can open (mp4, avi, mkv, mov, …).
    sample_fps : float
        How many frames to extract per second of video.
        Values < native FPS subsample; values > native FPS are capped.

    Returns
    -------
    List of (frame_index, timestamp_sec, bgr_ndarray) sorted by timestamp.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / native_fps if native_fps > 0 else 0

    print(f"Video info: {total_frames} frames @ {native_fps:.2f} fps  "
          f"({duration_sec:.1f} s)")

    # interval between sampled frames (in native frames)
    interval = max(1, int(round(native_fps / sample_fps)))
    print(f"Sampling every {interval} frames  (~{sample_fps} frame/s requested)\n")

    frames = []
    frame_idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            ts = frame_idx / native_fps if native_fps > 0 else frame_idx
            frames.append((frame_idx, ts, bgr))
        frame_idx += 1

    cap.release()
    return frames


# ── Detection helpers (copied verbatim from run_kfs_detection.py) ─────────────

def preprocess(bgr, stroke_thresh=80, close_k=5):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, stroke_thresh, 255, cv2.THRESH_BINARY)
    stroke_mask = cv2.bitwise_not(binary)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    stroke_closed = cv2.morphologyEx(stroke_mask, cv2.MORPH_CLOSE, k)
    return gray, cv2.bitwise_not(stroke_closed)


def verify_corner_slopes(binary, cx, cy, radius, n_samples=64, angle_thresh=100):
    h, w = binary.shape
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    xs = (cx + radius * np.cos(angles)).astype(int)
    ys = (cy + radius * np.sin(angles)).astype(int)

    vals = []
    white_count = 0
    for x, y in zip(xs, ys):
        val = binary[y, x] if (0 <= x < w and 0 <= y < h) else 255
        vals.append(val)
        if val > 127:
            white_count += 1

    white_pct = white_count / n_samples
    dark_indices = [i for i, v in enumerate(vals) if v <= 127]
    if not dark_indices:
        return white_pct, False

    segments, current_segment = [], [dark_indices[0]]
    for i in range(1, len(dark_indices)):
        if dark_indices[i] == dark_indices[i - 1] + 1:
            current_segment.append(dark_indices[i])
        else:
            segments.append(current_segment)
            current_segment = [dark_indices[i]]
    segments.append(current_segment)

    if (len(segments) > 1
            and segments[0][0] == 0
            and segments[-1][-1] == n_samples - 1):
        segments[0] = segments[-1] + segments[0]
        segments.pop()

    valid_slope = False
    if len(segments) == 1:
        width_deg = len(segments[0]) * 360 / n_samples
        if width_deg <= angle_thresh + 15:
            valid_slope = True
    elif len(segments) >= 2:
        segment_angles = []
        for seg in segments:
            sin_sum = sum(np.sin(angles[idx % n_samples]) for idx in seg)
            cos_sum = sum(np.cos(angles[idx % n_samples]) for idx in seg)
            segment_angles.append(np.arctan2(sin_sum, cos_sum))

        n_seg = len(segment_angles)
        for i in range(n_seg):
            for j in range(i + 1, n_seg):
                diff = np.degrees(abs(segment_angles[i] - segment_angles[j]))
                if diff > 180:
                    diff = 360 - diff
                if diff <= angle_thresh + 15:
                    valid_slope = True
                    break
            if valid_slope:
                break

    return white_pct, valid_slope


def detect_corners(binary, angle_thresh=155, min_area=4000, eps_frac=0.008):
    """
    Detect sharp-corner candidates, GROUPED BY their parent contour.

    A video frame may contain several KFS sheets at once. Each KFS is a
    separate external contour, so instead of flattening every corner into one
    list (the old frame-wide behaviour) we keep them grouped per contour.
    Each group carries the geometry we later use to decide which KFS is the
    dominant one (largest / closest):

        {
          'pts'      : [(x, y, angle), ...],   # corner candidates on this KFS
          'area'     : float,                  # contour area in px²
          'bbox'     : (x, y, w, h),           # bounding box
          'centroid' : (cx, cy),               # contour centroid
        }

    Returns
    -------
    List of such group dicts (one per accepted contour).
    """
    stroke_mask = cv2.bitwise_not(binary)
    # RETR_EXTERNAL: one contour per physical KFS sheet (ignore inner holes),
    # which is what we want when separating multiple KFS in a frame.
    contours, _ = cv2.findContours(stroke_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    groups = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps_frac * peri, True)
        verts = approx[:, 0, :]
        n = len(verts)
        if n < 3:
            continue

        pts = []
        for i in range(n):
            p0 = verts[i].astype(float)
            p1 = verts[(i - 1) % n].astype(float)
            p2 = verts[(i + 1) % n].astype(float)
            v1, v2 = p1 - p0, p2 - p0
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1 or n2 < 1:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            ang = np.degrees(np.arccos(cos_a))
            if ang < angle_thresh:
                pts.append((int(p0[0]), int(p0[1]), float(ang)))

        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            ctr = (M['m10'] / M['m00'], M['m01'] / M['m00'])
        else:
            ctr = (x + w / 2.0, y + h / 2.0)

        groups.append({
            'pts'      : pts,
            'area'     : float(area),
            'bbox'     : (x, y, w, h),
            'centroid' : ctr,
        })
    return groups


def select_dominant_kfs(groups, frame_shape, center_weight=0.3):
    """
    Pick the single KFS to classify when several are present in the frame.

    On the real robot a depth camera will tell us which KFS is physically
    closest. In this video-only pipeline we approximate "closest" with two
    image cues that both correlate with proximity:

      1. AREA  — a nearer KFS projects to a larger area in the frame
                 (dominant signal).
      2. CENTER PROXIMITY — the KFS the robot is driving toward tends to sit
                 near the optical center; we mildly reward being close to the
                 frame center so a large sheet at the far edge doesn't beat a
                 slightly smaller one the robot is actually facing.

    score = normalized_area * (1 - center_weight)
          + center_proximity * center_weight

    With a real depth stream you would replace `area` by the inverse of the
    median depth inside each contour's mask and drop the area proxy entirely;
    the selection logic is otherwise identical.

    Returns the winning group dict, or None if `groups` is empty.
    """
    if not groups:
        return None

    h, w = frame_shape[:2]
    fcx, fcy = w / 2.0, h / 2.0
    max_dist = np.hypot(fcx, fcy)            # corner-to-center distance
    max_area = max(g['area'] for g in groups)

    best, best_score = None, -1.0
    for g in groups:
        area_norm = g['area'] / max_area if max_area > 0 else 0.0
        cx, cy = g['centroid']
        dist = np.hypot(cx - fcx, cy - fcy)
        center_prox = 1.0 - (dist / max_dist if max_dist > 0 else 0.0)
        score = area_norm * (1.0 - center_weight) + center_prox * center_weight
        g['score'] = score          # stash for debugging / overlay
        if score > best_score:
            best, best_score = g, score
    return best


def deduplicate(pts, min_dist=30):
    if not pts:
        return []
    pts_sorted = sorted(pts, key=lambda p: p[2])
    kept = []
    for p in pts_sorted:
        px, py = p[0], p[1]
        if all(abs(px - k[0]) > min_dist or abs(py - k[1]) > min_dist
               for k in kept):
            kept.append(p)
    return kept


def classify_kfs(bgr, cfg=CFG):
    gray, binary = preprocess(bgr, cfg['stroke_thresh'], cfg['close_kernel'])

    # Per-KFS grouping (one group per contour/sheet in the frame).
    groups = detect_corners(binary, cfg['angle_thresh'],
                            cfg['min_area'], cfg['epsilon_frac'])

    # Pick the dominant KFS: largest / closest (depth proxy).
    dominant = select_dominant_kfs(
        groups, bgr.shape, cfg.get('center_weight', 0.3))

    if dominant is None:
        debug = bgr.copy()
        cv2.putText(debug, 'NO KFS DETECTED',
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 6)
        return dict(label='NONE', reason='no contour above min_area',
                    valid_count=0, total_candidates=0,
                    valid_pts=[], invalid_pts=[], binary=binary, debug_img=debug,
                    n_kfs=len(groups), dominant_bbox=None)

    # Deduplicate corners WITHIN the dominant KFS only.
    corners = deduplicate(dominant['pts'], cfg['dedup_dist'])

    debug = bgr.copy()

    # Draw the non-dominant KFS dimly (gray box) for context.
    for g in groups:
        if g is dominant:
            continue
        x, y, w, h = g['bbox']
        cv2.rectangle(debug, (x, y), (x + w, y + h), (120, 120, 120), 2)
        cv2.putText(debug, 'ignored KFS', (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

    # Highlight the dominant KFS bounding box.
    dx, dy, dw, dh = dominant['bbox']
    cv2.rectangle(debug, (dx, dy), (dx + dw, dy + dh), (255, 200, 0), 3)
    cv2.putText(debug, 'TARGET KFS (largest/closest)', (dx + 5, dy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    if len(corners) == 0:
        cv2.putText(debug, 'REAL (smooth / 0 sharp edges)',
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 200, 0), 6)
        return dict(label='REAL', reason='smooth – 0 sharp edges on target KFS',
                    valid_count=0, total_candidates=0,
                    valid_pts=[], invalid_pts=[], binary=binary, debug_img=debug,
                    n_kfs=len(groups), dominant_bbox=dominant['bbox'])

    valid_pts, invalid_pts = [], []
    r      = cfg['circle_radius']
    thr_min, thr_max = cfg['white_pct_min'], cfg['white_pct_max']
    ns     = cfg['n_samples']

    for (cx, cy, ang) in corners:
        wp, valid_slope = verify_corner_slopes(binary, cx, cy, r, ns,
                                               cfg['angle_thresh'])
        if thr_min <= wp <= thr_max and valid_slope:
            valid_pts.append((cx, cy, ang, wp))
        else:
            invalid_pts.append((cx, cy, ang, wp))

    valid_count = len(valid_pts)
    ct    = cfg['count_thresh']
    label = 'REAL' if valid_count > ct else 'FAKE'
    color = (0, 210, 0) if label == 'REAL' else (0, 0, 220)
    reason = (f'valid pts {valid_count} > {ct}'
              if label == 'REAL'
              else f'valid pts {valid_count} <= {ct}')

    for (cx, cy, ang, wp) in invalid_pts:
        cv2.circle(debug, (cx, cy), r, (0, 0, 200), 2)
        cv2.circle(debug, (cx, cy), 6, (0, 0, 200), -1)
        cv2.putText(debug, f'{wp*100:.0f}%', (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 180), 2)
    for (cx, cy, ang, wp) in valid_pts:
        cv2.circle(debug, (cx, cy), r, (0, 210, 0), 4)
        cv2.circle(debug, (cx, cy), 8, (0, 210, 0), -1)
        cv2.putText(debug, f'{wp*100:.0f}%', (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 160, 0), 2)

    txt = (f'{label}   valid={valid_count}/{len(corners)}  '
           f'(need >{ct})   KFS in frame={len(groups)}')
    cv2.putText(debug, txt, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 5)

    return dict(label=label, reason=reason,
                valid_count=valid_count, total_candidates=len(corners),
                valid_pts=valid_pts, invalid_pts=invalid_pts,
                binary=binary, debug_img=debug,
                n_kfs=len(groups), dominant_bbox=dominant['bbox'])


# ── Save annotated figure for one frame ──────────────────────────────────────

def save_frame_figure(bgr, result, frame_idx: int, timestamp: float,
                      out_dir: str) -> str:
    """Save a 3-panel figure (original | binary | annotated) for one frame."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original  (t={timestamp:.2f}s)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(result['binary'], cmap='gray')
    axes[1].set_title('Binary  (white=bg, black=stroke)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(result['debug_img'], cv2.COLOR_BGR2RGB))
    lbl  = result['label']
    vcnt = result['valid_count']
    tcnt = result['total_candidates']
    clr  = 'green' if lbl == 'REAL' else 'red'
    axes[2].set_title(f'Result: {lbl}   valid={vcnt}/{tcnt}',
                      fontsize=13, fontweight='bold', color=clr)
    axes[2].axis('off')

    patches = [
        mpatches.Patch(color='lime', label='Valid anchor (>=50% white)'),
        mpatches.Patch(color='red',  label='Invalid anchor (<50% white)'),
    ]
    axes[2].legend(handles=patches, loc='lower right', fontsize=9)

    fig.suptitle(
        f'Frame {frame_idx}  (t={timestamp:.2f}s)  →  {lbl}   '
        f'(valid anchors={vcnt}/{tcnt})',
        fontsize=15, fontweight='bold',
        color='darkgreen' if lbl == 'REAL' else 'darkred')

    plt.tight_layout()
    fname = f'frame_{frame_idx:06d}_t{timestamp:.2f}s_{lbl}.png'
    path  = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Summary chart ─────────────────────────────────────────────────────────────

def save_summary_chart(results: list, out_dir: str) -> str:
    """Bar chart of valid-anchor counts per frame with REAL/FAKE colouring."""
    timestamps   = [ts  for _, ts, r in results]
    valid_counts = [r['valid_count'] for _, _, r in results]
    labels       = [r['label']       for _, _, r in results]
    colors       = ['green' if l == 'REAL' else 'red' if l == 'FAKE'
                    else 'gray' for l in labels]

    fig, ax = plt.subplots(figsize=(max(14, len(results) * 0.6), 5))
    bars = ax.bar(timestamps, valid_counts, color=colors,
                  edgecolor='black', alpha=0.8, width=0.8)

    threshold = CFG['count_thresh']
    ax.axhline(threshold, color='blue', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold}')
    ax.set_xlabel('Timestamp (s)', fontsize=12)
    ax.set_ylabel('Valid anchor points (≥50% white surround)', fontsize=12)
    ax.set_title('KFS Detection — Anchor Point Count per Frame',
                 fontsize=14, fontweight='bold')

    for bar, lbl, vc in zip(bars, labels, valid_counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f'{lbl}\n({vc})',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

    gp = mpatches.Patch(color='green', label='REAL')
    rp = mpatches.Patch(color='red',   label='FAKE')
    ax.legend(
        handles=[gp, rp,
                 plt.Line2D([0], [0], color='blue', linestyle='--',
                            label=f'Threshold >{threshold}')],
        fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, 'summary_chart.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='KFS real/fake detection on video frames')
    parser.add_argument('--video', default=None,
                        help='Path to input video (overrides CFG video_path)')
    parser.add_argument('--fps',   type=float, default=None,
                        help='Frames to sample per second (overrides CFG sample_fps)')
    parser.add_argument('--out',   default=None,
                        help='Output directory (overrides CFG out_dir)')
    parser.add_argument('--skip',  type=int, default=None,
                        help='Skip first N extracted frames (overrides CFG skip_first_n)')
    args = parser.parse_args()

    # Allow CLI overrides
    if args.video is not None:
        CFG['video_path']  = args.video
    if args.fps is not None:
        CFG['sample_fps']  = args.fps
    if args.out is not None:
        CFG['out_dir']     = args.out
    if args.skip is not None:
        CFG['skip_first_n'] = args.skip

    os.makedirs(CFG['out_dir'], exist_ok=True)

    # ── Step 1: extract frames ────────────────────────────────────────────────
    print(f'Loading video: {CFG["video_path"]}')
    all_frames = extract_frames(CFG['video_path'], CFG['sample_fps'])

    skip = CFG['skip_first_n']
    frames = all_frames[skip:]
    print(f'Total extracted frames: {len(all_frames)}  '
          f'(skipping first {skip})  →  processing {len(frames)}\n')

    if not frames:
        sys.exit('[ERROR] No frames to process after skip.')

    # ── Step 2: detect on each frame ─────────────────────────────────────────
    results = []   # list of (frame_idx, timestamp, result_dict)
    for frame_idx, ts, bgr in frames:
        r = classify_kfs(bgr, CFG)
        results.append((frame_idx, ts, r))

        path = save_frame_figure(bgr, r, frame_idx, ts, CFG['out_dir'])
        if r['label'] == 'REAL':
            sym = '✅ REAL'
        elif r['label'] == 'FAKE':
            sym = '❌ FAKE'
        else:
            sym = '⚪ NONE'
        print(f"  Frame {frame_idx:>6}  t={ts:>7.2f}s  {sym}  "
              f"valid={r['valid_count']:>3}/{r['total_candidates']:<3}  "
              f"KFS={r.get('n_kfs', 0)}  reason: {r['reason']}")

    # ── Step 3: summary ───────────────────────────────────────────────────────
    chart_path = save_summary_chart(results, CFG['out_dir'])

    real_n = sum(1 for _, _, r in results if r['label'] == 'REAL')
    fake_n = sum(1 for _, _, r in results if r['label'] == 'FAKE')
    none_n = len(results) - real_n - fake_n

    print(f'\n{"─" * 60}')
    print(f'SUMMARY: {real_n} REAL  /  {fake_n} FAKE  /  {none_n} NONE  '
          f'out of {len(results)} frames')
    print(f'Output frames  : {CFG["out_dir"]}/')
    print(f'Summary chart  : {chart_path}')


if __name__ == '__main__':
    main()

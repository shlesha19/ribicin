"""
kfs_detection.py — single-file KFS detection pipeline for ABU Robocon 2026.

Input: Intel RealSense .bag recordings (color + aligned depth), read via
pyrealsense2. Depth is used both to pick the dominant (closest) KFS and to
localise it.

Stages (each its own class):
    KFSConfig            central tunables
    RealsenseBagSource   .bag -> sampled (color, depth) frames
    ImageFlattener       color frame -> binary stroke image
    ObjectDetector       binary -> KFS sheets (KFSGroup) + depth-based dominant
    RealFakeClassifier   anchors -> REAL / FAKE / NONE
    Localizer            real KFS -> forest block (1..12) + depth, stored to disk
    KFSPipeline          orchestrates all of the above

Usage
-----
    python kfs_detection.py --bag rec.bag --fps 2 --out ./out
    # or as a library:
    from kfs_detection import KFSConfig, KFSPipeline
    KFSPipeline(KFSConfig(bag_path="rec.bag")).run()

Requires: pyrealsense2, opencv-python, numpy
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None   # checked at runtime in RealsenseBagSource

warnings.filterwarnings("ignore")


# -- Config -------------------------------------------------------------------
@dataclass
class KFSConfig:
    # --- input (.bag) ---
    bag_path: str = "input.bag"
    sample_fps: float = 1.0       # frames to keep per second of recording
    skip_first_n: int = 0
    out_dir: str = "output_frames"

    # --- depth ---
    depth_min_m: float = 0.1      # ignore depth readings outside [min, max] (meters)
    depth_max_m: float = 6.0

    # --- preprocessing / flattening ---
    stroke_thresh: int = 80
    close_kernel: int = 15

    # --- object (KFS sheet) detection ---
    min_area: float = 16000
    epsilon_frac: float = 0.008
    angle_thresh: float = 100

    # --- real/fake anchor verification ---
    circle_radius: int = 20
    n_samples: int = 100
    white_pct_min: float = 0.70
    white_pct_max: float = 0.80
    dedup_dist: int = 5
    count_thresh: int = 5         # valid anchors > this => REAL


# -- RealSense .bag input -----------------------------------------------------
class RealsenseBagSource:
    """
    Reads an Intel RealSense .bag and yields
    (frame_idx, timestamp_sec, color_bgr, depth_m) tuples sampled at
    cfg.sample_fps. `depth_m` is a float32 array of metric depth (meters),
    aligned to the color frame; 0.0 means no reading.
    """

    def __init__(self, cfg: KFSConfig):
        self.cfg = cfg
        if rs is None:
            sys.exit("[ERROR] pyrealsense2 not installed. `pip install pyrealsense2`")

    def frames(self):
        c = self.cfg
        if not os.path.exists(c.bag_path):
            sys.exit(f"[ERROR] Cannot find bag: {c.bag_path}")

        pipe = rs.pipeline()
        cfg = rs.config()
        rs.config.enable_device_from_file(cfg, c.bag_path, repeat_playback=False)
        cfg.enable_stream(rs.stream.color)
        cfg.enable_stream(rs.stream.depth)

        profile = pipe.start(cfg)
        dev = profile.get_device().as_playback()
        dev.set_real_time(False)                 # process every frame, no drops

        depth_scale = (profile.get_device()
                       .first_depth_sensor().get_depth_scale())   # raw units -> meters
        align = rs.align(rs.stream.color)        # align depth to color

        print(f"Bag: {c.bag_path}  depth_scale={depth_scale:.5f} m/unit")

        out = []
        idx, kept, skipped = 0, 0, 0
        last_ts = -1.0
        min_dt = 1.0 / c.sample_fps if c.sample_fps > 0 else 0.0

        try:
            while True:
                try:
                    frames = pipe.wait_for_frames(5000)
                except RuntimeError:
                    break   # end of file
                frames = align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color or not depth:
                    idx += 1
                    continue

                ts = frames.get_timestamp() / 1000.0   # ms -> s
                idx += 1

                # subsample to ~sample_fps using timestamps
                if last_ts >= 0 and (ts - last_ts) < min_dt:
                    continue
                last_ts = ts

                if skipped < c.skip_first_n:
                    skipped += 1
                    continue

                color_bgr = np.asanyarray(color.get_data())            # HxWx3 BGR
                depth_raw = np.asanyarray(depth.get_data()).astype(np.float32)
                depth_m = depth_raw * depth_scale                      # meters
                depth_m[(depth_m < c.depth_min_m) | (depth_m > c.depth_max_m)] = 0.0

                out.append((idx, ts, color_bgr.copy(), depth_m))
                kept += 1
        finally:
            pipe.stop()

        # normalise timestamps to start at 0
        if out:
            t0 = out[0][1]
            out = [(i, t - t0, cc, d) for (i, t, cc, d) in out]
        print(f"Kept {kept} frames (skipped first {skipped})")
        return out


# -- Image flattening ---------------------------------------------------------
class ImageFlattener:
    """Turns a raw BGR frame into a clean binary image (white=bg, black=stroke)."""

    def __init__(self, cfg: KFSConfig):
        self.cfg = cfg

    def flatten(self, bgr):
        """Return (gray, binary). `binary` has white background, black strokes."""
        c = self.cfg
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, c.stroke_thresh, 255, cv2.THRESH_BINARY)
        stroke_mask = cv2.bitwise_not(binary)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (c.close_kernel, c.close_kernel))
        stroke_closed = cv2.morphologyEx(stroke_mask, cv2.MORPH_CLOSE, k)
        return gray, cv2.bitwise_not(stroke_closed)


# -- Object detection ---------------------------------------------------------
class KFSGroup:
    """One detected KFS sheet (contour) with corner candidates, geometry, depth."""

    __slots__ = ("pts", "area", "bbox", "centroid", "contour", "depth_m", "score")

    def __init__(self, pts, area, bbox, centroid, contour):
        self.pts = pts            # [(x, y, angle), ...]
        self.area = area
        self.bbox = bbox          # (x, y, w, h)
        self.centroid = centroid  # (cx, cy)
        self.contour = contour    # raw cv2 contour, for depth masking
        self.depth_m = None       # median metric depth inside the sheet
        self.score = 0.0


class ObjectDetector:
    """Detects KFS sheets (external contours) and their sharp-corner candidates."""

    def __init__(self, cfg: KFSConfig):
        self.cfg = cfg

    def detect(self, binary):
        """Return a list of KFSGroup, one per accepted contour."""
        c = self.cfg
        stroke_mask = cv2.bitwise_not(binary)
        contours, _ = cv2.findContours(stroke_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        groups = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < c.min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            verts = cv2.approxPolyDP(cnt, c.epsilon_frac * peri, True)[:, 0, :]
            if len(verts) < 3:
                continue

            pts = self._corner_candidates(verts, c.angle_thresh)
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            ctr = ((M["m10"] / M["m00"], M["m01"] / M["m00"])
                   if M["m00"] > 0 else (x + w / 2.0, y + h / 2.0))
            groups.append(KFSGroup(pts, float(area), (x, y, w, h), ctr, cnt))
        return groups

    @staticmethod
    def _corner_candidates(verts, angle_thresh):
        n = len(verts)
        pts = []
        for i in range(n):
            p0 = verts[i].astype(float)
            v1 = verts[(i - 1) % n].astype(float) - p0
            v2 = verts[(i + 1) % n].astype(float) - p0
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1 or n2 < 1:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            ang = np.degrees(np.arccos(cos_a))
            if ang < angle_thresh:
                pts.append((int(p0[0]), int(p0[1]), float(ang)))
        return pts

    def measure_depth(self, group, depth_m):
        """Median metric depth of valid (nonzero) pixels inside the contour."""
        mask = np.zeros(depth_m.shape, np.uint8)
        cv2.drawContours(mask, [group.contour], -1, 255, cv2.FILLED)
        vals = depth_m[(mask > 0) & (depth_m > 0)]
        group.depth_m = float(np.median(vals)) if vals.size else None
        return group.depth_m

    def select_dominant(self, groups, depth_m):
        """
        Pick the KFS the robot is facing = the physically CLOSEST sheet,
        i.e. the one with the smallest median depth. Groups with no valid
        depth reading are skipped; if none have depth, fall back to largest area.
        """
        if not groups:
            return None
        for g in groups:
            self.measure_depth(g, depth_m)

        with_depth = [g for g in groups if g.depth_m is not None]
        if with_depth:
            best = min(with_depth, key=lambda g: g.depth_m)
            best.score = -best.depth_m   # closer = higher score
            return best
        # fallback: no depth anywhere -> largest sheet
        return max(groups, key=lambda g: g.area)

    @staticmethod
    def deduplicate(pts, min_dist):
        """Drop corner candidates closer than min_dist to a kept point."""
        if not pts:
            return []
        kept = []
        for p in sorted(pts, key=lambda p: p[2]):
            if all(abs(p[0] - k[0]) > min_dist or abs(p[1] - k[1]) > min_dist
                   for k in kept):
                kept.append(p)
        return kept


# -- Real/Fake classification -------------------------------------------------
class RealFakeClassifier:
    """Verifies sharp-corner anchors and decides REAL / FAKE / NONE."""

    def __init__(self, cfg: KFSConfig):
        self.cfg = cfg

    def verify_corner(self, binary, cx, cy):
        """Return (white_pct, valid_slope) for the anchor ring around (cx, cy)."""
        c = self.cfg
        h, w = binary.shape
        angles = np.linspace(0, 2 * np.pi, c.n_samples, endpoint=False)
        xs = (cx + c.circle_radius * np.cos(angles)).astype(int)
        ys = (cy + c.circle_radius * np.sin(angles)).astype(int)

        vals = [binary[y, x] if (0 <= x < w and 0 <= y < h) else 255
                for x, y in zip(xs, ys)]
        white_pct = sum(v > 127 for v in vals) / c.n_samples
        dark = [i for i, v in enumerate(vals) if v <= 127]
        if not dark:
            return white_pct, False

        segments = self._segments(dark, c.n_samples)
        return white_pct, self._valid_slope(segments, angles, c)

    @staticmethod
    def _segments(dark, n_samples):
        segs, cur = [], [dark[0]]
        for i in range(1, len(dark)):
            if dark[i] == dark[i - 1] + 1:
                cur.append(dark[i])
            else:
                segs.append(cur)
                cur = [dark[i]]
        segs.append(cur)
        # wrap-around merge
        if len(segs) > 1 and segs[0][0] == 0 and segs[-1][-1] == n_samples - 1:
            segs[0] = segs[-1] + segs[0]
            segs.pop()
        return segs

    @staticmethod
    def _valid_slope(segments, angles, c):
        if len(segments) == 1:
            width_deg = len(segments[0]) * 360 / c.n_samples
            return width_deg <= c.angle_thresh + 15

        seg_angles = []
        for seg in segments:
            sin_sum = sum(np.sin(angles[i % c.n_samples]) for i in seg)
            cos_sum = sum(np.cos(angles[i % c.n_samples]) for i in seg)
            seg_angles.append(np.arctan2(sin_sum, cos_sum))

        n = len(seg_angles)
        for i in range(n):
            for j in range(i + 1, n):
                diff = np.degrees(abs(seg_angles[i] - seg_angles[j]))
                if diff > 180:
                    diff = 360 - diff
                if diff <= c.angle_thresh + 15:
                    return True
        return False

    def classify(self, binary, dominant):
        """
        Classify the dominant KFS group.
        Returns dict(label, valid_count, total_candidates, valid_pts, invalid_pts, reason).
        """
        c = self.cfg
        corners = ObjectDetector.deduplicate(dominant.pts, c.dedup_dist)
        if not corners:
            return dict(label="REAL", valid_count=0, total_candidates=0,
                        valid_pts=[], invalid_pts=[],
                        reason="smooth - 0 sharp edges on target KFS")

        valid_pts, invalid_pts = [], []
        for cx, cy, ang in corners:
            wp, valid = self.verify_corner(binary, cx, cy)
            row = (cx, cy, ang, wp)
            if c.white_pct_min <= wp <= c.white_pct_max and valid:
                valid_pts.append(row)
            else:
                invalid_pts.append(row)

        vc = len(valid_pts)
        label = "REAL" if vc > c.count_thresh else "FAKE"
        return dict(label=label, valid_count=vc, total_candidates=len(corners),
                    valid_pts=valid_pts, invalid_pts=invalid_pts,
                    reason=f"valid pts {vc} {'>' if label == 'REAL' else '<='} {c.count_thresh}")


# -- Localisation -------------------------------------------------------------
@dataclass
class RealKFSRecord:
    block: int                 # forest block index 1..12
    centroid: tuple            # (cx, cy) in color-frame pixels
    depth_m: float             # metric depth of the KFS sheet
    frame_idx: int
    timestamp: float
    valid_count: int


@dataclass
class Localizer:
    """
    Maps a detected real KFS to a Meihua-Forest block and stores it.
    The Forest is a 3-wide x 4-tall grid of 12 blocks per side (rulebook section 1).
    On the real rig, swap block_from_centroid for a camera-pose/odometry
    transform; depth_m is already metric and ready for that.
    """
    grid_cols: int = 3
    grid_rows: int = 4
    records: dict = field(default_factory=dict)   # block -> RealKFSRecord

    def block_from_centroid(self, centroid, frame_shape):
        """Map a centroid to a forest block (1..12), row-major top-left = 1."""
        h, w = frame_shape[:2]
        cx, cy = centroid
        col = min(int(cx / w * self.grid_cols), self.grid_cols - 1)
        row = min(int(cy / h * self.grid_rows), self.grid_rows - 1)
        return row * self.grid_cols + col + 1

    def register(self, result, dominant, frame_shape, frame_idx, timestamp):
        """If `result` is REAL, record its block + depth. Returns block or None."""
        if result["label"] != "REAL":
            return None
        block = self.block_from_centroid(dominant.centroid, frame_shape)
        self.records[block] = RealKFSRecord(
            block=block,
            centroid=tuple(round(v, 1) for v in dominant.centroid),
            depth_m=round(dominant.depth_m, 3) if dominant.depth_m else None,
            frame_idx=frame_idx,
            timestamp=round(timestamp, 2),
            valid_count=result["valid_count"],
        )
        return block

    def real_blocks(self):
        """Sorted list of forest blocks currently holding a confirmed real KFS."""
        return sorted(self.records)

    def save(self, path):
        data = {str(b): asdict(r) for b, r in sorted(self.records.items())}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path


# -- Pipeline orchestrator ----------------------------------------------------
class KFSPipeline:
    def __init__(self, cfg: KFSConfig = None):
        self.cfg = cfg or KFSConfig()
        self.source = RealsenseBagSource(self.cfg)
        self.flattener = ImageFlattener(self.cfg)
        self.detector = ObjectDetector(self.cfg)
        self.classifier = RealFakeClassifier(self.cfg)
        self.localizer = Localizer()

    def process_frame(self, bgr, depth_m, frame_idx=0, timestamp=0.0):
        """Run the full per-frame chain. Returns the classification result dict."""
        _, binary = self.flattener.flatten(bgr)
        groups = self.detector.detect(binary)
        dominant = self.detector.select_dominant(groups, depth_m)

        if dominant is None:
            return dict(label="NONE", valid_count=0, total_candidates=0,
                        valid_pts=[], invalid_pts=[], n_kfs=0, block=None,
                        depth_m=None, reason="no contour above min_area")

        result = self.classifier.classify(binary, dominant)
        result["n_kfs"] = len(groups)
        result["depth_m"] = dominant.depth_m
        result["block"] = self.localizer.register(
            result, dominant, bgr.shape, frame_idx, timestamp)
        return result

    def run(self):
        """Process the whole bag; returns list of (frame_idx, ts, result)."""
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        frames = self.source.frames()
        print(f"Processing {len(frames)} frames\n")

        results = []
        for frame_idx, ts, bgr, depth_m in frames:
            r = self.process_frame(bgr, depth_m, frame_idx, ts)
            results.append((frame_idx, ts, r))
            blk = f" block={r['block']}" if r.get("block") else ""
            dpt = f" depth={r['depth_m']:.2f}m" if r.get("depth_m") else ""
            print(f"  Frame {frame_idx:>6}  t={ts:>6.2f}s  {r['label']:<4}  "
                  f"valid={r['valid_count']:>3}/{r['total_candidates']:<3}"
                  f"  KFS={r.get('n_kfs', 0)}{dpt}{blk}")

        loc_path = self.localizer.save(
            os.path.join(self.cfg.out_dir, "real_kfs_locations.json"))
        print(f"\nReal KFS blocks: {self.localizer.real_blocks()}")
        print(f"Locations saved: {loc_path}")
        return results


# -- CLI ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="KFS real/fake detection on RealSense .bag recordings")
    p.add_argument("--bag")
    p.add_argument("--fps", type=float)
    p.add_argument("--out")
    p.add_argument("--skip", type=int)
    a = p.parse_args()

    cfg = KFSConfig()
    if a.bag is not None:  cfg.bag_path = a.bag
    if a.fps is not None:  cfg.sample_fps = a.fps
    if a.out is not None:  cfg.out_dir = a.out
    if a.skip is not None: cfg.skip_first_n = a.skip

    KFSPipeline(cfg).run()


if __name__ == "__main__":
    main()

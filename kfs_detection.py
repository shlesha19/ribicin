"""
KFS detection pipeline for Robocon 2026.

One class, `KFSDetector`, plus a small `RealSenseSource` helper. The bot scripts
just import this:

    from kfs_detection import KFSDetector, RealSenseSource

    det = KFSDetector()

    # testing: replay a recording that has color + depth
    src = RealSenseSource(testing=True, bag_path="clip.bag")
    # working: live camera feed
    # src = RealSenseSource(testing=False)

    det.run(src, "boxes.json")

Pipeline (per frame), each step is one public method:
    detect_boxes   -> crop each box (rgb + depth)
    flatten_faces  -> rectify each visible face to a head-on view
    classify_face  -> REAL / FAKE / NONE per face   (corner logic from prototypes)
    find_location  -> camera-relative (X, Y, Z) in meters
    save           -> JSON log / cache

Needs numpy + opencv always; pyrealsense2 only for live/.bag sources (imported
lazily so the 2D classify path still runs on a dev machine without the SDK).
"""

import json
from pathlib import Path

import cv2
import numpy as np


# ── Source: live RealSense or .bag playback ──────────────────────────────────
class RealSenseSource:
    """Yields aligned (rgb, depth, intrinsics, depth_scale) frames.

    testing=True replays `bag_path`; testing=False opens the live camera. Both
    use the same SDK, with depth aligned onto the color frame so a color pixel
    (u, v) has a matching depth value and intrinsics.

    `intrinsics` is a dict: {fx, fy, ppx, ppy}. `depth` is uint16 (raw units);
    multiply by `depth_scale` to get meters.
    """

    def __init__(self, testing=False, bag_path=None,
                 width=640, height=480, fps=30, sample_fps=None):
        import pyrealsense2 as rs   # lazy: only needed for a real source

        # how many frames per second to actually emit. For .bag testing we
        # default to 3 fps; live runs full rate unless a value is given.
        self.sample_fps = sample_fps if sample_fps is not None else (3.0 if testing else None)
        self._min_interval_ms = (1000.0 / self.sample_fps) if self.sample_fps else 0.0

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if testing:
            if not bag_path:
                raise ValueError("testing=True needs a bag_path")
            cfg.enable_device_from_file(bag_path, repeat_playback=False)
        else:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        profile = self.pipeline.start(cfg)
        if testing:
            # process every frame instead of dropping frames in real time
            profile.get_device().as_playback().set_real_time(False)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    def frame_records(self):
        """Generator of aligned frame records with source metadata.

        If `sample_fps` is set, frames are subsampled by timestamp to roughly
        that rate (e.g. 3 fps for .bag testing).
        """
        last_ts = None
        start_ts = None
        source_idx = 0
        try:
            while True:
                try:
                    frames = self.pipeline.wait_for_frames()
                except RuntimeError:
                    break  # end of .bag

                ts = frames.get_timestamp()
                if start_ts is None:
                    start_ts = ts
                timestamp_s = (ts - start_ts) / 1000.0

                frames = self.align.process(frames)
                depth_f = frames.get_depth_frame()
                color_f = frames.get_color_frame()
                if not depth_f or not color_f:
                    continue

                # subsample to sample_fps using the frame timestamp (ms)
                if self._min_interval_ms > 0:
                    if last_ts is not None and (ts - last_ts) < self._min_interval_ms:
                        source_idx += 1
                        continue
                    last_ts = ts

                i = color_f.profile.as_video_stream_profile().intrinsics
                intr = {"fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy}
                rgb = np.asanyarray(color_f.get_data())
                depth = np.asanyarray(depth_f.get_data())
                yield {
                    "rgb": rgb,
                    "depth": depth,
                    "intrinsics": intr,
                    "depth_scale": self.depth_scale,
                    "frame_index": source_idx,
                    "timestamp_s": timestamp_s,
                }
                source_idx += 1
        finally:
            self.close()

    def frames(self):
        """Generator of (rgb_bgr, depth_uint16, intrinsics, depth_scale)."""
        for rec in self.frame_records():
            yield rec["rgb"], rec["depth"], rec["intrinsics"], rec["depth_scale"]

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class KFSDetector:
    def __init__(self, **cfg):
        # --- box detection (red/blue colour + depth refinement), meters ---
        self.min_depth      = cfg.get("min_depth", 0.2)    # ignore closer than this
        self.max_depth      = cfg.get("max_depth", 3.0)    # ignore farther than this
        self.box_min_area   = cfg.get("box_min_area", 2000)  # px², min box blob
        self.color_min_area = cfg.get("color_min_area", 600)  # px², min red/blue blob
        self.min_color_extent = cfg.get("min_color_extent", 0.45)
        self.min_object_px  = cfg.get("min_object_px", 600)  # depth-refined object px
        self.min_seed_px    = cfg.get("min_seed_px", 40)    # red/blue px with depth
        self.box_pad        = cfg.get("box_pad", 10)        # px around colour bbox
        self.object_depth_tol = cfg.get("object_depth_tol", 0.20)  # m around box depth
        self.object_depth_frac = cfg.get("object_depth_frac", 0.12)

        # --- face flattening ---
        self.plane_tol      = cfg.get("plane_tol", 0.02)   # m, plane membership
        self.min_face_px    = cfg.get("min_face_px", 800)  # min pixels per face
        self.out_size       = cfg.get("out_size", 256)     # rectified face size (px)
        self.min_frame_clarity = cfg.get("min_frame_clarity", 120)
        self.min_box_clarity = cfg.get("min_box_clarity", 100)
        self.debug_max_lookahead_frames = cfg.get("debug_max_lookahead_frames", 30)
        self.min_symbol_area = cfg.get("min_symbol_area", 30)
        self.min_symbol_group_area = cfg.get("min_symbol_group_area", 90)
        self.symbol_border_margin = cfg.get("symbol_border_margin", 14)
        self.face_pad_frac = cfg.get("face_pad_frac", 0.45)
        self.symbol_group_close = cfg.get("symbol_group_close", 13)
        self.debug_verbose = cfg.get("debug_verbose", False)

        # --- real/fake (endpoint sharpness logic) ---
        self.stroke_thresh  = cfg.get("stroke_thresh", 80)
        self.angle_thresh   = cfg.get("angle_thresh", 100)
        self.epsilon_frac   = cfg.get("epsilon_frac", 0.008)
        self.circle_radius  = cfg.get("circle_radius", 20)
        self.n_samples      = cfg.get("n_samples", 100)
        self.white_min      = cfg.get("white_min", 0.70)
        self.white_max      = cfg.get("white_max", 0.80)
        self.count_thresh   = cfg.get("count_thresh", 3)
        self.kfs_min_area   = cfg.get("kfs_min_area", 4000)
        self.dedup_dist     = cfg.get("dedup_dist", 5)
        self.close_kernel   = cfg.get("close_kernel", 15)
        self.sharp_end_angle_thresh = cfg.get("sharp_end_angle_thresh", 95)
        self.endpoint_branch_ignore_radius = cfg.get("endpoint_branch_ignore_radius", 16)
        self.endpoint_tip_radius = cfg.get("endpoint_tip_radius", 10)
        self.endpoint_merge_radius = cfg.get("endpoint_merge_radius", 8)
        self.black_stroke_max = cfg.get("black_stroke_max", 95)
        self.black_sat_max = cfg.get("black_sat_max", 190)
        self.min_black_symbol_area = cfg.get("min_black_symbol_area", 80)
        self.min_classify_face_clarity = cfg.get("min_classify_face_clarity", 25)
        self.min_fake_endpoint_count = cfg.get("min_fake_endpoint_count", 2)
        self.max_fake_branch_count = cfg.get("max_fake_branch_count", 12)
        self.min_quad_area = cfg.get("min_quad_area", 450.0)
        self.min_quad_edge = cfg.get("min_quad_edge", 8.0)
        self.max_quad_aspect = cfg.get("max_quad_aspect", 6.0)
        self.min_warped_symbol_area = cfg.get(
            "min_warped_symbol_area", self.min_symbol_area)
        self.face_distance_abs_tol = cfg.get("face_distance_abs_tol", 0.08)
        self.face_distance_frac_tol = cfg.get("face_distance_frac_tol", 0.08)

        # --- localisation / association ---
        self.assoc_dist     = cfg.get("assoc_dist", 0.30)  # m, same-box matching

    # ════════════════════════════════════════════════════════════════════════
    # 1. DETECT BOXES  (red/blue colour candidates + depth validation)
    # ════════════════════════════════════════════════════════════════════════
    def _valid_depth_mask(self, depth, depth_scale=0.001):
        depth_m = depth.astype(np.float32) * depth_scale
        return ((depth > 0) &
                (depth_m > self.min_depth) &
                (depth_m < self.max_depth)).astype(np.uint8) * 255

    @staticmethod
    def _clean_binary(mask, open_size=5, close_size=7):
        if open_size > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (open_size, open_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if close_size > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (close_size, close_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _color_masks(self, rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        blue = cv2.inRange(hsv, np.array([90, 50, 20], np.uint8),
                           np.array([135, 255, 255], np.uint8))
        red_lo = cv2.inRange(hsv, np.array([0, 130, 30], np.uint8),
                             np.array([12, 255, 255], np.uint8))
        red_hi = cv2.inRange(hsv, np.array([168, 130, 30], np.uint8),
                             np.array([179, 255, 255], np.uint8))
        red = cv2.bitwise_or(red_lo, red_hi)

        blue = self._clean_binary(blue)
        red = self._clean_binary(red)
        color = cv2.bitwise_or(blue, red)
        return {"BLUE": blue, "RED": red, "ALL": color}

    def _refine_object_mask(self, depth_m, valid_depth, seed_mask, bbox):
        x, y, w, h = bbox
        H, W = depth_m.shape
        x0 = max(0, x - self.box_pad)
        y0 = max(0, y - self.box_pad)
        x1 = min(W, x + w + self.box_pad)
        y1 = min(H, y + h + self.box_pad)

        seed = (seed_mask > 0) & (valid_depth > 0)
        if int(seed.sum()) < self.min_seed_px:
            return None

        z = float(np.median(depth_m[seed]))
        tol = max(self.object_depth_tol, z * self.object_depth_frac)

        crop_depth = depth_m[y0:y1, x0:x1]
        crop_valid = valid_depth[y0:y1, x0:x1] > 0
        crop_seed = seed_mask[y0:y1, x0:x1] > 0
        near = (np.abs(crop_depth - z) <= tol) & crop_valid

        near_u8 = near.astype(np.uint8) * 255
        near_u8 = self._clean_binary(near_u8, open_size=3, close_size=9)

        # Keep only same-depth components that touch the red/blue seed. This
        # pulls in white marks on the box while rejecting unrelated depth pixels.
        keep = np.zeros_like(near_u8)
        contours, _ = cv2.findContours(near_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        seed_touch = cv2.dilate(crop_seed.astype(np.uint8) * 255,
                                cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, (9, 9)))
        for cnt in contours:
            component = np.zeros_like(near_u8)
            cv2.drawContours(component, [cnt], -1, 255, cv2.FILLED)
            if np.any((component > 0) & (seed_touch > 0)):
                cv2.drawContours(keep, [cnt], -1, 255, cv2.FILLED)

        if cv2.countNonZero(keep) < self.min_seed_px:
            return None
        return (x0, y0, x1 - x0, y1 - y0), keep, z

    def detect_boxes(self, rgb, depth, depth_scale=0.001):
        """Find red/blue boxes, then refine each candidate by valid depth.

        Returns a list of box dicts:
            {bbox:(x,y,w,h), rgb_crop, depth_crop, mask, color}
        `mask` is the depth-refined object mask cropped to the bbox.
        """
        depth_m = depth.astype(np.float32) * depth_scale
        valid_depth = self._valid_depth_mask(depth, depth_scale)
        masks = self._color_masks(rgb)
        boxes = []

        for color_name in ("BLUE", "RED"):
            contours, _ = cv2.findContours(masks[color_name], cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                color_area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                extent = color_area / max(1, w * h)
                if (color_area < self.color_min_area or
                        extent < self.min_color_extent):
                    continue

                seed_mask = np.zeros(depth.shape[:2], np.uint8)
                cv2.drawContours(seed_mask, [cnt], -1, 255, cv2.FILLED)
                refined = self._refine_object_mask(
                    depth_m, valid_depth, seed_mask, (x, y, w, h))
                if refined is None:
                    continue

                (x, y, w, h), mask_crop, object_depth = refined
                if cv2.countNonZero(mask_crop) < self.min_object_px:
                    continue

                boxes.append({
                    "bbox": (x, y, w, h),
                    "rgb_crop": rgb[y:y + h, x:x + w].copy(),
                    "depth_crop": depth[y:y + h, x:x + w].copy(),
                    "mask": mask_crop.copy(),
                    "color": color_name,
                    "object_depth_m": object_depth,
                })

        boxes.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        return boxes

    # ════════════════════════════════════════════════════════════════════════
    # 2. FLATTEN FACES  (split into faces, rectify each to head-on view)
    # ════════════════════════════════════════════════════════════════════════
    def _depth_to_points(self, depth_crop, mask, intr, bbox, depth_scale):
        """Crop depth -> Nx3 cloud (m) + the (u,v) pixel of each valid point.

        Only pixels inside the box `mask` are used, so background that happens to
        sit inside the bounding box is ignored.
        """
        x0, y0, _, _ = bbox
        ys, xs = np.nonzero((depth_crop > 0) & (mask > 0))
        z = depth_crop[ys, xs].astype(np.float32) * depth_scale
        u = xs + x0
        v = ys + y0
        X = (u - intr["ppx"]) * z / intr["fx"]
        Y = (v - intr["ppy"]) * z / intr["fy"]
        pts = np.stack([X, Y, z], axis=1)
        uv = np.stack([xs, ys], axis=1)        # local crop coords
        return pts, uv

    def _split_faces(self, pts, uv):
        """Greedy plane clustering: peel off the largest plane repeatedly.

        Returns a list of (face_pts, face_uv, normal, centroid), one per face.
        """
        faces = []
        remaining = np.ones(len(pts), bool)
        for _ in range(3):                      # a box shows at most 3 faces
            idx = np.nonzero(remaining)[0]
            if len(idx) < self.min_face_px:
                break
            P = pts[idx]
            centroid = P.mean(0)
            # PCA: smallest-eigenvector = plane normal
            _, _, vt = np.linalg.svd(P - centroid, full_matrices=False)
            normal = vt[2]
            dist = np.abs((P - centroid) @ normal)
            on_plane = dist < self.plane_tol
            if on_plane.sum() < self.min_face_px:
                break
            sel = idx[on_plane]
            faces.append((pts[sel], uv[sel], normal, pts[sel].mean(0)))
            remaining[sel] = False
        return faces

    @staticmethod
    def _face_rotation(normal):
        """Rotation matrix that turns the face normal to point at the camera (+Z)."""
        n = normal / (np.linalg.norm(normal) + 1e-9)
        if n[2] < 0:                            # face the camera
            n = -n
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(n, z)
        s = np.linalg.norm(axis)
        if s < 1e-6:
            return np.eye(3)
        axis /= s
        angle = np.arccos(np.clip(n @ z, -1, 1))
        R, _ = cv2.Rodrigues(axis * angle)
        return R

    def _build_warp_maps(self, P, uv, R, centroid):
        """Rotate cloud flat, scatter (X',Y') onto a grid -> map_x/map_y."""
        Pr = (P - centroid) @ R.T               # rotate; face now ~ XY plane
        xy = Pr[:, :2]
        lo, hi = xy.min(0), xy.max(0)
        span = np.maximum(hi - lo, 1e-6)
        S = self.out_size
        gx = np.clip(((xy[:, 0] - lo[0]) / span[0] * (S - 1)).astype(int), 0, S - 1)
        gy = np.clip(((xy[:, 1] - lo[1]) / span[1] * (S - 1)).astype(int), 0, S - 1)

        map_x = np.zeros((S, S), np.float32)
        map_y = np.zeros((S, S), np.float32)
        acc   = np.zeros((S, S), np.float32)
        # scatter-accumulate source pixel coords into each output bin
        np.add.at(map_x, (gy, gx), uv[:, 0])
        np.add.at(map_y, (gy, gx), uv[:, 1])
        np.add.at(acc,   (gy, gx), 1.0)
        filled = acc > 0
        map_x[filled] /= acc[filled]
        map_y[filled] /= acc[filled]

        # fill holes by dilating the filled values
        hole = (~filled).astype(np.uint8)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for _ in range(3):
            if hole.sum() == 0:
                break
            dx = cv2.dilate(map_x, kern)
            dy = cv2.dilate(map_y, kern)
            take = (hole == 1) & (cv2.dilate(filled.astype(np.uint8), kern) == 1)
            map_x[take] = dx[take]
            map_y[take] = dy[take]
            filled = filled | take
            hole = (~filled).astype(np.uint8)
        return map_x, map_y

    @staticmethod
    def clarity_score(image):
        """Sharpness score: higher means clearer."""
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _order_quad(pts):
        pts = np.asarray(pts, dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        return np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ], dtype=np.float32)

    def _symbol_mask(self, bgr, object_mask=None):
        """Mask white/black markings against a red/blue box face."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        color_mask = self._color_masks(bgr)["ALL"] > 0
        if object_mask is None:
            valid = np.ones(gray.shape, bool)
        else:
            valid = object_mask > 0

        greenish = (hue >= 35) & (hue <= 90) & (sat > 45) & (val > 45)
        saturated_bg = (sat > 150) & (val > 55) & (val < 170)
        neutral = valid & (~color_mask) & (~greenish) & (~saturated_bg)

        local = cv2.GaussianBlur(gray, (0, 0), 5)
        high_contrast = cv2.absdiff(gray, local) > 18
        white = ((val > 135) | ((gray > 105) & high_contrast)) & (sat < 175)
        black = ((val < 75) | ((gray < 105) & high_contrast)) & (sat < 150)
        mask = ((white | black) & neutral).astype(np.uint8) * 255
        mask = self._clean_binary(mask, open_size=3, close_size=7)

        bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge)

        cleaned = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        H, W = mask.shape
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_symbol_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            m = min(self.symbol_border_margin, max(4, min(H, W) // 20))
            touches_border = (
                x <= m or y <= m or x + w >= W - m or y + h >= H - m
            )
            if touches_border:
                if object_mask is not None:
                    continue
                if cv2.contourArea(cnt) < 500 or min(w, h) < 12:
                    continue
            cv2.drawContours(cleaned, [cnt], -1, 255, cv2.FILLED)
        return cleaned

    def _black_symbol_mask(self, bgr, symbol_hint=None):
        """Return only the dark KFS stroke, excluding white outline/background."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        if symbol_hint is None:
            hint = self._symbol_mask(bgr)
        else:
            hint = (symbol_hint > 0).astype(np.uint8) * 255

        if cv2.countNonZero(hint) == 0:
            return hint

        hint_bool = hint > 0
        hint_vals = gray[hint_bool]
        adaptive_cut = float(np.percentile(hint_vals, 35)) + 12.0
        dark_cut = int(max(45, min(self.black_stroke_max, adaptive_cut)))

        # The broad symbol hint finds the mark; these tests remove the white
        # outline and saturated face/background pixels from that region.
        greenish = (hue >= 35) & (hue <= 90) & (sat > 45) & (val > 45)
        neutral_dark = (sat < self.black_sat_max) | (val < 70)
        dark = ((gray <= dark_cut) | (val <= dark_cut))
        mask = (hint_bool & dark & neutral_dark & (~greenish)).astype(np.uint8) * 255

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        cleaned = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= 8:
                cv2.drawContours(cleaned, [cnt], -1, 255, cv2.FILLED)
        return cleaned

    def _symbol_groups(self, symbol_mask):
        """Group fragmented strokes into one mask per visible marking."""
        if cv2.countNonZero(symbol_mask) < self.min_symbol_area:
            return []

        k = max(3, int(self.symbol_group_close) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        grouped = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, kernel)
        grouped = cv2.dilate(
            grouped,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (grouped > 0).astype(np.uint8), 8)
        groups = []
        for label in range(1, n):
            x, y, w, h, _ = stats[label]
            component = labels == label
            mask = np.zeros_like(symbol_mask)
            mask[component & (symbol_mask > 0)] = 255
            area = int(cv2.countNonZero(mask))
            if area < self.min_symbol_group_area:
                continue
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            groups.append({
                "mask": mask,
                "bbox": (x0, y0, x1 - x0, y1 - y0),
                "center": (float(xs.mean()), float(ys.mean())),
                "area": float(area),
            })

        groups.sort(key=lambda g: (g["bbox"][1], g["bbox"][0]))
        return groups

    def _same_color_face_mask(self, crop, object_mask, color_name):
        masks = self._color_masks(crop)
        color = masks.get(color_name, masks["ALL"])
        color = self._clean_binary(color, open_size=3, close_size=15)
        color = cv2.dilate(
            color,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1)
        if object_mask is not None:
            obj = cv2.dilate(
                object_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1)
            color = cv2.bitwise_and(color, obj)
        return color

    @staticmethod
    def _mask_center(symbol_mask):
        ys, xs = np.nonzero(symbol_mask > 0)
        if len(xs) == 0:
            return None, 0
        return (float(xs.mean()), float(ys.mean())), int(len(xs))

    @staticmethod
    def _merge_positions(values, tol=8):
        if not values:
            return []
        vals = sorted(float(v) for v in values)
        groups = [[vals[0]]]
        for v in vals[1:]:
            if abs(v - groups[-1][-1]) <= tol:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [float(np.mean(g)) for g in groups]

    def _vertical_face_boundaries(self, edges, color_mask, groups):
        ys, xs = np.nonzero(color_mask > 0)
        if len(xs) == 0:
            return []

        verticals = [float(xs.min()), float(xs.max())]
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=18,
            minLineLength=max(12, min(edges.shape[:2]) // 4),
            maxLineGap=10)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0, :]:
                dx, dy = float(x2 - x1), float(y2 - y1)
                length = np.hypot(dx, dy)
                if length < 12:
                    continue
                if abs(dx) < abs(dy) * 0.65:
                    verticals.append((float(x1) + float(x2)) / 2.0)

        by_x = sorted(groups, key=lambda g: g["bbox"][0])
        for a, b in zip(by_x, by_x[1:]):
            ax, _, aw, _ = a["bbox"]
            bx, _, _, _ = b["bbox"]
            gap = bx - (ax + aw)
            if gap > max(12, self.symbol_group_close):
                verticals.append(ax + aw + gap / 2.0)
        return self._merge_positions(verticals)

    @staticmethod
    def _quad_bbox(quad):
        q = np.asarray(quad, dtype=np.float32)
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    @staticmethod
    def _bbox_iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        return inter / max(1e-6, area_a + area_b - inter)

    def _fallback_symbol_quad(self, group, color_mask):
        H, W = color_mask.shape
        x, y, w, h = group["bbox"]
        pad = int(max(w, h) * self.face_pad_frac)

        ys, xs = np.nonzero(color_mask > 0)
        if len(xs) > 0:
            bx0, bx1 = int(xs.min()), int(xs.max())
            by0, by1 = int(ys.min()), int(ys.max())
        else:
            bx0, by0, bx1, by1 = 0, 0, W - 1, H - 1

        x0 = max(bx0, x - pad, 0)
        y0 = max(by0, y - pad, 0)
        x1 = min(bx1, x + w + pad, W - 1)
        y1 = min(by1, y + h + pad, H - 1)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        dtype=np.float32)

    def _quad_from_color_region(self, color_mask, group, boundaries):
        cx, _ = group["center"]
        H, W = color_mask.shape
        lefts = [b for b in boundaries if b < cx - 2]
        rights = [b for b in boundaries if b > cx + 2]
        if lefts and rights:
            x0 = max(0, int(round(max(lefts))) - 2)
            x1 = min(W - 1, int(round(min(rights))) + 2)
        else:
            return self._fallback_symbol_quad(group, color_mask), "symbol_rect_fallback"

        if x1 - x0 < max(16, group["bbox"][2]):
            return self._fallback_symbol_quad(group, color_mask), "symbol_rect_fallback"

        band = np.zeros_like(color_mask)
        band[:, x0:x1 + 1] = color_mask[:, x0:x1 + 1]
        band = cv2.morphologyEx(
            band, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

        group_touch = cv2.dilate(
            group["mask"],
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_face_px:
                continue
            component = np.zeros_like(color_mask)
            cv2.drawContours(component, [cnt], -1, 255, cv2.FILLED)
            overlap = cv2.countNonZero(cv2.bitwise_and(component, group_touch))
            if overlap > best_score:
                best = cnt
                best_score = overlap

        if best is None:
            return self._fallback_symbol_quad(group, color_mask), "symbol_rect_fallback"

        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.035 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx[:, 0, :].astype(np.float32)
            return self._order_quad(quad), "color_face_contour"

        hull = cv2.convexHull(best)
        if len(hull) >= 4:
            quad = hull[:, 0, :].astype(np.float32)
            return self._order_quad(quad), "color_face_hull"

        rect = cv2.minAreaRect(best)
        quad = cv2.boxPoints(rect).astype(np.float32)
        return self._order_quad(quad), "color_face_rect"

    def _face_edges(self, crop, object_mask, symbol_mask=None):
        if object_mask is None:
            object_mask = np.ones(crop.shape[:2], np.uint8) * 255
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=object_mask)
        masked = cv2.GaussianBlur(masked, (3, 3), 0)
        edges = cv2.Canny(masked, 45, 135)
        mask_edges = cv2.Canny(object_mask, 40, 120)
        edges = cv2.bitwise_or(edges, mask_edges)
        edges = cv2.bitwise_and(edges, edges, mask=object_mask)
        if symbol_mask is not None:
            symbol_zone = cv2.dilate(
                symbol_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
            edges[symbol_zone > 0] = 0
        return edges

    def _quad_debug_image(self, crop, faces, symbol_mask, edges, groups):
        img = crop.copy()
        if edges is not None:
            img[edges > 0] = (0, 255, 0)
        if symbol_mask is not None:
            img[symbol_mask > 0] = (255, 255, 255)
        palette = [(0, 255, 255), (255, 0, 255), (0, 180, 255),
                   (255, 255, 0)]
        for idx, group in enumerate(groups):
            color = palette[idx % len(palette)]
            x, y, w, h = group["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        for idx, face in enumerate(faces):
            color = palette[idx % len(palette)]
            q = np.round(np.asarray(face["quad"], dtype=np.float32)).astype(int)
            cv2.polylines(img, [q], True, color, 2)
            cx = int(np.mean(q[:, 0]))
            cy = int(np.mean(q[:, 1]))
            cv2.putText(img, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
        return img

    def _symbol_center(self, symbol_mask):
        center, area = self._mask_center(symbol_mask)
        return center, float(area)

    def _warp_quad(self, crop, quad, flags=cv2.INTER_LINEAR, border_value=0):
        S = self.out_size
        dst = np.array([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(self._order_quad(quad), dst)
        return cv2.warpPerspective(
            crop, M, (S, S), flags=flags, borderValue=border_value)

    def _quad_quality(self, quad, crop_shape):
        q = self._order_quad(quad)
        H, W = crop_shape[:2]
        edges = [
            float(np.linalg.norm(q[(i + 1) % 4] - q[i]))
            for i in range(4)
        ]
        min_edge = min(edges) if edges else 0.0
        width = (edges[0] + edges[2]) / 2.0
        height = (edges[1] + edges[3]) / 2.0
        aspect = max(width, height) / max(1e-6, min(width, height))
        area = float(abs(cv2.contourArea(q)))
        unique = len(np.unique(np.round(q, 1), axis=0))
        in_bounds = bool(
            np.all(q[:, 0] >= -1) and np.all(q[:, 0] <= W) and
            np.all(q[:, 1] >= -1) and np.all(q[:, 1] <= H))

        reason = "ok"
        if unique < 4:
            reason = "duplicate_quad_points"
        elif area < self.min_quad_area:
            reason = "quad_area_too_small"
        elif min_edge < self.min_quad_edge:
            reason = "quad_edge_too_short"
        elif aspect > self.max_quad_aspect:
            reason = "quad_aspect_too_extreme"
        elif not in_bounds:
            reason = "quad_out_of_crop_bounds"

        return {
            "valid": reason == "ok",
            "reason": reason,
            "area": area,
            "min_edge": float(min_edge),
            "aspect": float(aspect),
            "in_bounds": in_bounds,
        }

    @staticmethod
    def _face_score(face):
        quality = face.get("quad_quality", {})
        return (
            float(face.get("warped_symbol_area", 0)),
            float(quality.get("area", 0.0)),
            -float(quality.get("aspect", 999.0)),
        )

    def _selected_face_debug_image(self, crop, face):
        img = crop.copy()
        q = np.round(np.asarray(face["quad"], dtype=np.float32)).astype(int)
        cv2.polylines(img, [q], True, (0, 255, 255), 2)
        cx = int(np.mean(q[:, 0]))
        cy = int(np.mean(q[:, 1]))
        cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
        return img

    def flatten_faces(self, box, intr, depth_scale=0.001):
        """Rectify the single most visible marked face with a 2D warp.

        Returns a list of face dicts:
            either [] or [{image, quad, method, clarity_score, symbol_area}]
        """
        crop = box["rgb_crop"]
        object_mask = box["mask"]
        clarity = self.clarity_score(crop)
        if clarity < self.min_box_clarity:
            return []

        symbol_mask = self._symbol_mask(crop, object_mask)
        groups = self._symbol_groups(symbol_mask)
        if not groups:
            return []

        color_mask = self._same_color_face_mask(
            crop, object_mask, box.get("color", "ALL"))
        if cv2.countNonZero(color_mask) < self.min_face_px:
            return []

        edges = self._face_edges(crop, color_mask, symbol_mask)
        boundaries = self._vertical_face_boundaries(edges, color_mask, groups)
        candidates = []
        used = []
        for group in groups:
            quad, method = self._quad_from_color_region(
                color_mask, group, boundaries)
            quad_quality = self._quad_quality(quad, crop.shape)
            if not quad_quality["valid"]:
                continue
            bbox = self._quad_bbox(quad)
            if any(self._bbox_iou(bbox, prev) > 0.82 for prev in used):
                continue
            image = self._warp_quad(crop, quad)
            if image is None or image.size == 0:
                continue
            warped_symbol = self._warp_quad(
                group["mask"], quad, flags=cv2.INTER_NEAREST, border_value=0)
            warped_symbol = (warped_symbol > 0).astype(np.uint8) * 255
            warped_symbol_area = int(cv2.countNonZero(warped_symbol))
            if warped_symbol_area < self.min_warped_symbol_area:
                continue
            used.append(bbox)
            candidates.append({
                "image": image,
                "quad": [[float(x), float(y)] for x, y in quad],
                "method": method,
                "clarity_score": clarity,
                "symbol_area": float(group["area"]),
                "warped_symbol_area": int(warped_symbol_area),
                "quad_quality": quad_quality,
                "symbol_mask": warped_symbol,
                "crop_symbol_mask": symbol_mask,
                "crop_symbol_group": group["mask"],
                "edges": edges,
            })

        if not candidates:
            return []

        best = max(candidates, key=self._face_score)
        best["selection_score"] = [float(v) for v in self._face_score(best)]
        best["selected_face_debug"] = self._selected_face_debug_image(crop, best)
        return [best]

    # ════════════════════════════════════════════════════════════════════════
    # 3. CLASSIFY FACE  (2D corner/anchor logic ported from the prototypes)
    # ════════════════════════════════════════════════════════════════════════
    def _to_binary(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, self.stroke_thresh, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (self.close_kernel, self.close_kernel))
        closed = cv2.morphologyEx(cv2.bitwise_not(b), cv2.MORPH_CLOSE, k)
        return cv2.bitwise_not(closed)

    def _find_corners(self, binary):
        contours, _ = cv2.findContours(cv2.bitwise_not(binary),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.kfs_min_area:
                continue
            approx = cv2.approxPolyDP(cnt, self.epsilon_frac * cv2.arcLength(cnt, True), True)
            verts = approx[:, 0, :]
            if len(verts) < 3:
                continue
            for i in range(len(verts)):
                p0 = verts[i].astype(float)
                v1 = verts[i - 1].astype(float) - p0
                v2 = verts[(i + 1) % len(verts)].astype(float) - p0
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1 or n2 < 1:
                    continue
                ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
                if ang < self.angle_thresh:
                    pts.append((int(p0[0]), int(p0[1])))
        return pts

    def _check_corner(self, binary, cx, cy):
        h, w = binary.shape
        ang = np.linspace(0, 2 * np.pi, self.n_samples, endpoint=False)
        xs = (cx + self.circle_radius * np.cos(ang)).astype(int)
        ys = (cy + self.circle_radius * np.sin(ang)).astype(int)
        vals = [binary[y, x] if 0 <= x < w and 0 <= y < h else 255
                for x, y in zip(xs, ys)]
        white_pct = sum(v > 127 for v in vals) / self.n_samples

        dark = [i for i, v in enumerate(vals) if v <= 127]
        if not dark:
            return white_pct, False
        segs, cur = [], [dark[0]]
        for i in range(1, len(dark)):
            if dark[i] == dark[i - 1] + 1:
                cur.append(dark[i])
            else:
                segs.append(cur)
                cur = [dark[i]]
        segs.append(cur)
        if len(segs) > 1 and segs[0][0] == 0 and segs[-1][-1] == self.n_samples - 1:
            segs[0] = segs.pop() + segs[0]

        if len(segs) == 1:
            return white_pct, len(segs[0]) * 360 / self.n_samples <= self.angle_thresh + 15
        means = [np.arctan2(sum(np.sin(ang[i]) for i in s),
                            sum(np.cos(ang[i]) for i in s)) for s in segs]
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                d = np.degrees(abs(means[i] - means[j]))
                d = 360 - d if d > 180 else d
                if d <= self.angle_thresh + 15:
                    return white_pct, True
        return white_pct, False

    def _remove_duplicates(self, pts):
        kept = []
        for px, py in pts:
            if all(abs(px - kx) > self.dedup_dist or abs(py - ky) > self.dedup_dist
                   for kx, ky in kept):
                kept.append((px, py))
        return kept

    @staticmethod
    def _skeletonize(mask):
        img = (mask > 0).astype(np.uint8) * 255
        skel = np.zeros_like(img)
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(img) > 0:
            eroded = cv2.erode(img, elem)
            opened = cv2.dilate(eroded, elem)
            temp = cv2.subtract(img, opened)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded
        return skel

    def _endpoint_angle(self, symbol_mask, endpoint):
        contours, _ = cv2.findContours(symbol_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        ep = np.array(endpoint, dtype=np.float32)
        best = None
        best_dist = None
        for cnt in contours:
            pts = cnt[:, 0, :].astype(np.float32)
            dists = np.linalg.norm(pts - ep, axis=1)
            idx = int(np.argmin(dists))
            dist = float(dists[idx])
            if best_dist is None or dist < best_dist:
                best = (pts, idx)
                best_dist = dist
        if best is None:
            return None
        pts, idx = best
        if len(pts) < 8:
            return None
        step = min(max(4, self.endpoint_tip_radius // 2), len(pts) // 4)
        p0 = pts[idx]
        p1 = pts[(idx - step) % len(pts)]
        p2 = pts[(idx + step) % len(pts)]
        v1, v2 = p1 - p0, p2 - p0
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            return None
        return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2),
                                                  -1, 1))))

    @staticmethod
    def _cluster_points(points, radius):
        if not points:
            return []
        remaining = [np.array(p, dtype=np.float32) for p in points]
        clusters = []
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            changed = True
            while changed:
                changed = False
                keep = []
                center = np.mean(cluster, axis=0)
                for pt in remaining:
                    if np.linalg.norm(pt - center) <= radius:
                        cluster.append(pt)
                        changed = True
                    else:
                        keep.append(pt)
                remaining = keep
            center = np.mean(cluster, axis=0)
            clusters.append((int(round(center[0])), int(round(center[1]))))
        return clusters

    @staticmethod
    def _nearest_contour_point(symbol_mask, point):
        contours, _ = cv2.findContours(symbol_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            return point
        target = np.array(point, dtype=np.float32)
        best_pt = None
        best_dist = None
        for cnt in contours:
            pts = cnt[:, 0, :].astype(np.float32)
            dists = np.linalg.norm(pts - target, axis=1)
            idx = int(np.argmin(dists))
            dist = float(dists[idx])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_pt = pts[idx]
        if best_pt is None:
            return point
        return (int(round(best_pt[0])), int(round(best_pt[1])))

    @staticmethod
    def _point_segment_distance(point, a, b):
        p = np.array(point, dtype=np.float32)
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-6:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _near_symbol_hull(self, symbol_mask, point, max_dist):
        contours, _ = cv2.findContours(symbol_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            return False
        best = None
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_symbol_area:
                continue
            hull = cv2.convexHull(cnt)[:, 0, :]
            if len(hull) < 2:
                continue
            for i in range(len(hull)):
                dist = self._point_segment_distance(
                    point, hull[i], hull[(i + 1) % len(hull)])
                if best is None or dist < best:
                    best = dist
        return best is not None and best <= max_dist

    def _endpoint_candidates(self, symbol_mask, skeleton):
        skel_bool = skeleton > 0
        kernel = np.ones((3, 3), np.uint8)
        neighbor_count = cv2.filter2D(skel_bool.astype(np.uint8), -1, kernel)
        neighbor_count = neighbor_count - skel_bool.astype(np.uint8)

        branch_mask = skel_bool & (neighbor_count >= 5)
        branch_u8 = branch_mask.astype(np.uint8) * 255
        ignored_intersections = []
        if cv2.countNonZero(branch_u8) > 0:
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(
                branch_u8, 8)
            for label in range(1, n):
                if stats[label, cv2.CC_STAT_AREA] < 2:
                    continue
                cx, cy = centroids[label]
                ignored_intersections.append((int(round(cx)), int(round(cy))))
            r = self.endpoint_branch_ignore_radius
            branch_zone = cv2.dilate(
                branch_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * r + 1, 2 * r + 1)))
        else:
            branch_zone = np.zeros_like(skeleton)

        endpoint_mask = skel_bool & (neighbor_count == 1) & (branch_zone == 0)
        ey, ex = np.nonzero(endpoint_mask)
        medial_points = list(zip(ex.tolist(), ey.tolist()))
        merged = self._cluster_points(medial_points, self.endpoint_merge_radius)
        contour_points = [self._nearest_contour_point(symbol_mask, pt)
                          for pt in merged]
        endpoints = self._cluster_points(contour_points,
                                         self.endpoint_merge_radius)
        endpoints = [
            pt for pt in endpoints
            if self._near_symbol_hull(
                symbol_mask, pt, max(8, int(self.endpoint_tip_radius * 1.4)))
        ]
        ignored_intersections = self._cluster_points(
            ignored_intersections, self.endpoint_merge_radius)
        return endpoints, ignored_intersections

    def _endpoint_debug_image(self, face_img, symbol_mask, skeleton,
                              endpoints, ignored_intersections, sharp_points,
                              round_points):
        dbg = face_img.copy()
        dbg[skeleton > 0] = (0, 255, 255)
        for x, y in ignored_intersections:
            cv2.circle(dbg, (int(x), int(y)), 6, (0, 0, 255), 2)
            cv2.line(dbg, (int(x) - 5, int(y) - 5),
                     (int(x) + 5, int(y) + 5), (0, 0, 255), 1)
            cv2.line(dbg, (int(x) - 5, int(y) + 5),
                     (int(x) + 5, int(y) - 5), (0, 0, 255), 1)
        for x, y in endpoints:
            cv2.circle(dbg, (int(x), int(y)), 4, (0, 255, 0), -1)
        for x, y in round_points:
            cv2.circle(dbg, (int(x), int(y)), 7, (255, 0, 0), 2)
        for x, y in sharp_points:
            cv2.circle(dbg, (int(x), int(y)), 7, (0, 255, 255), 2)
        return dbg

    def classify_face(self, face_img, symbol_hint=None):
        """Classify one flattened face by sharpness of true symbol endpoints.

        label in {REAL, FAKE, NONE}. info has valid_count, n_corners.
        """
        empty_info = {
            "valid_count": 0,
            "n_corners": 0,
            "sharp_end_count": 0,
            "round_end_count": 0,
            "branch_count": 0,
            "endpoint_count": 0,
            "endpoint_candidates": [],
            "ignored_intersections": [],
            "face_clarity": 0.0,
            "black_symbol_area": 0,
            "confidence": 0.0,
            "quality_reason": "empty_face",
            "symbol_source": "none",
            "hint_fallback_reason": None,
        }
        if face_img is None or face_img.size == 0:
            return "NONE", empty_info

        face_clarity = self.clarity_score(face_img)
        hint_mask = None
        fallback_reason = None
        if symbol_hint is not None:
            hint_mask = (symbol_hint > 0).astype(np.uint8) * 255

        if hint_mask is not None and cv2.countNonZero(hint_mask) >= self.min_symbol_area:
            symbol_mask = hint_mask
            symbol_source = "warped_hint"
        else:
            symbol_mask = self._symbol_mask(face_img)
            symbol_source = "auto" if hint_mask is None else "auto_fallback"
            if hint_mask is not None:
                fallback_reason = "hint_symbol_too_small_auto_fallback"

        empty_info["face_clarity"] = float(face_clarity)
        empty_info["symbol_mask"] = symbol_mask
        empty_info["hint_symbol_mask"] = hint_mask
        empty_info["symbol_source"] = symbol_source
        empty_info["hint_fallback_reason"] = fallback_reason

        if cv2.countNonZero(symbol_mask) < self.min_symbol_area:
            empty_info["quality_reason"] = "symbol_mask_too_small"
            return "NONE", empty_info

        black_symbol_mask = self._black_symbol_mask(face_img, symbol_mask)
        black_area = int(cv2.countNonZero(black_symbol_mask))
        if (black_area < self.min_black_symbol_area and
                symbol_source == "warped_hint"):
            auto_symbol_mask = self._symbol_mask(face_img)
            auto_black_mask = self._black_symbol_mask(face_img, auto_symbol_mask)
            auto_black_area = int(cv2.countNonZero(auto_black_mask))
            if (cv2.countNonZero(auto_symbol_mask) >= self.min_symbol_area and
                    auto_black_area >= self.min_black_symbol_area):
                symbol_mask = auto_symbol_mask
                black_symbol_mask = auto_black_mask
                black_area = auto_black_area
                symbol_source = "auto_fallback"
                fallback_reason = "hint_black_too_small_auto_fallback"

        empty_info["black_symbol_area"] = black_area
        empty_info["black_symbol_mask"] = black_symbol_mask
        empty_info["symbol_mask"] = symbol_mask
        empty_info["symbol_source"] = symbol_source
        empty_info["hint_fallback_reason"] = fallback_reason
        if black_area < self.min_black_symbol_area:
            empty_info["quality_reason"] = (
                "hint_black_too_small_auto_fallback_failed"
                if fallback_reason else "black_symbol_too_small")
            empty_info["endpoints_image"] = face_img.copy()
            return "NONE", empty_info

        skeleton = self._skeletonize(black_symbol_mask)
        endpoints, ignored_intersections = self._endpoint_candidates(
            black_symbol_mask, skeleton)

        sharp_points = []
        round_points = []
        for pt in endpoints:
            angle = self._endpoint_angle(black_symbol_mask, pt)
            if angle is not None and angle <= self.sharp_end_angle_thresh:
                sharp_points.append(pt)
            else:
                round_points.append(pt)

        quality_reason = "ok"
        confidence = 0.0
        if face_clarity < self.min_classify_face_clarity:
            label = "NONE"
            quality_reason = "face_too_blurry"
        elif not endpoints:
            label = "NONE"
            quality_reason = "no_endpoints"
        elif sharp_points:
            label = "REAL"
            confidence = min(1.0, 0.85 + 0.03 * len(sharp_points))
        elif len(ignored_intersections) > self.max_fake_branch_count:
            label = "NONE"
            quality_reason = "too_many_branches"
        elif len(round_points) >= self.min_fake_endpoint_count:
            label = "FAKE"
            confidence = 0.70
        else:
            label = "NONE"
            quality_reason = "too_few_round_endpoints"
        if quality_reason == "ok" and fallback_reason:
            quality_reason = fallback_reason

        endpoint_debug = self._endpoint_debug_image(
            face_img, black_symbol_mask, skeleton, endpoints,
            ignored_intersections, sharp_points, round_points)
        return label, {
            "valid_count": len(sharp_points),
            "n_corners": len(endpoints),
            "sharp_end_count": len(sharp_points),
            "round_end_count": len(round_points),
            "branch_count": len(ignored_intersections),
            "endpoint_count": len(endpoints),
            "endpoint_candidates": [[int(x), int(y)] for x, y in endpoints],
            "ignored_intersections": [
                [int(x), int(y)] for x, y in ignored_intersections
            ],
            "symbol_mask": symbol_mask,
            "hint_symbol_mask": hint_mask,
            "black_symbol_mask": black_symbol_mask,
            "skeleton": skeleton,
            "endpoints_image": endpoint_debug,
            "face_clarity": float(face_clarity),
            "black_symbol_area": int(black_area),
            "confidence": float(confidence),
            "quality_reason": quality_reason,
            "symbol_source": symbol_source,
            "hint_fallback_reason": fallback_reason,
        }

    # ════════════════════════════════════════════════════════════════════════
    # 4. FIND LOCATION  (camera-relative XYZ, meters)
    # ════════════════════════════════════════════════════════════════════════
    def _deproject_pixel(self, u, v, z, intr):
        X = (u - intr["ppx"]) * z / intr["fx"]
        Y = (v - intr["ppy"]) * z / intr["fy"]
        return (float(X), float(Y), float(z))

    @staticmethod
    def _distance_xyz(point):
        if point is None:
            return None
        return float(np.linalg.norm(np.asarray(point, dtype=np.float32)))

    def _bbox_centroid_depth(self, box, depth_scale=0.001, radius=3):
        """Depth at bbox center; falls back to a small valid-depth window."""
        _, _, w, h = box["bbox"]
        depth_m = box["depth_crop"].astype(np.float32) * depth_scale
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        ix = int(round(cx))
        iy = int(round(cy))

        exact = float(depth_m[iy, ix])
        if self.min_depth < exact < self.max_depth:
            return exact, "center_pixel"

        y0, y1 = max(0, iy - radius), min(depth_m.shape[0], iy + radius + 1)
        x0, x1 = max(0, ix - radius), min(depth_m.shape[1], ix + radius + 1)
        window = depth_m[y0:y1, x0:x1]
        valid = window[(window > self.min_depth) & (window < self.max_depth)]
        if valid.size > 0:
            return float(np.median(valid)), f"{2 * radius + 1}x{2 * radius + 1}_window"

        object_depths = depth_m[
            (depth_m > self.min_depth) &
            (depth_m < self.max_depth) &
            (box["mask"] > 0)]
        if object_depths.size > 0:
            return float(np.median(object_depths)), "object_mask_median_fallback"
        return None, "no_valid_depth"

    def _local_depth_at(self, depth_m, u, v, valid_mask=None, radius=3):
        H, W = depth_m.shape[:2]
        ix = int(round(u))
        iy = int(round(v))
        if 0 <= ix < W and 0 <= iy < H:
            exact = float(depth_m[iy, ix])
            exact_valid = self.min_depth < exact < self.max_depth
            if valid_mask is not None:
                exact_valid = exact_valid and bool(valid_mask[iy, ix])
            if exact_valid:
                return exact, "center_pixel"

        x0, x1 = max(0, ix - radius), min(W, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(H, iy + radius + 1)
        if x0 < x1 and y0 < y1:
            window = depth_m[y0:y1, x0:x1]
            valid = (window > self.min_depth) & (window < self.max_depth)
            if valid_mask is not None:
                valid &= valid_mask[y0:y1, x0:x1]
            values = window[valid]
            if values.size > 0:
                return float(np.median(values)), (
                    f"{2 * radius + 1}x{2 * radius + 1}_window")
        return None, "no_valid_depth"

    def _quad_local_mask(self, shape, quad):
        mask = np.zeros(shape[:2], np.uint8)
        q = np.round(self._order_quad(quad)).astype(np.int32)
        cv2.fillConvexPoly(mask, q, 255)
        return mask

    def face_localization_info(self, box, face, intr, depth_scale=0.001,
                               loc_info=None):
        """Distance from crop center to the detected face center."""
        if loc_info is None:
            loc_info = self.localization_info(box, intr, depth_scale)

        x, y, _, _ = box["bbox"]
        quad = np.asarray(face["quad"], dtype=np.float32)
        face_local = quad.mean(axis=0)
        face_u = float(x + face_local[0])
        face_v = float(y + face_local[1])

        depth_m = box["depth_crop"].astype(np.float32) * depth_scale
        quad_mask = self._quad_local_mask(depth_m.shape, quad) > 0
        valid_face = quad_mask & (box["mask"] > 0)
        face_depth, face_depth_source = self._local_depth_at(
            depth_m, float(face_local[0]), float(face_local[1]), valid_face)
        if face_depth is None:
            values = depth_m[
                valid_face &
                (depth_m > self.min_depth) &
                (depth_m < self.max_depth)]
            if values.size > 0:
                face_depth = float(np.median(values))
                face_depth_source = "face_quad_median_fallback"

        face_xyz = None
        if face_depth is not None:
            face_xyz = self._deproject_pixel(face_u, face_v, face_depth, intr)

        crop_distance = loc_info["bbox_centroid_distance_m"]
        face_distance = self._distance_xyz(face_xyz)
        distance_delta = None
        if crop_distance is not None and face_distance is not None:
            distance_delta = abs(float(face_distance) - float(crop_distance))
        tolerance = max(
            self.face_distance_abs_tol,
            self.face_distance_frac_tol * float(crop_distance or 0.0))
        distance_ok = (
            distance_delta is not None and distance_delta <= tolerance)

        return {
            "crop_center_pixel": loc_info["bbox_centroid_pixel"],
            "crop_center_depth_m": loc_info["bbox_centroid_depth_m"],
            "crop_center_depth_source": loc_info["bbox_centroid_depth_source"],
            "crop_center_xyz_m": loc_info["bbox_centroid_xyz_m"],
            "crop_center_distance_m": crop_distance,
            "face_center_pixel": [float(face_u), float(face_v)],
            "face_center_depth_m": (
                float(face_depth) if face_depth is not None else None),
            "face_center_depth_source": face_depth_source,
            "face_center_xyz_m": (
                [float(v) for v in face_xyz] if face_xyz else None),
            "face_center_distance_m": face_distance,
            "distance_delta_m": (
                float(distance_delta) if distance_delta is not None else None),
            "distance_tolerance_m": float(tolerance),
            "distance_ok": bool(distance_ok),
        }

    def localization_info(self, box, intr, depth_scale=0.001):
        """Object-mask location plus bbox-centroid depth/distance diagnostics."""
        x, y, w, h = box["bbox"]
        sub = box["depth_crop"].astype(np.float32) * depth_scale
        valid_obj = ((sub > self.min_depth) &
                     (sub < self.max_depth) &
                     (box["mask"] > 0))
        ys, xs = np.nonzero(valid_obj)

        object_xyz = None
        object_pixel = None
        object_depth = None
        if len(xs) > 0:
            object_depth = float(np.median(sub[ys, xs]))
            object_u = float(np.mean(xs)) + x
            object_v = float(np.mean(ys)) + y
            object_pixel = (object_u, object_v)
            object_xyz = self._deproject_pixel(object_u, object_v,
                                               object_depth, intr)

        bbox_u = float(x + (w - 1) / 2.0)
        bbox_v = float(y + (h - 1) / 2.0)
        bbox_depth, bbox_depth_source = self._bbox_centroid_depth(
            box, depth_scale)
        bbox_xyz = None
        if bbox_depth is not None:
            bbox_xyz = self._deproject_pixel(bbox_u, bbox_v, bbox_depth, intr)

        return {
            "object_centroid_pixel": (
                [float(object_pixel[0]), float(object_pixel[1])]
                if object_pixel else None),
            "object_depth_m": object_depth,
            "object_xyz_m": (
                [float(v) for v in object_xyz] if object_xyz else None),
            "object_distance_m": self._distance_xyz(object_xyz),
            "bbox_centroid_pixel": [float(bbox_u), float(bbox_v)],
            "bbox_centroid_depth_m": (
                float(bbox_depth) if bbox_depth is not None else None),
            "bbox_centroid_depth_source": bbox_depth_source,
            "bbox_centroid_xyz_m": (
                [float(v) for v in bbox_xyz] if bbox_xyz else None),
            "bbox_centroid_distance_m": self._distance_xyz(bbox_xyz),
            "crop_center_pixel": [float(bbox_u), float(bbox_v)],
            "crop_center_depth_m": (
                float(bbox_depth) if bbox_depth is not None else None),
            "crop_center_depth_source": bbox_depth_source,
            "crop_center_xyz_m": (
                [float(v) for v in bbox_xyz] if bbox_xyz else None),
            "crop_center_distance_m": self._distance_xyz(bbox_xyz),
            "valid_object_depth_px": int(len(xs)),
        }

    def find_location(self, box, intr, depth_scale=0.001):
        """Deproject the box's depth centroid to camera-relative (X, Y, Z) m."""
        loc = self.localization_info(box, intr, depth_scale)["object_xyz_m"]
        return tuple(loc) if loc else None

    @staticmethod
    def _is_predictive_label(label):
        return label in ("REAL", "FAKE")

    def _face_result_record(self, face, label, info, face_loc, artifacts=None):
        record = {
            "label": label,
            "prediction_accepted": bool(
                self._is_predictive_label(label) and
                face_loc.get("distance_ok", False)),
            "valid_count": int(info["valid_count"]),
            "n_corners": int(info["n_corners"]),
            "sharp_end_count": int(info["sharp_end_count"]),
            "round_end_count": int(info["round_end_count"]),
            "branch_count": int(info["branch_count"]),
            "endpoint_count": int(info["endpoint_count"]),
            "endpoint_candidates": info.get("endpoint_candidates", []),
            "ignored_intersections": info.get("ignored_intersections", []),
            "face_clarity": float(info.get("face_clarity", 0.0)),
            "black_symbol_area": int(info.get("black_symbol_area", 0)),
            "confidence": float(info.get("confidence", 0.0)),
            "quality_reason": info.get("quality_reason", "unknown"),
            "symbol_source": info.get("symbol_source", "unknown"),
            "hint_fallback_reason": info.get("hint_fallback_reason"),
            "quad": face["quad"],
            "quad_quality": face.get("quad_quality", {}),
            "method": face["method"],
            "clarity_score": float(face["clarity_score"]),
            "symbol_area": float(face["symbol_area"]),
            "warped_symbol_area": int(face.get("warped_symbol_area", 0)),
            "localization": face_loc,
            "crop_center_xyz_m": face_loc.get("crop_center_xyz_m"),
            "crop_center_distance_m": face_loc.get("crop_center_distance_m"),
            "face_center_xyz_m": face_loc.get("face_center_xyz_m"),
            "face_center_distance_m": face_loc.get("face_center_distance_m"),
            "distance_delta_m": face_loc.get("distance_delta_m"),
            "distance_tolerance_m": face_loc.get("distance_tolerance_m"),
            "distance_ok": bool(face_loc.get("distance_ok", False)),
        }
        if artifacts:
            record.update(artifacts)
        return record

    # ════════════════════════════════════════════════════════════════════════
    # 5. SAVE  (JSON log / cache)
    # ════════════════════════════════════════════════════════════════════════
    def save(self, records, path):
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        return path

    # ── orchestration ────────────────────────────────────────────────────────
    def process_frame(self, rgb, depth, intr, depth_scale=0.001, frame_idx=0):
        """Run the full pipeline on one frame. Returns a list of box records."""
        frame_clarity = self.clarity_score(rgb)
        if frame_clarity < self.min_frame_clarity:
            return []

        records = []
        for box in self.detect_boxes(rgb, depth, depth_scale):
            loc_info = self.localization_info(box, intr, depth_scale)
            loc = loc_info["crop_center_xyz_m"]
            faces = self.flatten_faces(box, intr, depth_scale)
            if not faces:
                continue
            face_results = []
            for face in faces:
                label, info = self.classify_face(
                    face["image"], face.get("symbol_mask"))
                face_loc = self.face_localization_info(
                    box, face, intr, depth_scale, loc_info)
                result = self._face_result_record(face, label, info, face_loc)
                if result["prediction_accepted"]:
                    face_results.append({
                        "label": result["label"],
                        "confidence": result["confidence"],
                        "prediction_accepted": True,
                        "selected_face_score": face.get("selection_score", []),
                        "selected_symbol_area": face.get("warped_symbol_area", 0),
                        "selected_quad_area": (
                            face.get("quad_quality", {}).get("area")),
                    })
            if not face_results:
                continue
            records.append({
                "frame": frame_idx,
                "frame_clarity": frame_clarity,
                "bbox": list(box["bbox"]),
                "color": box.get("color"),
                "location_xyz_m": loc,
                "distance_m": loc_info["crop_center_distance_m"],
                "faces": face_results,
            })
        return records

    def _depth_visual(self, depth, mask=None, depth_scale=0.001):
        depth_m = depth.astype(np.float32) * depth_scale
        valid = (depth_m > self.min_depth) & (depth_m < self.max_depth)
        if mask is not None:
            valid &= mask > 0
        if np.any(valid):
            lo = float(np.percentile(depth_m[valid], 2))
            hi = float(np.percentile(depth_m[valid], 98))
            if hi <= lo:
                hi = lo + 1e-6
        else:
            lo, hi = self.min_depth, self.max_depth
        norm = np.clip((depth_m - lo) / (hi - lo), 0, 1)
        img = (norm * 255).astype(np.uint8)
        img[~valid] = 0
        return cv2.applyColorMap(img, cv2.COLORMAP_JET)

    @staticmethod
    def _rel_path(path, root):
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    def debug_process_frame(self, rgb, depth, intr, depth_scale=0.001,
                            frame_idx=0, timestamp_s=None,
                            requested_time_s=None, out_dir="debug",
                            initial_frame_idx=None, initial_timestamp_s=None,
                            initial_clarity=None, skipped_blurry_frames=0):
        """Process one frame and save visual artifacts for inspection."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_clarity = self.clarity_score(rgb)

        color_path = out_dir / "color.png"
        overlay_path = out_dir / "overlay.png"

        cv2.imwrite(str(color_path), rgb)

        overlay = rgb.copy()
        boxes_out = []
        accepted_count = 0
        for box_idx, box in enumerate(self.detect_boxes(rgb, depth, depth_scale)):
            loc_info = self.localization_info(box, intr, depth_scale)
            loc = loc_info["crop_center_xyz_m"]
            distance = loc_info["crop_center_distance_m"]
            faces = self.flatten_faces(box, intr, depth_scale)

            x, y, w, h = box["bbox"]
            crop_path = out_dir / f"box_{box_idx}_crop.png"
            cv2.imwrite(str(crop_path), box["rgb_crop"])

            artifacts = {
                "crop": self._rel_path(crop_path, out_dir),
            }
            selected_face = None
            selected_face_summary = None
            verdict = "NONE"
            reject_reason = None
            status = "rejected"

            if faces:
                face = faces[0]
                label, info = self.classify_face(
                    face["image"], face.get("symbol_mask"))
                face_loc = self.face_localization_info(
                    box, face, intr, depth_scale, loc_info)

                selected_face_path = out_dir / f"box_{box_idx}_selected_face.png"
                flatten_path = out_dir / f"box_{box_idx}_flatten.png"
                endpoints_path = out_dir / f"box_{box_idx}_endpoints.png"
                cv2.imwrite(str(selected_face_path),
                            face.get("selected_face_debug", box["rgb_crop"]))
                cv2.imwrite(str(flatten_path), face["image"])
                if "endpoints_image" in info:
                    cv2.imwrite(str(endpoints_path), info["endpoints_image"])
                else:
                    cv2.imwrite(str(endpoints_path), face["image"])

                artifacts.update({
                    "selected_face": self._rel_path(selected_face_path, out_dir),
                    "flatten": self._rel_path(flatten_path, out_dir),
                    "endpoints": self._rel_path(endpoints_path, out_dir),
                })

                result = self._face_result_record(face, label, info, face_loc)
                selected_face_summary = {
                    "label": label,
                    "confidence": float(info.get("confidence", 0.0)),
                    "score": face.get("selection_score", []),
                    "symbol_area": int(face.get("warped_symbol_area", 0)),
                    "quad_area": face.get("quad_quality", {}).get("area"),
                    "quad_aspect": face.get("quad_quality", {}).get("aspect"),
                    "distance_ok": bool(result.get("distance_ok", False)),
                }
                selected_face = selected_face_summary
                if result["prediction_accepted"]:
                    status = "accepted"
                    verdict = label
                    accepted_count += 1
                elif self._is_predictive_label(label):
                    reject_reason = "face_distance_mismatch"
                else:
                    reject_reason = "no_classifiable_face"
            else:
                symbol_mask = self._symbol_mask(box["rgb_crop"], box["mask"])
                groups = self._symbol_groups(symbol_mask)
                reject_reason = "no_flattenable_face" if groups else "no_symbol_group"

            if status == "accepted":
                draw_color = (255, 0, 0) if box.get("color") == "BLUE" else (0, 0, 255)
            else:
                draw_color = (128, 128, 128)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), draw_color, 2)
            bx, by = loc_info["crop_center_pixel"]
            cv2.drawMarker(overlay, (int(round(bx)), int(round(by))),
                           draw_color, cv2.MARKER_CROSS, 12, 2)
            label_text = f"box {box_idx} {box.get('color', '')}".strip()
            if status == "rejected":
                label_text += f" rejected:{reject_reason}"
            else:
                label_text += f" {verdict}"
            if loc and distance is not None:
                label_text += f" z={loc[2]:.2f}m d={distance:.2f}m"
            cv2.putText(overlay, label_text, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1,
                        cv2.LINE_AA)

            boxes_out.append({
                "box_index": int(box_idx),
                "bbox": [int(v) for v in box["bbox"]],
                "color": box.get("color"),
                "status": status,
                "verdict": verdict,
                "reject_reason": reject_reason,
                "location_xyz_m": [float(v) for v in loc] if loc else None,
                "distance_m": distance,
                "selected_face": selected_face,
                "artifacts": artifacts,
            })

        cv2.imwrite(str(overlay_path), overlay)

        result = {
            "frame_index": int(frame_idx),
            "requested_timestamp_s": (
                float(requested_time_s) if requested_time_s is not None else None
            ),
            "selected_timestamp_s": (
                float(timestamp_s) if timestamp_s is not None else None
            ),
            "n_candidates": int(len(boxes_out)),
            "n_boxes": int(accepted_count),
            "artifacts": {
                "color": self._rel_path(color_path, out_dir),
                "overlay": self._rel_path(overlay_path, out_dir),
            },
            "boxes": boxes_out,
        }
        result_path = out_dir / "result.json"
        self.save(result, result_path)
        return result

    def _merge_boxes(self, all_records):
        """Merge the same box seen across frames by camera-XYZ proximity."""
        boxes = []   # each: {locations:[...], faces:[...]}
        for rec in all_records:
            loc = rec["location_xyz_m"]
            if loc is None:
                continue
            predictive_faces = [
                f for f in rec.get("faces", [])
                if f.get("label") in ("REAL", "FAKE") and
                f.get("prediction_accepted", True)
            ]
            if not predictive_faces:
                continue
            match = None
            for b in boxes:
                ref = np.mean(b["locations"], axis=0)
                if np.linalg.norm(np.array(loc) - ref) < self.assoc_dist:
                    match = b
                    break
            if match is None:
                match = {"locations": [], "faces": []}
                boxes.append(match)
            match["locations"].append(loc)
            match["faces"].extend(predictive_faces)
        out = []
        for b in boxes:
            center = np.mean(b["locations"], axis=0).tolist()
            center_distance = self._distance_xyz(center)
            labels = [f["label"] for f in b["faces"]]
            real_votes = [f for f in b["faces"] if f["label"] == "REAL"]
            fake_votes = [f for f in b["faces"] if f["label"] == "FAKE"]
            real_score = sum(float(f.get("confidence", 1.0)) for f in real_votes)
            fake_score = sum(float(f.get("confidence", 1.0)) for f in fake_votes)
            if real_score > fake_score and real_votes:
                verdict = "REAL"
            elif fake_score > real_score and fake_votes:
                verdict = "FAKE"
            else:
                verdict = "NONE"
            out.append({
                "center_xyz_m": center,
                "center_distance_m": center_distance,
                "n_observations": len(b["locations"]),
                "verdict": verdict,
                "face_labels": labels,
                "real_votes": len(real_votes),
                "fake_votes": len(fake_votes),
                "real_score": real_score,
                "fake_score": fake_score,
            })
        return out

    def run(self, source, out_path="boxes.json"):
        """Loop a RealSenseSource, process every frame, save merged boxes."""
        all_records = []
        for i, (rgb, depth, intr, depth_scale) in enumerate(source.frames()):
            all_records.extend(self.process_frame(rgb, depth, intr, depth_scale, i))
        boxes = self._merge_boxes(all_records)
        self.save({"boxes": boxes, "frames": all_records}, out_path)
        return boxes

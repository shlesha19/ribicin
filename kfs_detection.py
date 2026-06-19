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
        self.symbol_border_margin = cfg.get("symbol_border_margin", 14)
        self.face_pad_frac = cfg.get("face_pad_frac", 0.45)

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
        self.sharp_end_angle_thresh = cfg.get("sharp_end_angle_thresh", 85)
        self.endpoint_branch_ignore_radius = cfg.get("endpoint_branch_ignore_radius", 10)
        self.endpoint_tip_radius = cfg.get("endpoint_tip_radius", 10)

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
        color_mask = self._color_masks(bgr)["ALL"]
        if object_mask is None:
            valid = np.ones(gray.shape, bool)
        else:
            valid = object_mask > 0

        non_box = valid & (color_mask == 0)
        white = (gray > 135) & non_box
        black = (gray < 70) & non_box
        mask = (white | black).astype(np.uint8) * 255
        mask = self._clean_binary(mask, open_size=3, close_size=5)

        cleaned = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        H, W = mask.shape
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_symbol_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            m = self.symbol_border_margin
            if x <= m or y <= m or x + w >= W - m or y + h >= H - m:
                continue
            cv2.drawContours(cleaned, [cnt], -1, 255, cv2.FILLED)
        return cleaned

    def _symbol_center(self, symbol_mask):
        contours, _ = cv2.findContours(symbol_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area < self.min_symbol_area:
            return None, area
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            x, y, w, h = cv2.boundingRect(cnt)
            return (x + w / 2.0, y + h / 2.0), area
        return (m["m10"] / m["m00"], m["m01"] / m["m00"]), area

    def _face_edges(self, crop, object_mask):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=object_mask)
        masked = cv2.GaussianBlur(masked, (3, 3), 0)
        edges = cv2.Canny(masked, 50, 140)
        edges = cv2.bitwise_and(edges, edges, mask=object_mask)
        return edges

    def _quad_from_edges(self, edges, symbol_mask, object_mask):
        center, symbol_area = self._symbol_center(symbol_mask)
        if center is None:
            return None, "no_symbol", symbol_area
        cx, cy = center
        H, W = edges.shape

        line_edges = edges.copy()
        symbol_zone = cv2.dilate(
            symbol_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        line_edges[symbol_zone > 0] = 0

        lines = cv2.HoughLinesP(line_edges, 1, np.pi / 180, threshold=20,
                                minLineLength=max(12, min(W, H) // 5),
                                maxLineGap=12)
        left = right = top = bottom = None
        if lines is not None:
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = [float(v) for v in line]
                dx, dy = x2 - x1, y2 - y1
                length = np.hypot(dx, dy)
                if length < 10:
                    continue
                if abs(dx) < abs(dy) * 0.55:
                    x = (x1 + x2) / 2.0
                    if x < cx and (left is None or x < left):
                        left = x
                    if x > cx and (right is None or x > right):
                        right = x
                elif abs(dy) < abs(dx) * 0.55:
                    y = (y1 + y2) / 2.0
                    if y < cy and (top is None or y < top):
                        top = y
                    if y > cy and (bottom is None or y > bottom):
                        bottom = y

        if None not in (left, right, top, bottom):
            margin = 4.0
            quad = np.array([
                [max(0, left - margin), max(0, top - margin)],
                [min(W - 1, right + margin), max(0, top - margin)],
                [min(W - 1, right + margin), min(H - 1, bottom + margin)],
                [max(0, left - margin), min(H - 1, bottom + margin)],
            ], dtype=np.float32)
            return self._order_quad(quad), "edge_lines", symbol_area

        contours, _ = cv2.findContours(symbol_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        else:
            x, y, w, h = cv2.boundingRect(object_mask)
        pad = int(max(w, h) * self.face_pad_frac)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W - 1, x + w + pad), min(H - 1, y + h + pad)
        quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        dtype=np.float32)
        return quad, "symbol_rect_fallback", symbol_area

    def _warp_quad(self, crop, quad):
        S = self.out_size
        dst = np.array([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(self._order_quad(quad), dst)
        return cv2.warpPerspective(crop, M, (S, S))

    def _quad_debug_image(self, crop, quad, symbol_mask, edges):
        img = crop.copy()
        if quad is not None:
            q = np.round(quad).astype(int)
            cv2.polylines(img, [q], True, (0, 255, 255), 2)
        img[edges > 0] = (0, 255, 0)
        img[symbol_mask > 0] = (255, 255, 255)
        return img

    def flatten_faces(self, box, intr, depth_scale=0.001):
        """Rectify the visible marked face with a simple 2D perspective warp.

        Returns a list of face dicts:
            {image, quad, method, clarity_score, symbol_area}
        """
        crop = box["rgb_crop"]
        object_mask = box["mask"]
        clarity = self.clarity_score(crop)
        if clarity < self.min_box_clarity:
            return []

        symbol_mask = self._symbol_mask(crop, object_mask)
        edges = self._face_edges(crop, object_mask)
        quad, method, symbol_area = self._quad_from_edges(
            edges, symbol_mask, object_mask)
        if quad is None:
            return []

        image = self._warp_quad(crop, quad)
        return [{
            "image": image,
            "quad": [[float(x), float(y)] for x, y in quad],
            "method": method,
            "clarity_score": clarity,
            "symbol_area": float(symbol_area),
            "symbol_mask": symbol_mask,
            "edges": edges,
            "quad_debug": self._quad_debug_image(crop, quad, symbol_mask, edges),
        }]

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

    def _endpoint_debug_image(self, face_img, symbol_mask, skeleton,
                              endpoints, branches, sharp_points):
        dbg = face_img.copy()
        dbg[skeleton > 0] = (0, 255, 255)
        for x, y in branches:
            cv2.circle(dbg, (int(x), int(y)), 4, (0, 0, 255), -1)
        for x, y in endpoints:
            cv2.circle(dbg, (int(x), int(y)), 4, (0, 255, 0), -1)
        for x, y in sharp_points:
            cv2.circle(dbg, (int(x), int(y)), 7, (0, 255, 255), 2)
        return dbg

    def classify_face(self, face_img):
        """Classify one flattened face by sharpness of true symbol endpoints.

        label in {REAL, FAKE, NONE}. info has valid_count, n_corners.
        """
        if face_img is None or face_img.size == 0:
            return "NONE", {
                "valid_count": 0,
                "n_corners": 0,
                "sharp_end_count": 0,
                "round_end_count": 0,
                "branch_count": 0,
                "endpoint_count": 0,
            }

        symbol_mask = self._symbol_mask(face_img)
        if cv2.countNonZero(symbol_mask) < self.min_symbol_area:
            return "NONE", {
                "valid_count": 0,
                "n_corners": 0,
                "sharp_end_count": 0,
                "round_end_count": 0,
                "branch_count": 0,
                "endpoint_count": 0,
                "symbol_mask": symbol_mask,
            }

        skeleton = self._skeletonize(symbol_mask)
        skel_bool = skeleton > 0
        kernel = np.ones((3, 3), np.uint8)
        neighbor_count = cv2.filter2D(skel_bool.astype(np.uint8), -1, kernel)
        neighbor_count = neighbor_count - skel_bool.astype(np.uint8)

        endpoint_mask = skel_bool & (neighbor_count == 1)
        branch_mask = skel_bool & (neighbor_count >= 3)
        branch_u8 = branch_mask.astype(np.uint8) * 255
        if cv2.countNonZero(branch_u8) > 0:
            r = self.endpoint_branch_ignore_radius
            branch_zone = cv2.dilate(
                branch_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * r + 1, 2 * r + 1)))
            endpoint_mask &= branch_zone == 0

        ey, ex = np.nonzero(endpoint_mask)
        by, bx = np.nonzero(branch_mask)
        endpoints = list(zip(ex.tolist(), ey.tolist()))
        branches = list(zip(bx.tolist(), by.tolist()))

        sharp_points = []
        round_count = 0
        for pt in endpoints:
            angle = self._endpoint_angle(symbol_mask, pt)
            if angle is not None and angle <= self.sharp_end_angle_thresh:
                sharp_points.append(pt)
            else:
                round_count += 1

        if not endpoints:
            label = "NONE"
        elif sharp_points:
            label = "REAL"
        else:
            label = "FAKE"

        endpoint_debug = self._endpoint_debug_image(
            face_img, symbol_mask, skeleton, endpoints, branches, sharp_points)
        return label, {
            "valid_count": len(sharp_points),
            "n_corners": len(endpoints),
            "sharp_end_count": len(sharp_points),
            "round_end_count": int(round_count),
            "branch_count": len(branches),
            "endpoint_count": len(endpoints),
            "symbol_mask": symbol_mask,
            "skeleton": skeleton,
            "endpoints_image": endpoint_debug,
        }

    # ════════════════════════════════════════════════════════════════════════
    # 4. FIND LOCATION  (camera-relative XYZ, meters)
    # ════════════════════════════════════════════════════════════════════════
    def find_location(self, box, intr, depth_scale=0.001):
        """Deproject the box's depth centroid to camera-relative (X, Y, Z) m."""
        x, y, _, _ = box["bbox"]
        sub = box["depth_crop"].astype(np.float32) * depth_scale
        ys, xs = np.nonzero((sub > self.min_depth) &
                            (sub < self.max_depth) &
                            (box["mask"] > 0))
        if len(xs) == 0:
            return None
        z = float(np.median(sub[ys, xs]))
        u = float(np.mean(xs)) + x
        v = float(np.mean(ys)) + y
        X = (u - intr["ppx"]) * z / intr["fx"]
        Y = (v - intr["ppy"]) * z / intr["fy"]
        return (X, Y, z)

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
            loc = self.find_location(box, intr, depth_scale)
            faces = self.flatten_faces(box, intr, depth_scale)
            face_results = []
            for face in faces:
                label, info = self.classify_face(face["image"])
                face_results.append({
                    "label": label,
                    "valid_count": int(info["valid_count"]),
                    "sharp_end_count": int(info["sharp_end_count"]),
                    "round_end_count": int(info["round_end_count"]),
                    "branch_count": int(info["branch_count"]),
                    "endpoint_count": int(info["endpoint_count"]),
                    "quad": face["quad"],
                    "method": face["method"],
                    "clarity_score": float(face["clarity_score"]),
                    "symbol_area": float(face["symbol_area"]),
                })
            records.append({
                "frame": frame_idx,
                "frame_clarity": frame_clarity,
                "bbox": list(box["bbox"]),
                "color": box.get("color"),
                "location_xyz_m": list(loc) if loc else None,
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
        depth_path = out_dir / "depth.png"
        overlay_path = out_dir / "boxes_overlay.png"
        blue_mask_path = out_dir / "blue_mask.png"
        red_mask_path = out_dir / "red_mask.png"
        color_mask_path = out_dir / "color_mask.png"
        valid_depth_mask_path = out_dir / "valid_depth_mask.png"

        cv2.imwrite(str(color_path), rgb)
        cv2.imwrite(str(depth_path), self._depth_visual(depth, depth_scale=depth_scale))
        masks = self._color_masks(rgb)
        valid_depth = self._valid_depth_mask(depth, depth_scale)
        cv2.imwrite(str(blue_mask_path), masks["BLUE"])
        cv2.imwrite(str(red_mask_path), masks["RED"])
        cv2.imwrite(str(color_mask_path), masks["ALL"])
        cv2.imwrite(str(valid_depth_mask_path), valid_depth)

        overlay = rgb.copy()
        boxes_out = []
        for box_idx, box in enumerate(self.detect_boxes(rgb, depth, depth_scale)):
            loc = self.find_location(box, intr, depth_scale)
            faces = self.flatten_faces(box, intr, depth_scale)

            x, y, w, h = box["bbox"]
            crop_path = out_dir / f"box_{box_idx}_crop.png"
            mask_path = out_dir / f"box_{box_idx}_mask.png"
            object_mask_path = out_dir / f"box_{box_idx}_object_mask.png"
            box_depth_path = out_dir / f"box_{box_idx}_depth.png"
            symbol_mask_path = out_dir / f"box_{box_idx}_symbol_mask.png"
            edges_path = out_dir / f"box_{box_idx}_edges.png"
            quad_path = out_dir / f"box_{box_idx}_face_quad.png"
            cv2.imwrite(str(crop_path), box["rgb_crop"])
            cv2.imwrite(str(mask_path), box["mask"])
            cv2.imwrite(str(object_mask_path), box["mask"])
            cv2.imwrite(str(box_depth_path),
                        self._depth_visual(box["depth_crop"], box["mask"], depth_scale))

            face_results = []
            face_labels = []
            for face_idx, face in enumerate(faces):
                label, info = self.classify_face(face["image"])
                face_path = out_dir / f"box_{box_idx}_face_{face_idx}.png"
                face_symbol_path = (
                    out_dir / f"box_{box_idx}_face_{face_idx}_symbol_mask.png")
                endpoints_path = (
                    out_dir / f"box_{box_idx}_face_{face_idx}_endpoints.png")
                cv2.imwrite(str(face_path), face["image"])
                cv2.imwrite(str(face_symbol_path), info.get(
                    "symbol_mask", np.zeros(face["image"].shape[:2], np.uint8)))
                if "endpoints_image" in info:
                    cv2.imwrite(str(endpoints_path), info["endpoints_image"])
                else:
                    cv2.imwrite(str(endpoints_path), face["image"])
                face_labels.append(label)
                face_results.append({
                    "label": label,
                    "valid_count": int(info["valid_count"]),
                    "n_corners": int(info["n_corners"]),
                    "sharp_end_count": int(info["sharp_end_count"]),
                    "round_end_count": int(info["round_end_count"]),
                    "branch_count": int(info["branch_count"]),
                    "endpoint_count": int(info["endpoint_count"]),
                    "quad": face["quad"],
                    "method": face["method"],
                    "clarity_score": float(face["clarity_score"]),
                    "symbol_area": float(face["symbol_area"]),
                    "image_path": self._rel_path(face_path, out_dir),
                    "symbol_mask_path": self._rel_path(face_symbol_path, out_dir),
                    "endpoints_path": self._rel_path(endpoints_path, out_dir),
                })

            if faces:
                cv2.imwrite(str(symbol_mask_path), faces[0]["symbol_mask"])
                cv2.imwrite(str(edges_path), faces[0]["edges"])
                cv2.imwrite(str(quad_path), faces[0]["quad_debug"])
            else:
                cv2.imwrite(str(symbol_mask_path),
                            self._symbol_mask(box["rgb_crop"], box["mask"]))
                cv2.imwrite(str(edges_path),
                            self._face_edges(box["rgb_crop"], box["mask"]))
                cv2.imwrite(str(quad_path), box["rgb_crop"])

            draw_color = (255, 0, 0) if box.get("color") == "BLUE" else (0, 0, 255)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), draw_color, 2)
            label_text = f"box {box_idx} {box.get('color', '')}".strip()
            if face_labels:
                label_text += " " + "/".join(face_labels)
            if loc:
                label_text += f" z={loc[2]:.2f}m"
            cv2.putText(overlay, label_text, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1,
                        cv2.LINE_AA)

            boxes_out.append({
                "box_index": int(box_idx),
                "bbox": [int(v) for v in box["bbox"]],
                "color": box.get("color"),
                "object_depth_m": float(box["object_depth_m"]),
                "location_xyz_m": [float(v) for v in loc] if loc else None,
                "faces": face_results,
                "artifacts": {
                    "crop": self._rel_path(crop_path, out_dir),
                    "mask": self._rel_path(mask_path, out_dir),
                    "object_mask": self._rel_path(object_mask_path, out_dir),
                    "depth": self._rel_path(box_depth_path, out_dir),
                    "symbol_mask": self._rel_path(symbol_mask_path, out_dir),
                    "edges": self._rel_path(edges_path, out_dir),
                    "face_quad": self._rel_path(quad_path, out_dir),
                },
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
            "initial_frame_index": (
                int(initial_frame_idx) if initial_frame_idx is not None else int(frame_idx)
            ),
            "initial_timestamp_s": (
                float(initial_timestamp_s)
                if initial_timestamp_s is not None
                else (float(timestamp_s) if timestamp_s is not None else None)
            ),
            "initial_clarity": (
                float(initial_clarity)
                if initial_clarity is not None
                else float(frame_clarity)
            ),
            "selected_clarity": float(frame_clarity),
            "min_frame_clarity": float(self.min_frame_clarity),
            "skipped_blurry_frames": int(skipped_blurry_frames),
            "n_boxes": int(len(boxes_out)),
            "artifacts": {
                "color": self._rel_path(color_path, out_dir),
                "depth": self._rel_path(depth_path, out_dir),
                "boxes_overlay": self._rel_path(overlay_path, out_dir),
                "blue_mask": self._rel_path(blue_mask_path, out_dir),
                "red_mask": self._rel_path(red_mask_path, out_dir),
                "color_mask": self._rel_path(color_mask_path, out_dir),
                "valid_depth_mask": self._rel_path(valid_depth_mask_path, out_dir),
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
            match["faces"].extend(rec["faces"])
        out = []
        for b in boxes:
            center = np.mean(b["locations"], axis=0).tolist()
            labels = [f["label"] for f in b["faces"]]
            verdict = "REAL" if labels.count("REAL") >= labels.count("FAKE") else "FAKE"
            out.append({
                "center_xyz_m": center,
                "n_observations": len(b["locations"]),
                "verdict": verdict,
                "face_labels": labels,
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

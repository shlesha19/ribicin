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

Pipeline (per frame):
    box_detect   -> crop each box (rgb + depth)
    flatten      -> rectify each visible face to a head-on view
    classify_face-> REAL / FAKE / NONE per face   (corner logic from prototypes)
    localise     -> camera-relative (X, Y, Z) in meters
    save         -> JSON log / cache

Needs numpy + opencv always; pyrealsense2 only for live/.bag sources (imported
lazily so the 2D classify path still runs on a dev machine without the SDK).
"""

import json

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
                 width=640, height=480, fps=30):
        import pyrealsense2 as rs   # lazy: only needed for a real source
        self.rs = rs

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
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    def frames(self):
        """Generator of (rgb_bgr, depth_uint16, intrinsics, depth_scale)."""
        try:
            while True:
                try:
                    frames = self.pipeline.wait_for_frames()
                except RuntimeError:
                    break  # end of .bag
                frames = self.align.process(frames)
                depth_f = frames.get_depth_frame()
                color_f = frames.get_color_frame()
                if not depth_f or not color_f:
                    continue
                i = color_f.profile.as_video_stream_profile().intrinsics
                intr = {"fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy}
                rgb = np.asanyarray(color_f.get_data())
                depth = np.asanyarray(depth_f.get_data())
                yield rgb, depth, intr, self.depth_scale
        finally:
            self.close()

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class KFSDetector:
    def __init__(self, **cfg):
        # --- box detection (depth segmentation), meters ---
        self.min_depth      = cfg.get("min_depth", 0.2)    # ignore closer than this
        self.max_depth      = cfg.get("max_depth", 3.0)    # ignore farther than this
        self.box_min_area   = cfg.get("box_min_area", 2000)  # px², min box blob

        # --- face flattening ---
        self.plane_tol      = cfg.get("plane_tol", 0.02)   # m, plane membership
        self.min_face_px    = cfg.get("min_face_px", 800)  # min pixels per face
        self.out_size       = cfg.get("out_size", 256)     # rectified face size (px)
        self.normal_cache_deg = cfg.get("normal_cache_deg", 5.0)  # reuse remap if <this

        # --- real/fake (2D corner logic, ported from the prototypes) ---
        self.stroke_thresh  = cfg.get("stroke_thresh", 80)
        self.angle_thresh   = cfg.get("angle_thresh", 100)
        self.epsilon_frac   = cfg.get("epsilon_frac", 0.008)
        self.circle_radius  = cfg.get("circle_radius", 20)
        self.n_samples      = cfg.get("n_samples", 100)
        self.white_min      = cfg.get("white_min", 0.70)
        self.white_max      = cfg.get("white_max", 0.80)
        self.count_thresh   = cfg.get("count_thresh", 5)
        self.kfs_min_area   = cfg.get("kfs_min_area", 4000)
        self.dedup_dist     = cfg.get("dedup_dist", 5)
        self.close_kernel   = cfg.get("close_kernel", 15)

        # --- localisation / association ---
        self.assoc_dist     = cfg.get("assoc_dist", 0.30)  # m, same-box matching

        self._flatten_cache = {}   # face-slot -> (normal, map_x, map_y)

    # ════════════════════════════════════════════════════════════════════════
    # 1. BOX DETECTION  (depth-based segmentation)
    # ════════════════════════════════════════════════════════════════════════
    def box_detect(self, rgb, depth, depth_scale=0.001):
        """Find boxes as depth foreground blobs.

        Returns a list of box dicts:
            {bbox:(x,y,w,h), rgb_crop, depth_crop, mask}   (mask is full-frame)
        """
        depth_m = depth.astype(np.float32) * depth_scale
        valid = (depth_m > self.min_depth) & (depth_m < self.max_depth)
        fg = (valid.astype(np.uint8)) * 255

        # clean up speckle, close small gaps
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.box_min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros(depth.shape[:2], np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
            boxes.append({
                "bbox": (x, y, w, h),
                "rgb_crop": rgb[y:y + h, x:x + w].copy(),
                "depth_crop": depth[y:y + h, x:x + w].copy(),
                "mask": mask,
            })
        return boxes

    # ════════════════════════════════════════════════════════════════════════
    # 2. FLATTEN  (split into faces, rectify each to head-on view)
    # ════════════════════════════════════════════════════════════════════════
    def _deproject(self, depth_crop, intr, bbox, depth_scale):
        """Crop depth -> Nx3 cloud (m) + the (u,v) pixel of each valid point."""
        x0, y0, _, _ = bbox
        ys, xs = np.nonzero(depth_crop > 0)
        z = depth_crop[ys, xs].astype(np.float32) * depth_scale
        u = xs + x0
        v = ys + y0
        X = (u - intr["ppx"]) * z / intr["fx"]
        Y = (v - intr["ppy"]) * z / intr["fy"]
        pts = np.stack([X, Y, z], axis=1)
        uv = np.stack([xs, ys], axis=1)        # local crop coords
        return pts, uv

    def _segment_faces(self, pts, uv):
        """Greedy plane clustering: peel off the largest plane repeatedly.

        Returns a list of (plane_pts, plane_uv) groups, each a visible face.
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
    def _rodrigues_to_z(normal):
        """Rotation aligning `normal` with +Z."""
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

    def _build_remap(self, P, uv, R, centroid):
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

    def flatten(self, box, intr, depth_scale=0.001, cache_key=None):
        """Rectify each visible face of `box` to a head-on image.

        Returns a list of face dicts:
            {image, normal, centroid, n_points}
        """
        pts, uv = self._deproject(box["depth_crop"], intr, box["bbox"], depth_scale)
        if len(pts) < self.min_face_px:
            return []

        faces_out = []
        for i, (P, fuv, normal, centroid) in enumerate(self._segment_faces(pts, uv)):
            slot = (cache_key, i) if cache_key is not None else None
            R = self._rodrigues_to_z(normal)

            cached = self._flatten_cache.get(slot) if slot else None
            if cached is not None:
                c_norm, map_x, map_y = cached
                ang = np.degrees(np.arccos(
                    np.clip(abs(c_norm @ (normal / (np.linalg.norm(normal) + 1e-9))), -1, 1)))
                if ang > self.normal_cache_deg:
                    cached = None
            if cached is None:
                map_x, map_y = self._build_remap(P, fuv, R, centroid)
                if slot:
                    self._flatten_cache[slot] = (
                        normal / (np.linalg.norm(normal) + 1e-9), map_x, map_y)

            image = cv2.remap(box["rgb_crop"], map_x, map_y, cv2.INTER_LINEAR)
            faces_out.append({
                "image": image,
                "normal": normal,
                "centroid": centroid,
                "n_points": len(P),
            })
        return faces_out

    # ════════════════════════════════════════════════════════════════════════
    # 3. REAL / FAKE  (2D corner/anchor logic ported from the prototypes)
    # ════════════════════════════════════════════════════════════════════════
    def _binary(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, self.stroke_thresh, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (self.close_kernel, self.close_kernel))
        closed = cv2.morphologyEx(cv2.bitwise_not(b), cv2.MORPH_CLOSE, k)
        return cv2.bitwise_not(closed)

    def _corners(self, binary):
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

    def _dedup(self, pts):
        kept = []
        for px, py in pts:
            if all(abs(px - kx) > self.dedup_dist or abs(py - ky) > self.dedup_dist
                   for kx, ky in kept):
                kept.append((px, py))
        return kept

    def classify_face(self, face_img):
        """Classify one flattened face. Returns (label, info).

        label in {REAL, FAKE, NONE}. info has valid_count, n_corners.
        """
        if face_img is None or face_img.size == 0:
            return "NONE", {"valid_count": 0, "n_corners": 0}
        binary = self._binary(face_img)
        corners = self._dedup(self._corners(binary))
        if not corners:
            return "REAL", {"valid_count": 0, "n_corners": 0}  # smooth, no sharp edges
        valid = 0
        for cx, cy in corners:
            wp, ok = self._check_corner(binary, cx, cy)
            if ok and self.white_min <= wp <= self.white_max:
                valid += 1
        label = "REAL" if valid > self.count_thresh else "FAKE"
        return label, {"valid_count": valid, "n_corners": len(corners)}

    # ════════════════════════════════════════════════════════════════════════
    # 4. LOCALISE  (camera-relative XYZ, meters)
    # ════════════════════════════════════════════════════════════════════════
    def localise(self, box, depth, intr, depth_scale=0.001):
        """Deproject the box's depth centroid to camera-relative (X, Y, Z) m."""
        x, y, w, h = box["bbox"]
        sub = box["depth_crop"].astype(np.float32) * depth_scale
        ys, xs = np.nonzero((sub > self.min_depth) & (sub < self.max_depth))
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
        records = []
        for bi, box in enumerate(self.box_detect(rgb, depth, depth_scale)):
            loc = self.localise(box, depth, intr, depth_scale)
            faces = self.flatten(box, intr, depth_scale, cache_key=(frame_idx, bi))
            face_results = []
            for fi, face in enumerate(faces):
                label, info = self.classify_face(face["image"])
                face_results.append({
                    "label": label,
                    "valid_count": info["valid_count"],
                    "normal": [float(c) for c in face["normal"]],
                })
            records.append({
                "frame": frame_idx,
                "bbox": list(box["bbox"]),
                "location_xyz_m": list(loc) if loc else None,
                "faces": face_results,
            })
        return records

    def _associate(self, all_records):
        """Merge same box across frames by camera-XYZ proximity."""
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
        """Loop a RealSenseSource, process every frame, save associated boxes."""
        all_records = []
        for i, (rgb, depth, intr, depth_scale) in enumerate(source.frames()):
            all_records.extend(self.process_frame(rgb, depth, intr, depth_scale, i))
        boxes = self._associate(all_records)
        self.save({"boxes": boxes, "frames": all_records}, out_path)
        return boxes

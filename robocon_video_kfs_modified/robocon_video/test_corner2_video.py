"""
test_corner2_video.py  –  pure segment-logic unit test.
No file I/O; logic is identical to the original test_corner2.py.
"""

import numpy as np


def check_segment(segments, n_samples, angle_thresh):
    if len(segments) == 1:
        seg = segments[0]
        width_deg = len(seg) * 360 / n_samples
        return width_deg <= angle_thresh + 15
    elif len(segments) >= 2:
        segment_angles = []
        for seg in segments:
            sin_sum = sum(np.sin(i * 2 * np.pi / n_samples) for i in seg)
            cos_sum = sum(np.cos(i * 2 * np.pi / n_samples) for i in seg)
            mean_ang = np.arctan2(sin_sum, cos_sum)
            segment_angles.append(mean_ang)

        valid_slope = False
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
        return valid_slope
    return False


# ── tests ─────────────────────────────────────────────────────────────────────
print(check_segment([[0, 1, 2, 3, 4, 5, 56, 57, 58, 59, 60, 61, 62, 63]], 64, 100))
print(check_segment([[21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]], 64, 100))

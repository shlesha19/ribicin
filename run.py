"""
Run the KFS detection pipeline on a .bag recording or the live camera.

Examples
--------
    python run.py --bag clip.bag                 # testing on a recording (3 fps)
    python run.py --bag clip.bag --fps 5         # different sampling rate
    python run.py --live                         # live RealSense camera
    python run.py --bag clip.bag --out result.json
    python run.py --bag clip.bag --debug-time 38 # inspect one frame in debug/

Needs pyrealsense2 (installed on the Jetson) for the camera/.bag source.
"""

import argparse
from pathlib import Path

from kfs_detection import KFSDetector, RealSenseSource


def _debug_time_label(seconds):
    label = f"{seconds:.3f}".replace("-", "neg_").replace(".", "_")
    return f"frame_{label}s"


def _make_debug_dir(seconds):
    base = Path("debug") / _debug_time_label(seconds)
    path = base
    n = 1
    while path.exists():
        path = base.with_name(f"{base.name}_{n}")
        n += 1
    return path


def _closest_bag_frame(bag_path, target_s):
    source = RealSenseSource(testing=True, bag_path=bag_path, sample_fps=0.0)
    best = None
    best_delta = None
    for rec in source.frame_records():
        delta = abs(rec["timestamp_s"] - target_s)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = dict(rec)
            best["rgb"] = rec["rgb"].copy()
            best["depth"] = rec["depth"].copy()
    return best


def main():
    p = argparse.ArgumentParser(description="Run KFS box detection.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bag", help="path to a .bag recording (testing mode)")
    src.add_argument("--live", action="store_true", help="use the live RealSense camera")
    p.add_argument("--fps", type=float, default=None,
                   help="frames per second to process "
                        "(default 3 for --bag, full rate for --live)")
    p.add_argument("--out", default="boxes.json", help="output JSON path")
    p.add_argument("--debug-time", type=float, default=None,
                   help="process only the bag frame closest to this timestamp "
                        "in seconds and save debug artifacts under debug/")
    args = p.parse_args()

    if args.debug_time is not None:
        if not args.bag:
            p.error("--debug-time requires --bag")

        rec = _closest_bag_frame(args.bag, args.debug_time)
        if rec is None:
            p.error(f"no aligned color/depth frames found in {args.bag}")

        det = KFSDetector()
        out_dir = _make_debug_dir(args.debug_time)
        result = det.debug_process_frame(
            rec["rgb"],
            rec["depth"],
            rec["intrinsics"],
            rec["depth_scale"],
            frame_idx=rec["frame_index"],
            timestamp_s=rec["timestamp_s"],
            requested_time_s=args.debug_time,
            out_dir=out_dir,
        )
        print(f"\nDebug frame saved to {out_dir}")
        print(f"Requested {args.debug_time:.3f}s, selected "
              f"{rec['timestamp_s']:.3f}s at frame {rec['frame_index']}")
        print(f"Found {result['n_boxes']} box(es). Result: {out_dir / 'result.json'}")
        return

    source = RealSenseSource(testing=bool(args.bag), bag_path=args.bag,
                             sample_fps=args.fps)
    det = KFSDetector()
    boxes = det.run(source, args.out)

    print(f"\nFound {len(boxes)} box(es). Results saved to {args.out}")
    for i, b in enumerate(boxes):
        x, y, z = b["center_xyz_m"]
        print(f"  box {i}: {b['verdict']:4}  at ({x:.2f}, {y:.2f}, {z:.2f}) m  "
              f"seen {b['n_observations']}x")


if __name__ == "__main__":
    main()

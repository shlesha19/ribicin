"""
Run the KFS detection pipeline on a .bag recording or the live camera.

Examples
--------
    python run.py --bag clip.bag                 # testing on a recording (3 fps)
    python run.py --bag clip.bag --fps 5         # different sampling rate
    python run.py --live                         # live RealSense camera
    python run.py --bag clip.bag --out result.json
    python run.py --bag clip.bag --debug-time 38 # inspect one frame in debug/
    python run.py --bag clip.bag --debug-time 9 15 27

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


def _copy_frame_record(rec, det):
    out = dict(rec)
    out["rgb"] = rec["rgb"].copy()
    out["depth"] = rec["depth"].copy()
    out["clarity"] = det.clarity_score(out["rgb"])
    return out


def _debug_bag_frame(bag_path, target_s, det):
    source = RealSenseSource(testing=True, bag_path=bag_path, sample_fps=0.0)
    initial = None
    selected = None
    skipped = 0
    last = None

    for rec in source.frame_records():
        last = rec
        if initial is None and rec["timestamp_s"] < target_s:
            continue
        if initial is None:
            initial = _copy_frame_record(rec, det)

        candidate = _copy_frame_record(rec, det)
        if candidate["clarity"] >= det.min_frame_clarity:
            selected = candidate
            break
        if skipped >= det.debug_max_lookahead_frames:
            selected = candidate
            break
        skipped += 1

    if selected is None and last is not None:
        selected = _copy_frame_record(last, det)
        if initial is None:
            initial = selected

    if selected is None:
        return None

    selected["initial_frame_index"] = initial["frame_index"]
    selected["initial_timestamp_s"] = initial["timestamp_s"]
    selected["initial_clarity"] = initial["clarity"]
    selected["skipped_blurry_frames"] = skipped
    return selected


def main():
    p = argparse.ArgumentParser(description="Run KFS box detection.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bag", help="path to a .bag recording (testing mode)")
    src.add_argument("--live", action="store_true", help="use the live RealSense camera")
    p.add_argument("--fps", type=float, default=None,
                   help="frames per second to process "
                        "(default 3 for --bag, full rate for --live)")
    p.add_argument("--out", default="boxes.json", help="output JSON path")
    p.add_argument("--debug-time", type=float, nargs="+", default=None,
                   help="process one or more bag timestamps in seconds; "
                        "if a frame is blurry, use the next clear frame and "
                        "save debug artifacts under debug/")
    args = p.parse_args()

    if args.debug_time is not None:
        if not args.bag:
            p.error("--debug-time requires --bag")

        det = KFSDetector()
        for target_s in args.debug_time:
            rec = _debug_bag_frame(args.bag, target_s, det)
            if rec is None:
                p.error(f"no aligned color/depth frames found in {args.bag}")

            out_dir = _make_debug_dir(target_s)
            result = det.debug_process_frame(
                rec["rgb"],
                rec["depth"],
                rec["intrinsics"],
                rec["depth_scale"],
                frame_idx=rec["frame_index"],
                timestamp_s=rec["timestamp_s"],
                requested_time_s=target_s,
                out_dir=out_dir,
                initial_frame_idx=rec["initial_frame_index"],
                initial_timestamp_s=rec["initial_timestamp_s"],
                initial_clarity=rec["initial_clarity"],
                skipped_blurry_frames=rec["skipped_blurry_frames"],
            )
            print(f"\nDebug frame saved to {out_dir}")
            print(f"Requested {target_s:.3f}s, selected "
                  f"{rec['timestamp_s']:.3f}s at frame {rec['frame_index']}")
            print(f"Clarity {rec['clarity']:.1f} "
                  f"(skipped {rec['skipped_blurry_frames']} blurry frame(s))")
            print(f"Found {result['n_boxes']} accepted box(es). "
                  f"Result: {out_dir / 'result.json'}")
        return

    source = RealSenseSource(testing=bool(args.bag), bag_path=args.bag,
                             sample_fps=args.fps)
    det = KFSDetector()
    boxes = det.run(source, args.out)

    print(f"\nFound {len(boxes)} box(es). Results saved to {args.out}")
    for i, b in enumerate(boxes):
        x, y, z = b["center_xyz_m"]
        d = b.get("center_distance_m")
        dist_text = f" d={d:.2f}m" if d is not None else ""
        print(f"  box {i}: {b['verdict']:4}  at ({x:.2f}, {y:.2f}, {z:.2f}) m  "
              f"{dist_text} seen {b['n_observations']}x")


if __name__ == "__main__":
    main()

"""
Run the KFS detection pipeline on a .bag recording or the live camera.

Examples
--------
    python run.py --bag clip.bag                 # testing on a recording (3 fps)
    python run.py --bag clip.bag --fps 5         # different sampling rate
    python run.py --live                         # live RealSense camera
    python run.py --bag clip.bag --out result.json

Needs pyrealsense2 (installed on the Jetson) for the camera/.bag source.
"""

import argparse

from kfs_detection import KFSDetector, RealSenseSource


def main():
    p = argparse.ArgumentParser(description="Run KFS box detection.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bag", help="path to a .bag recording (testing mode)")
    src.add_argument("--live", action="store_true", help="use the live RealSense camera")
    p.add_argument("--fps", type=float, default=None,
                   help="frames per second to process "
                        "(default 3 for --bag, full rate for --live)")
    p.add_argument("--out", default="boxes.json", help="output JSON path")
    args = p.parse_args()

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

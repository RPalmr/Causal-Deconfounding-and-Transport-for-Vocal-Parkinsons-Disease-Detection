import argparse
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import transport_sid_proxy as tsp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xfer-reps", type=int, default=200)
    ap.add_argument("--cb-repeats", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default=os.path.join(BASE, "results"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tsp.OUT = args.out_dir

    print(f"Running sID-proxy transport: xfer_reps={args.xfer_reps}, "
          f"cb_repeats={args.cb_repeats}, out_dir={tsp.OUT}")
    tsp.transfer(args.xfer_reps, args.cb_repeats)

    out_csv = os.path.join(tsp.OUT, "pipeline_transfer_sid_proxy.csv")
    if os.path.exists(out_csv):
        print(f"Wrote {out_csv}")
    else:
        print(f"WARNING: {out_csv} was not produced (all intervals cached?)")


if __name__ == "__main__":
    main()
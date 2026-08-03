"""Reorders an arbitrary CSV trace's columns into the time,obj_id,size format neuracar.trace expects."""

import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument("--time-col", type=int, required=True, help="0-indexed")
    parser.add_argument("--obj-id-col", type=int, required=True, help="0-indexed")
    parser.add_argument("--size-col", type=int, required=True, help="0-indexed")
    parser.add_argument("--has-header", action="store_true")
    args = parser.parse_args()

    with open(args.in_path, "r") as fin, open(args.out_path, "w", newline="") as fout:
        reader = csv.reader((line for line in fin if not line.startswith("#")))
        writer = csv.writer(fout)
        if args.has_header:
            next(reader)
        for row in reader:
            writer.writerow([row[args.time_col], row[args.obj_id_col], row[args.size_col]])

    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
2
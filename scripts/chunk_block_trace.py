"""Converts a variable-length block-storage I/O trace (version,time,op,size,lbn - size in bytes,
lbn a block number) into fixed-size time,obj_id,size rows the Python simulator can use: each I/O
request is split into one row per BLOCK_SIZE-byte block it touches, so every object (block) always
has the same size, satisfying AdaptiveCache's one-size-per-object assumption. This is the standard
way to turn overlapping variable-length block I/O into a cache-simulation trace."""

import argparse
import csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument("--time-col", type=int, default=1)
    parser.add_argument("--size-col", type=int, default=3)
    parser.add_argument("--lbn-col", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--has-header", action="store_true")
    args = parser.parse_args()

    n_in = n_out = 0
    with open(args.in_path, "r") as fin, open(args.out_path, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        if args.has_header:
            next(reader)
        for row in reader:
            time = row[args.time_col]
            size = int(row[args.size_col])
            lbn = int(row[args.lbn_col])
            n_in += 1
            # Truncate to whole blocks: a trailing partial block would sometimes collide with a
            # full-size block at the same address from a different request, violating the
            # simulator's one-size-per-object assumption. Dropping the leftover (< block_size
            # bytes, negligible next to the ~37KB average request) keeps every block's size constant.
            full_blocks = size // args.block_size
            for i in range(full_blocks):
                writer.writerow([time, str(lbn + i), args.block_size])
                n_out += 1

    print(f"Wrote {args.out_path}: {n_in} I/O requests -> {n_out} fixed-{args.block_size}B-block rows")


if __name__ == "__main__":
    main()

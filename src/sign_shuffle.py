"""Compare negative patterns with signs shuffled across cleaned hyperlink events."""

import argparse
import csv
from datetime import datetime
from functools import partial
from pathlib import Path

from time_shuffle import compare


def load_signed_events(path):
    """Read validated signed events from the preparation pipeline's edges CSV."""
    events = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source", "target", "timestamp", "sentiment"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("expected cleaned CSV columns: source, target, timestamp, sentiment")
        for line, row in enumerate(reader, 2):
            try:
                source = row["source"].strip()
                target = row["target"].strip()
                timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                sign = int(row["sentiment"])
                if not source or not target or source == target or sign not in (-1, 1):
                    raise ValueError("require nonempty distinct endpoints and sign -1 or 1")
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid row {line}: {exc}") from exc
            events.append((source, target, timestamp, sign))
    return events


def shuffle_signs(events, negative_count, rng):
    """Select the event positions assigned negative signs in a uniform permutation.

    Sampling positions without replacement is equivalent to shuffling binary signs.
    Repeated events remain separate positions and can receive different signs.
    """
    positions = rng.sample(range(len(events)), negative_count)
    return [events[position][:3] for position in positions]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="cleaned edges CSV with both signs")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--deltas", type=int, nargs="+", default=[1, 3, 7])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.trials < 2 or any(delta <= 0 for delta in args.deltas):
        parser.error("use at least two trials and positive window lengths")
    if args.input.resolve() == args.csv.resolve():
        parser.error("input and output must be different files")
    try:
        events = load_signed_events(args.input)
        negative = [event[:3] for event in events if event[3] == -1]
        sampler = partial(shuffle_signs, events, len(negative))
        rows = list(compare(negative, args.deltas, args.trials, args.seed, sampler=sampler))
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["delta_days", "pattern", "metric", "observed",
                             "null_mean", "null_sample_sd", "z_score", "trials", "seed"])
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()

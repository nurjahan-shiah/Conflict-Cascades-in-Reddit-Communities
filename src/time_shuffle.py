"""Compare negative hyperlink patterns with a timestamp-shuffle baseline."""

import argparse
import csv
import random
from pathlib import Path
from statistics import mean, stdev

from pattern_counts import PATTERNS, count_patterns, load_negative_events


def shuffle_timestamps(events, rng):
    """Keep each directed pair and randomly reassign the observed timestamps."""
    timestamps = [timestamp for _, _, timestamp in events]
    rng.shuffle(timestamps)
    return [(source, target, timestamp)
            for (source, target, _), timestamp in zip(events, timestamps)]


def compare(events, deltas, trials, seed, *, sampler=None):
    """Summarize observed and randomized counts for each window and metric.

    An optional sampler(rng) returns negative events for an alternative null
    model. Omitting it retains the timestamp-shuffle behavior.
    """
    if trials < 2:
        raise ValueError("at least two trials are required")
    observed = count_patterns(events, deltas)
    samples = {
        (delta, pattern, metric): []
        for delta, table in observed.items()
        for pattern in PATTERNS
        for metric in table[pattern]
    }
    rng = random.Random(seed)
    for _ in range(trials):
        randomized = sampler(rng) if sampler else shuffle_timestamps(events, rng)
        simulated = count_patterns(randomized, deltas)
        for (delta, pattern, metric), counts in samples.items():
            counts.append(simulated[delta][pattern][metric])
    for (delta, pattern, metric), counts in samples.items():
        actual = observed[delta][pattern][metric]
        center = mean(counts)
        spread = stdev(counts)
        z_score = (actual - center) / spread if spread else None
        yield delta, pattern, metric, actual, center, spread, z_score, trials, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="cleaned edges CSV")
    parser.add_argument("--csv", type=Path, required=True, help="output summary CSV")
    parser.add_argument("--deltas", type=int, nargs="+", default=[1, 3, 7])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.trials < 2 or any(delta <= 0 for delta in args.deltas):
        parser.error("use at least two trials and positive window lengths")
    if args.input.resolve() == args.csv.resolve():
        parser.error("input and output must be different files")
    try:
        events = load_negative_events(args.input)
        rows = list(compare(events, args.deltas, args.trials, args.seed))
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["delta_days", "pattern", "metric", "observed",
                             "null_mean", "null_sample_sd", "z_score", "trials", "seed"])
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()

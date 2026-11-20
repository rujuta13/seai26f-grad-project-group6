"""The one shared train/test split that every treatment (T1-T7) is scored on.

For each domain: read the MOOT csv as MOOT ships it, draw a seeded random sample of
SAMPLE rows, and cut it 80/20 into train and test. Rows are copied byte for byte (line
endings included), so no data is edited, only sampled. Same seed, same rows, forever.

    python3 harness/data.py          # writes data/splits/<domain>-{train,test}.csv
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "splits")

SEED = 1234567891
SAMPLE = 1000   # the prof's cap: at most 1000 rows, picked at random
TRAIN = 0.8

DOMAINS = {"ssw": "SS-W.csv",
           "players": "all_players.csv",
           "marketing": "Marketing_Analytics.csv"}


def split(lines, seed=SEED, sample=SAMPLE, train=TRAIN):
    """header + rows -> (train lines, test lines), each starting with the header."""
    header, rows = lines[0], lines[1:]
    picked = random.Random(seed).sample(rows, min(sample, len(rows)))
    cut = int(len(picked) * train)
    return [header] + picked[:cut], [header] + picked[cut:]


def build(domain, src_dir=DATA, out_dir=OUT):
    with open(os.path.join(src_dir, DOMAINS[domain]), newline="") as f:
        lines = f.readlines()
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for part, chunk in zip(("train", "test"), split(lines)):
        path = os.path.join(out_dir, f"{domain}-{part}.csv")
        with open(path, "w", newline="") as f:
            f.writelines(chunk)
        paths.append((path, len(chunk) - 1))
    return paths


def main(domains):
    for d in domains or DOMAINS:
        for path, n in build(d):
            print(f"{os.path.relpath(path, ROOT)}: {n} rows")


if __name__ == "__main__":
    main(sys.argv[1:])

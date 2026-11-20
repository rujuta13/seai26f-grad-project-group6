"""Deliberately naive answers to every contract (mean, random, constant).
Only here to exercise the harness scorers; NOT skill code, NOT T7."""
import random
from harness.contracts import read, goals, inputs

def families(train_csv):
    _, rows = read(train_csv); return [i % 3 for i in range(len(rows))]
def best(train_csv):
    return [0]
def guess(train_csv, row):
    h, rows = read(train_csv)
    out = {}
    for g in goals(h):
        m = sum(r[g] for r in rows) / len(rows)
        out[g] = (m, min(r[g] for r in rows), max(r[g] for r in rows))
    return out
def same(xs, ys):
    return True
def stream(train_csv, rows):
    return [i % 2 == 0 for i in range(len(rows))]
def drift(train_csv, rows):
    return None
def blame(train_csv):
    h, _ = read(train_csv); return inputs(h)[:3]
def explain(train_csv):
    h, _ = read(train_csv); return "Rows differ on " + inputs(h)[0]
def why(train_csv, row):
    return "No reason."
def choose(blind_csv, label, budget=30, goal=None):
    label(0); return 0
def fix(train_csv, row, frozen=()):
    return dict(row)
def odd(train_csv, rows):
    return [random.random() for _ in rows]
def race(train_csv, test_csv):
    return {"ours": 1.0, "random": 1.0}

"""New lines of code per step, read from git tags.

Counts non-blank, non-comment Python lines under src/ and domains/ at each tag
step0, step1, ... and prints how many each step added.

    python3 harness/loc.py
"""
import re
import subprocess

DIRS = ("src/", "domains/")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def loc(tag):
    files = [f for f in git("ls-tree", "-r", "--name-only", tag).split()
             if f.endswith(".py") and f.startswith(DIRS)]
    n = 0
    for f in files:
        for line in git("show", f"{tag}:{f}").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n


def steps():
    tags = [t for t in git("tag").split() if re.fullmatch(r"step\d+", t)]
    return sorted(tags, key=lambda t: int(t[4:]))


def main():
    tags = steps()
    sizes = [loc(t) for t in tags]
    new = [b - a for a, b in zip([0] + sizes, sizes)]
    total = sum(n for n in new[1:]) or 1
    print(f"{'step':8} {'total':>6} {'new':>6} {'%new':>6}")
    for t, s, n in zip(tags, sizes, new):
        pct = "" if t == "step0" else f"{100 * n / total:5.1f}%"
        print(f"{t:8} {s:6d} {n:6d} {pct:>6}")


if __name__ == "__main__":
    main()

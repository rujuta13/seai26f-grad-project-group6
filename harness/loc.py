"""New lines of code per step, read from git tags.

Counts non-blank, non-comment Python lines at each tag and prints how many each step added.

    python3 harness/loc.py                 # T7: tags step0, step1, ... (src/ and domains/)
    python3 harness/loc.py t2-r1-step      # an LLM trial: tags t2-r1-step1, ... (trial/)
"""
import re
import subprocess
import sys

T7_DIRS = ("src/", "domains/")
TRIAL_DIRS = ("trial/",)


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def count(text):
    """Non-blank, non-comment lines in one file's text."""
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def loc(tag, dirs=T7_DIRS):
    files = [f for f in git("ls-tree", "-r", "--name-only", tag).split()
             if f.endswith(".py") and f.startswith(dirs)]
    return sum(count(git("show", f"{tag}:{f}")) for f in files)


def steps(prefix="step"):
    tags = [t for t in git("tag").split() if re.fullmatch(re.escape(prefix) + r"\d+", t)]
    return sorted(tags, key=lambda t: int(t[len(prefix):]))


def main(prefix="step"):
    dirs = T7_DIRS if prefix == "step" else TRIAL_DIRS
    tags = steps(prefix)
    sizes = [loc(t, dirs) for t in tags]
    new = [b - a for a, b in zip([0] + sizes, sizes)]
    total = sum(n for t, n in zip(tags, new) if t != "step0") or 1
    print(f"{'step':16} {'total':>6} {'new':>6} {'%new':>6}")
    for t, s, n in zip(tags, sizes, new):
        pct = "" if t == "step0" else f"{100 * n / total:5.1f}%"
        print(f"{t:16} {s:6d} {n:6d} {pct:>6}")


if __name__ == "__main__":
    main(*sys.argv[1:2])

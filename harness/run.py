"""Run one trial of one LLM treatment (T1-T6) up a domain's ladder, one fresh call per step.

    python3 harness/run.py --treatment 2 --repeat 1 --steps 1-10 [--dry-run] [--push]

Per step N:
  1. build the prompt from fixed parts (library notes, this trial's earlier code, the step's
     contract from harness/contracts.py, and, for T2/T4/T6 only, the reuse line);
  2. one headless `claude -p` call in a brand-new empty directory, with no tools, no saved
     session, no user/project settings or MCP servers, and our own fixed system prompt;
  3. save prompt.txt, answer.json and stepN.py in the trial's own branch runs/tT-rR
     (a git worktree under .runs/), score it with harness/score.py, commit, and tag it
     tT-rR-stepN;
  4. append one row to results/results.csv.
One attempt per step: code that crashes is scored as a failure, never retried. If the call
itself fails (e.g. the Pro usage limit), the runner stops without recording the step;
rerun the same command later and it resumes at that step.
"""
import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from harness import contracts as c   # noqa: E402
from harness import data             # noqa: E402
from harness.loc import count         # noqa: E402

MODEL = "claude-haiku-5-5"   # cheapest current model; competency numbers check it is not too weak
EFFORT = "high"
RESULTS = os.path.join(ROOT, "results", "results.csv")
RUNS = os.path.join(ROOT, ".runs")
EZR = os.path.join(HERE, "libs", "ezr.py")
SCORE_TIMEOUT = 900        # seconds a step's code may take to be scored
CALL_TIMEOUT = 1800        # seconds one claude -p call may take

LIBRARY = {
    "python": "Use only the Python 3.13 standard library. No third-party packages.",
    "sklearn": ("Use the Python 3.13 standard library plus scikit-learn 1.7 and numpy 1.26. "
                "No other third-party packages."),
    "ezr": ("Use the Python 3.13 standard library plus the ezr library, importable as "
            "`import ezr` (its full source is below). No other third-party packages."),
}
TREATMENTS = {1: ("python", False), 2: ("python", True),
              3: ("sklearn", False), 4: ("sklearn", True),
              5: ("ezr", False), 6: ("ezr", True)}

SYSTEM = ("You write Python code for a data-analysis project. You cannot run code or read "
          "files: everything you know is in the message. Reply with exactly one ```python "
          "code block holding the complete requested file, and nothing else.")

REUSE = ("Reuse the code in the earlier steps wherever you can: import what you need from "
         "them instead of rewriting it, and write only what is new.")

CONVENTIONS = """\
Data conventions:
- Data files are csv files (MOOT format). Column names ending in "+" are goals to maximise,
  "-" goals to minimise; names ending in "X" are ignored; every other column is an input.
- A *row* is a dict {column name: value}; numbers are floats, other cells are strings.
- Row indices count data rows of a csv from 0 (header excluded).
- `train_csv` / `test_csv` are paths to such files, about 800 / 200 rows."""

FIELDS = ["domain", "treatment", "repeat", "step", "skill", "new_loc", "ran_ok", "score",
          "baseline", "better", "seconds", "cost_usd", "input_tokens", "output_tokens",
          "session_id", "model", "harness_sha", "when", "extra"]


# ---------------------------------------------------------------- helpers

def sh(*args, cwd=ROOT, check=True):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=check).stdout.strip()


def harness_sha():
    sha = sh("git", "rev-parse", "--short", "HEAD")
    dirty = sh("git", "status", "--porcelain", "--", "harness", "docs/requirements.md")
    return sha + ("+dirty" if dirty else "")


def steps_arg(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def domain_of(n):
    return c.DOMAIN_ORDER[(n - 1) // 10]


def sample_rows(domain, k=2):
    with open(os.path.join(data.OUT, f"{domain}-train.csv"), newline="") as f:
        return [next(f).rstrip("\r\n") for _ in range(k + 1)]


# ---------------------------------------------------------------- the prompt

def prompt(treatment, n, earlier):
    """earlier: [(k, code)] for this trial's steps before n, in order."""
    lib, reuse = TREATMENTS[treatment]
    domain = domain_of(n)
    parts = [LIBRARY[lib]]
    if lib == "ezr":
        with open(EZR) as f:
            parts.append("```python\n# ezr.py\n" + f.read() + "```")
    parts += [CONVENTIONS,
              "The data file starts like this (header, then two rows):\n```\n"
              + "\n".join(sample_rows(domain)) + "\n```"]
    if earlier:
        parts.append("Earlier steps of this project. Their files sit in the same directory as "
                     "yours and can be imported (e.g. `from step1 import ...`):")
        parts += [f"### step{k}.py\n```python\n{code}\n```" for k, code in earlier]
    parts.append(c.brief(domain, n))
    if reuse:
        parts.append(REUSE)
    parts.append(f"Reply with the complete step{n}.py in one ```python block.")
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------- one call

def call(text, auth, model):
    """One fresh headless call. Returns the parsed json reply (raises on failure)."""
    cmd = ["claude", "-p", "--model", model, "--effort", EFFORT,
           "--system-prompt", SYSTEM, "--tools", "", "--no-session-persistence",
           "--output-format", "json", "--max-budget-usd", "2"]
    env = dict(os.environ)
    if auth == "api":
        cmd.append("--bare")
    else:
        cmd += ["--setting-sources", "", "--strict-mcp-config"]
        env.pop("ANTHROPIC_API_KEY", None)
    fresh = tempfile.mkdtemp(prefix="claude-call-")      # empty: nothing to read or remember
    try:
        p = subprocess.run(cmd, input=text, cwd=fresh, env=env, capture_output=True,
                           text=True, timeout=CALL_TIMEOUT)
    finally:
        shutil.rmtree(fresh, ignore_errors=True)
    try:
        reply = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude exited {p.returncode}: {(p.stderr or p.stdout)[-500:]}")
    if p.returncode or reply.get("is_error"):
        raise RuntimeError(f"claude call failed: {str(reply.get('result'))[-500:]}")
    return reply


def extract(answer):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", answer or "", re.S)
    return max(blocks, key=len) if blocks else ""


def tokens(reply):
    u = reply.get("usage") or {}
    inp = sum(u.get(k, 0) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                          "cache_read_input_tokens"))
    return inp, u.get("output_tokens")


# ---------------------------------------------------------------- the trial's branch

def worktree(t, r):
    name = f"t{t}-r{r}"
    path, branch = os.path.join(RUNS, name), f"runs/{name}"
    if not os.path.isdir(path):
        os.makedirs(RUNS, exist_ok=True)
        exists = sh("git", "branch", "--list", branch)
        if exists:
            sh("git", "worktree", "add", path, branch)
        else:
            sh("git", "worktree", "add", "-b", branch, path, "HEAD")
    os.makedirs(os.path.join(path, "trial"), exist_ok=True)
    return path, name


def earlier_code(trial_dir, n):
    """Every earlier step of this trial (domain 2 builds on domain 1's code, and so on)."""
    out = []
    for k in range(1, n):
        f = os.path.join(trial_dir, f"step{k}.py")
        if os.path.exists(f):
            with open(f) as fh:
                out.append((k, fh.read().rstrip("\n")))
    return out


def spent():
    if not os.path.exists(RESULTS):
        return 0.0
    with open(RESULTS, newline="") as f:
        return sum(float(r["cost_usd"] or 0) for r in csv.DictReader(f))


def log(row):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------- main loop

def run(t, r, steps, auth, model, budget, dry, push):
    if dry:  # earlier code shown as placeholders: a real trial pastes its own files there
        n = steps[0]
        print(prompt(t, n, [(k, f"# (this trial's step{k}.py)") for k in range(1, n)]))
        return
    path, name = worktree(t, r)
    trial = os.path.join(path, "trial")
    sha = harness_sha()
    for n in steps:
        tag = f"{name}-step{n}"
        if sh("git", "tag", "--list", tag):
            print(f"{tag}: done already, skipping")
            continue
        if budget and spent() >= budget:
            print(f"budget ${budget} reached (spent ${spent():.2f}); stopping")
            return
        domain, skill = domain_of(n), c.step_of(domain_of(n), n)
        text = prompt(t, n, earlier_code(trial, n))
        print(f"{tag}: {skill} ... ", end="", flush=True)
        t0 = time.time()
        try:
            reply = call(text, auth, model)
        except Exception as e:  # no answer at all: stop, so a rerun resumes here
            print(f"\n  call failed, nothing recorded: {e}")
            return
        secs = time.time() - t0
        code = extract(reply.get("result"))
        here = os.path.join(trial, f"step{n}")
        os.makedirs(here, exist_ok=True)
        with open(os.path.join(here, "prompt.txt"), "w") as f:
            f.write(text)
        with open(os.path.join(here, "answer.json"), "w") as f:
            json.dump(reply, f, indent=1)
        with open(os.path.join(trial, f"step{n}.py"), "w") as f:
            f.write(code + "\n")
        p = subprocess.run([sys.executable, os.path.join(HERE, "score.py"), "--domain", domain,
                            "--step", str(n), "--code", trial], capture_output=True, text=True,
                           cwd=trial, timeout=SCORE_TIMEOUT, check=False)
        try:
            s = json.loads(p.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            s = c.result(False, error=f"scorer gave no result: {p.stderr[-500:]}")
        with open(os.path.join(here, "score.json"), "w") as f:
            json.dump(s, f, indent=1)
        inp, out = tokens(reply)
        row = dict(domain=domain, treatment=t, repeat=r, step=n, skill=skill,
                   new_loc=count(code), ran_ok=s["ok"], score=s.get("score"),
                   baseline=s.get("baseline"), better=s.get("better"), seconds=round(secs, 1),
                   cost_usd=reply.get("total_cost_usd"), input_tokens=inp, output_tokens=out,
                   session_id=reply.get("session_id"), model=model, harness_sha=sha,
                   when=time.strftime("%Y-%m-%d %H:%M:%S"),
                   extra=json.dumps(s.get("extra", {}), default=str))
        msg = (f"{tag}: {skill} (treatment {t}, repeat {r})\n\n"
               f"Code written by {model} in one fresh headless call; harness {sha}.\n"
               f"new_loc={row['new_loc']} ok={s['ok']} score={s.get('score')}")
        sh("git", "add", "trial", cwd=path)
        sh("git", "commit", "-q", "-m", msg, cwd=path)
        sh("git", "tag", tag, cwd=path)
        log(row)
        print(f"{row['new_loc']} lines, ok={s['ok']}, score={s.get('score')}, "
              f"{secs:.0f}s, ${row['cost_usd']}")
    if push:
        sh("git", "push", "-q", "origin", f"runs/{name}")
        sh("git", "push", "-q", "origin", *[f"{name}-step{n}" for n in steps])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--treatment", type=int, required=True, choices=sorted(TREATMENTS))
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--steps", type=steps_arg, default=steps_arg("1-10"))
    p.add_argument("--auth", choices=["pro", "api"], default="pro")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--budget", type=float, default=0,
                   help="stop once results.csv records this many (notional) dollars")
    p.add_argument("--dry-run", action="store_true", help="print the first step's prompt only")
    p.add_argument("--push", action="store_true", help="push the trial's branch and tags")
    a = p.parse_args()
    if not os.path.exists(os.path.join(data.OUT, "ssw-train.csv")):
        sys.exit("no splits: run `make splits` first")
    print(f"python {platform.python_version()}, model {a.model}, auth {a.auth}")
    run(a.treatment, a.repeat, a.steps, a.auth, a.model, a.budget, a.dry_run, a.push)


if __name__ == "__main__":
    main()

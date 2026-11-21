"""step16.py: justify, for the two client questions.

Q1: "Explain to the manager why we should sign this player."
Q2: "Why did you rate this player low? Give me reasons I can check on video."

Both answers come from step6. explain() gives the reasons that make a row good,
and why() gives a one-or-two-sentence verdict on one row. step6 already writes
checkable, plain-language reasons that cite columns and values from the training
file, so this module reuses those functions and changes only the wording from
"build" to "player" for the client.
"""
from __future__ import annotations

from step6 import explain as _explain
from step6 import why as _why


def explain(train_csv) -> str:
    """Reasons, in plain words a skeptical expert can check, for what makes
    players good. Every claim cites a column and value that can be looked up
    in train_csv."""
    text = _explain(train_csv)
    return text.replace("builds", "players").replace("build", "player")


def why(train_csv, row) -> str:
    """One or two sentences on why this player (goals included) is rated good
    or bad, naming the goal values and the cutoffs they were compared against."""
    text = _why(train_csv, row)
    return text.replace("builds", "players").replace("build", "player")


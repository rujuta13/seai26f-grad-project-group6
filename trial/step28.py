"""step28.py: the fewest attribute changes that bring a drifting customer back.

fix(train_csv, row, frozen=()) answers both client questions:
  - "This customer is drifting away: what's the smallest nudge that brings
    them back?" Pass the customer's attributes as row, and the attributes
    the business cannot change as frozen.
  - "Which product offer would turn a deal-only shopper into a regular one?"
    The same call. The changed attributes are the offer to make.

The search is step18.fix, reused unchanged: all single changes first, then
pairs, then triples, and the first size that yields a predicted-good customer
wins. Standard library only.
"""
from step18 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Return row's inputs with as few values changed as possible so the
    customer becomes good. Columns named in `frozen` are never changed.

    A single column name given as a string is treated as one frozen column,
    not as a sequence of characters. The row is returned unchanged if it is
    already good or if nothing can be changed.
    """
    if isinstance(frozen, str):
        frozen = (frozen,)
    return _fix(train_csv, row, frozen=tuple(frozen))


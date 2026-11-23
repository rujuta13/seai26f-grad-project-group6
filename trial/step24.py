from step4 import drift as _drift
from step4 import stream as _stream


def _to_value(cell):
    """Parse a cell the same way step4 parses the training file."""
    if cell is None or isinstance(cell, bool):
        return cell
    if isinstance(cell, (int, float)):
        return float(cell)
    try:
        return float(cell)
    except (TypeError, ValueError):
        return cell


def _normalise(rows):
    return [{k: _to_value(v) for k, v in dict(r).items()} for r in rows]


def stream(train_csv, rows) -> list[bool]:
    """Decide, one row at a time, whether each arriving customer is worth a welcome offer.

    A row is kept (True) iff no earlier reference build, either from train_csv or
    from the rows already streamed, Pareto-dominates it on the goal columns.
    Each decision uses only train_csv and the rows before it.
    """
    return _stream(train_csv, _normalise(rows))


def drift(train_csv, rows) -> int | None:
    """Return the index into `rows` at which new sign-ups stop resembling train_csv.

    Warns at the first row index at which the most recent window of rows is, as
    a group, far from the training inputs (nearest-neighbour distance above the
    training baseline more often than chance allows). Returns None if no such
    warning is raised.
    """
    return _drift(train_csv, _normalise(rows))


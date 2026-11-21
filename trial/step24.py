# step24.py
"""Flow for the client: welcome offers as sign-ups arrive, and a drift warning.

stream(): each new sign-up is judged on arrival, against the best-5% cut of
train_csv (step4.stream). Each decision uses only train_csv and that row.

drift(): step4.drift. It tests consecutive blocks of new rows against the
training nearness reference at a stricter KS level, and returns the index of
the last row in the first block that no longer matches. Every decision uses
only the rows up to that index.
"""
import step4


def stream(train_csv, rows) -> list[bool]:
  "Keep (True) each sign-up worth a welcome offer, judged as it arrives."
  out = []
  for row in rows:                      # one at a time, in arrival order
    out.append(bool(step4.stream(train_csv, [row])[0]))
  return out


def drift(train_csv, rows) -> int | None:
  "Index into ROWS of the row at which new sign-ups stop looking like train_csv, or None."
  i = step4.drift(train_csv, list(rows))
  return None if i is None else int(i)


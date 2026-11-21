# step14.py
"""Flow: keep good scouting reports as they arrive, and warn on drift.

stream(): each report is judged on arrival, against the best-5% cut of
train_csv (step4.stream). Only train_csv and that report are used.

drift(): step4.drift. It tests consecutive blocks of new rows against the
training nearness reference, at a stricter KS level, and warns at the end
of the first block that no longer matches. Each decision uses only the
rows up to that point.
"""
import step4


def stream(train_csv, rows) -> list[bool]:
  "Keep (True) each row worth following up, judged as it arrives."
  out = []
  for row in rows:                      # one at a time, in arrival order
    out.append(step4.stream(train_csv, [row])[0])
  return out


def drift(train_csv, rows) -> int | None:
  "Index of the row at which new rows stop looking like train_csv, or None."
  return step4.drift(train_csv, rows)


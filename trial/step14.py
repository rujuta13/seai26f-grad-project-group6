# step14.py
import step2
import step4

def stream(train_csv, rows) -> list[bool]:
  "For each row, in order: True if it is worth keeping (good on the goals). Goals a scouting report lacks are estimated from train_csv alone."
  tbl, score = step4._scorer(train_csv)
  out = []
  for row in rows:
    full    = step2._row(tbl, row)                    # row in full column layout
    missing = [c for c in tbl.y if full[c.at] == "?"]
    if missing:                                       # scouting reports lack goals:
      est = step2.guess(train_csv, row)               # estimate them from train only
      for c in missing: full[c.at] = est[c.txt][0]
    out.append(bool(score(full) > 0))                 # >0 means better than train average
  return out

def drift(train_csv, rows) -> int | None:
  "Index of the first row where the last W new rows stop looking like train_csv, or None if they never do."
  return step4.drift(train_csv, rows)


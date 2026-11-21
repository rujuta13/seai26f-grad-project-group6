# step26.py
import step2
import step6
import step25

def explain(train_csv: str) -> str:
  "Reasons a sales director can check for what makes customers good, plus the inputs that set responders apart."
  out = [step6.explain(train_csv)]
  tbl = step2._table(train_csv)
  try:
    resp_at = step25._at(tbl, "RESPONSE")
    comp_at = step25._at(tbl, "COMPLAIN")
  except ValueError:                       # no campaign columns: nothing more to add
    return out[0]
  good, bad = step25._groups(tbl, resp_at, comp_at)
  if good and bad:
    names = step25.blame(train_csv)[:3]
    out.append("Campaign view: the inputs that most separate customers who responded "
               "without complaining from those who did not respond or complained are "
               + ", ".join(names) + ".")
  return "\n".join(out)

def why(train_csv: str, row) -> str:
  "One or two sentences: is ROW (goals included) good or bad, and why."
  return step6.why(train_csv, row)


# step26.py
#!/usr/bin/env python3 -B
"Justify the campaign and the low-value call in plain words a sales director can check, via ezr."
import step16

def explain(train_csv) -> str:
  """Plain reasons for the next campaign: the input conditions that the best
  quarter of training customers share, with the rate of each among them against
  the overall rate, so anyone can check a line by filtering the training CSV."""
  body = step16.explain(train_csv)
  head = ("Case for the next campaign: target customers who share the traits "
          "below. These are associations in the training data, to be tested "
          "in the campaign itself.")
  return f"{head}\n{body}"

def why(train_csv, row) -> str:
  """One or two sentences on why ROW (goals included, estimated where missing)
  is high or low value, judged against the training best-quarter cutoff."""
  s = step16.why(train_csv, row)
  return (s.replace("This build is bad", "This customer is low value")
           .replace("This build is good", "This customer is high value")
           .replace("This build", "This customer"))


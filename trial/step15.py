# step15.py
import step5

def blame(train_csv) -> list[str]:
  """Input column names, most to blame for bad rows (bad on the goals) first.
  Answers both client questions: columns near the top are the ones whose values
  differ most between bad and good builds, so they explain weak penalty takers
  (Penalties+ is a goal, so its own rows are bad) and the physically weaker
  players (Strength+ is a goal). Reuses step5.blame unchanged."""
  return step5.blame(train_csv)


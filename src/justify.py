"""Justify: reasons a person can check, for a recommendation or for one rating."""
from functools import lru_cache

from certify import gap
from remember import spread

averages = lru_cache(maxsize=4)(spread)


def recommend(t, rng, n=100, small=0.02):
    """Per input column: (name, setting, reason). A column is set only if `same` calls the gap real
    and the gap is at least `small` (2%) of the worse median; otherwise leave it off."""
    out = []
    for c in t.x:
        on, off = ([r for r in t.rows if r.cells[c.at] == v] for v in "10")
        found = [(goal, m_on, m_off) for goal, (m_on, m_off, noise)
                 in zip(t.y, gap(t, rng.sample(on, n), rng.sample(off, n))) if not noise and abs(m_on - m_off) >= small * max(m_on, m_off)]
        if not found:
            out.append((c.name, "off", "no real effect worth 2%, so leave it off (simpler)"))
            continue
        goal, m_on, m_off = max(found, key=lambda f: abs(f[1] - f[2]) / max(f[1], f[2]))
        setting = "on" if m_on < m_off else "off"
        lo, hi = min(m_on, m_off), max(m_on, m_off)
        out.append((c.name, setting, f"{goal.name} is {(hi - lo) / hi:.0%} better ({lo:.4g} vs {hi:.4g})"))
    return out


def one_line(t, row, found, median, k=3):
    """Why `row` is rated good or bad: up to `k` of the columns `blame` found real on which the row
    sits on the same side as the rated-bad (or rated-good) rows. A number counts only if the row is
    at least half a standard deviation from the average."""
    bad, spec, why = t.disty(row) > median, averages(t), []
    for name, d, ok in found:
        c = next(c for c in t.x if c.name == name)
        v = row.cells[c.at]
        if not ok or v == "?":
            continue
        if c.num:
            if abs(v - spec[c.at][0]) >= spec[c.at][1] / 2 and ((v > spec[c.at][0]) == (d > 0)) == bad:
                why.append(f"{name} {v:.0f} (average {spec[c.at][0]:.0f})")
        elif ((v == "1") == (d > 0)) == bad:
            why.append(f"{name} is {'on' if v == '1' else 'off'}")
    return f"Rated {'bad' if bad else 'good'}: " + (", ".join(why[:k]) if why else "no single column stands out") + "."

"""Justify: reasons a person can check, for a recommendation or for one rating."""
from certify import gap


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


def one_line(t, row, found, median):
    """Why `row` is rated good or bad: the flags `blame` found real that point the same way."""
    bad = t.disty(row) > median
    at = {c.name: c.at for c in t.x}
    why = [f"{n} is {'on' if row.cells[at[n]] == '1' else 'off'}" for n, d, ok in found
           if ok and ((row.cells[at[n]] == "1") == (d > 0)) == bad]
    return f"Rated {'bad' if bad else 'good'}: " + (", ".join(why[:3]) if why else "no single flag stands out") + "."

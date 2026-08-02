"""
Slice construction.

Every slice built here is a hypothesis test, and every one of them is counted
into the correction family -- including the ones that lose. A tool that tests
forty slices and corrects for the six it chose to show you is not correcting
for anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Exchange-local hour bands for US futures. Stated in the report as an
# assumption, because broker exports rarely carry a timezone and a silently
# wrong session map produces confident nonsense.
SESSIONS = [
    ("Overnight (18:00-02:59)", lambda h: (h >= 18) | (h < 3)),
    ("London (03:00-08:29)", lambda h: (h >= 3) & (h < 8.5)),
    ("RTH open (08:30-10:29)", lambda h: (h >= 8.5) & (h < 10.5)),
    ("Midday (10:30-13:29)", lambda h: (h >= 10.5) & (h < 13.5)),
    ("Afternoon (13:30-17:00)", lambda h: (h >= 13.5) & (h < 18)),
]
DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build(df: pd.DataFrame, min_n: int = 30) -> list[dict]:
    """Returns [{dimension, label, mask, n, testable}] for every candidate slice."""
    n = len(df)
    out: list[dict] = []

    def add(dim: str, label: str, mask: np.ndarray):
        mask = np.asarray(mask, dtype=bool)
        k = int(mask.sum())
        if k == 0 or k == n:
            return
        out.append({"dimension": dim, "label": label, "mask": mask,
                    "n": k, "testable": k >= min_n})

    for root, g in df.groupby("root", dropna=True):
        add("Instrument", str(root), (df["root"] == root).to_numpy())

    if df["direction"].notna().any():
        for d in ("long", "short"):
            add("Direction", d.capitalize(), (df["direction"] == d).to_numpy())

    if df["setup"].notna().any():
        tags = df["setup"].fillna("").astype(str)
        for tag in [t for t in tags.unique() if t and t.lower() != "nan"][:25]:
            add("Setup tag", tag, (tags == tag).to_numpy())

    et = df["entry_time"]
    if et.notna().any():
        hour = et.dt.hour + et.dt.minute / 60.0
        for name, fn in SESSIONS:
            add("Session", name, fn(hour).fillna(False).to_numpy())
        for i, name in enumerate(DOW[:5]):
            add("Day of week", name, (et.dt.dayofweek == i).fillna(False).to_numpy())

        # first trade of the session vs everything after it (overtrading proxy)
        day = et.dt.date
        seq = df.groupby(day, dropna=False).cumcount()
        add("Trade order in day", "First trade of day", (seq == 0).to_numpy())
        add("Trade order in day", "4th trade of day or later", (seq >= 3).to_numpy())

    if df["exit_time"].notna().any() and et.notna().any():
        hold = (df["exit_time"] - et).dt.total_seconds() / 60.0
        if hold.notna().sum() >= min_n * 2:
            q = hold.quantile([0.25, 0.5, 0.75])
            add("Hold time", f"Shortest 25% (<{q[0.25]:.0f} min)", (hold <= q[0.25]).fillna(False).to_numpy())
            add("Hold time", f"Longest 25% (>{q[0.75]:.0f} min)", (hold >= q[0.75]).fillna(False).to_numpy())

    if df["qty"].notna().sum() >= min_n * 2 and df["qty"].nunique() > 2:
        q = df["qty"].quantile([0.5, 0.9])
        add("Position size", f"Largest 10% (>={q[0.9]:.0f})", (df["qty"] >= q[0.9]).fillna(False).to_numpy())
        add("Position size", f"Below median (<{q[0.5]:.0f})", (df["qty"] < q[0.5]).fillna(False).to_numpy())

    # behavioural: what follows a loss / a win
    pnl = df["net_pnl"].to_numpy(dtype=float)
    if n > min_n * 2:
        prev = np.r_[np.nan, pnl[:-1]]
        add("After a loss", "Trade immediately after a loss", prev < 0)
        add("After a loss", "Trade immediately after a win", prev > 0)

    # one two-way interaction, gated on sample size
    if et.notna().any():
        hour = et.dt.hour + et.dt.minute / 60.0
        for root, _ in df.groupby("root", dropna=True):
            rmask = (df["root"] == root).to_numpy()
            for name, fn in SESSIONS:
                m = rmask & fn(hour).fillna(False).to_numpy()
                if m.sum() >= min_n:
                    add("Instrument x Session", f"{root} - {name}", m)

    return out

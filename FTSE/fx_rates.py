"""Point-in-time EUR/GBP cross rates from CME 6E and 6B futures.

Both futures are quoted in USD per unit of the non-USD currency, so a
same-expiry synthetic EUR/GBP forward is ``6E / 6B`` (GBP per EUR).  Quotes are
selected strictly as-of the requested timestamp; later quotes are never used.
"""
from __future__ import annotations

import csv
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


DEFAULT_TICK_ROOT = Path(r"D:\tick_data\100_ms")
DEFAULT_CACHE = Path(__file__).resolve().parent / "eurgbp_6e6b_0800_1000.parquet"
LONDON = ZoneInfo("Europe/London")
MONTH_CODE = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}


def _approx_last_trade(expiry: str) -> pd.Timestamp | None:
    """Approximate CME FX last-trade date: two business days before third Wed."""
    if len(expiry) != 3 or expiry[0] not in MONTH_CODE or not expiry[1:].isdigit():
        return None
    year = 2000 + int(expiry[1:])
    month = MONTH_CODE[expiry[0]]
    first = pd.Timestamp(year=year, month=month, day=1)
    first_wed = first + pd.Timedelta(days=(2 - first.dayofweek) % 7)
    third_wed = first_wed + pd.Timedelta(days=14)
    return third_wed - pd.offsets.BDay(2)


def _common_front_files(root: Path, day: pd.Timestamp) -> tuple[Path, Path, str]:
    """Return the most liquid common 6E/6B expiry available on one date."""
    stamp = pd.Timestamp(day).strftime("%Y%m%d")
    by_asset: dict[str, dict[str, Path]] = {}
    for asset in ("6E", "6B"):
        folder = root / asset / stamp
        files = {
            p.stem.removeprefix(f"FUT_{asset}"): p
            for p in folder.glob(f"FUT_{asset}*.csv")
            if p.stat().st_size > 0
        } if folder.exists() else {}
        by_asset[asset] = files

    common = set(by_asset["6E"]).intersection(by_asset["6B"])
    if not common:
        raise FileNotFoundError(f"no common 6E/6B contract for {stamp}")

    # Choose the nearest non-expired common maturity. File size alone is unsafe:
    # serial back months can be heavily populated in one currency but are not the
    # front contract. Freshness checks below still reject an illiquid selected leg.
    dated = [(x, _approx_last_trade(x)) for x in common]
    eligible = [(x, last) for x, last in dated if last is not None and last >= pd.Timestamp(day)]
    if eligible:
        expiry = min(eligible, key=lambda x: x[1])[0]
    else:
        expiry = max(
            common,
            key=lambda x: min(
                by_asset["6E"][x].stat().st_size,
                by_asset["6B"][x].stat().st_size,
            ),
        )
    return by_asset["6E"][expiry], by_asset["6B"][expiry], expiry


def _asof_mids(
    path: Path,
    targets: dict[str, float],
    max_stale_seconds: float,
) -> dict[str, tuple[float, float]]:
    """Return label -> (mid, quote Unix timestamp) using quotes at/before target."""
    ordered = sorted(targets.items(), key=lambda x: x[1])
    out: dict[str, tuple[float, float]] = {}
    last: tuple[float, float] | None = None
    target_i = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].startswith("#") or len(row) < 7:
                continue
            try:
                ts = float(row[1])
            except (TypeError, ValueError):
                continue

            while target_i < len(ordered) and ts > ordered[target_i][1]:
                label, target = ordered[target_i]
                if last is not None and 0 <= target - last[1] <= max_stale_seconds:
                    out[label] = last
                target_i += 1
            if target_i >= len(ordered):
                break

            if row[2] == "Q" and row[4] and row[5]:
                try:
                    bid, ask = float(row[4]), float(row[5])
                except ValueError:
                    continue
                if bid > 0 and ask >= bid:
                    last = ((bid + ask) / 2.0, ts)

    while target_i < len(ordered):
        label, target = ordered[target_i]
        if last is not None and 0 <= target - last[1] <= max_stale_seconds:
            out[label] = last
        target_i += 1
    return out


def _epoch(day: pd.Timestamp, clock: time) -> float:
    local = pd.Timestamp.combine(pd.Timestamp(day).date(), clock).tz_localize(LONDON)
    return float(local.tz_convert("UTC").timestamp())


def rate_for_day(
    day: pd.Timestamp,
    *,
    tick_root: Path = DEFAULT_TICK_ROOT,
    entry_time: time = time(8, 0),
    exit_time: time = time(10, 0),
    max_stale_seconds: float = 10.0,
) -> dict:
    """Build one event's entry/exit EURGBP rates from same-expiry as-of quotes."""
    day = pd.Timestamp(day).normalize()
    six_e, six_b, expiry = _common_front_files(Path(tick_root), day)
    targets = {"entry": _epoch(day, entry_time), "exit": _epoch(day, exit_time)}
    e = _asof_mids(six_e, targets, max_stale_seconds)
    b = _asof_mids(six_b, targets, max_stale_seconds)
    missing = [label for label in targets if label not in e or label not in b]
    if missing:
        raise ValueError(
            f"missing fresh 6E/6B quote for {day.date()} at {', '.join(missing)}"
        )

    row = {
        "date": day,
        "fx_contract": expiry,
        "fx_entry_time": entry_time.strftime("%H:%M:%S"),
        "fx_exit_time": exit_time.strftime("%H:%M:%S"),
    }
    for label, target in targets.items():
        six_e_mid, six_e_ts = e[label]
        six_b_mid, six_b_ts = b[label]
        row[f"six_e_{label}"] = six_e_mid
        row[f"six_b_{label}"] = six_b_mid
        row[f"eurgbp_{label}"] = six_e_mid / six_b_mid
        row[f"fx_{label}_max_age_s"] = max(target - six_e_ts, target - six_b_ts)
        row[f"fx_{label}_quote_skew_s"] = abs(six_e_ts - six_b_ts)
    return row


def eurgbp_history(
    dates,
    *,
    cache: Path = DEFAULT_CACHE,
    tick_root: Path = DEFAULT_TICK_ROOT,
    entry_time: time = time(8, 0),
    exit_time: time = time(10, 0),
    max_stale_seconds: float = 10.0,
    force: bool = False,
) -> pd.DataFrame:
    """Load/build cached point-in-time EURGBP rates for the requested dates."""
    requested = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
    cache = Path(cache)
    if cache.exists() and not force:
        existing = pd.read_parquet(cache)
        existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
    else:
        existing = pd.DataFrame(columns=["date"])

    if len(existing) and {"eurgbp_entry", "eurgbp_exit"}.issubset(existing.columns):
        valid_existing = existing["eurgbp_entry"].notna() & existing["eurgbp_exit"].notna()
        have = set(existing.loc[valid_existing, "date"])
    else:
        have = set()
    rows = []
    for day in requested:
        if day in have:
            continue
        try:
            rows.append(rate_for_day(
                day,
                tick_root=Path(tick_root),
                entry_time=entry_time,
                exit_time=exit_time,
                max_stale_seconds=max_stale_seconds,
            ))
        except (FileNotFoundError, ValueError) as exc:
            rows.append({"date": day, "fx_error": str(exc)})

    if rows:
        existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        existing = existing.sort_values("date").drop_duplicates("date", keep="last")
        cache.parent.mkdir(parents=True, exist_ok=True)
        existing.to_parquet(cache, index=False)

    out = existing.set_index("date").reindex(requested)
    for col in ("eurgbp_entry", "eurgbp_exit"):
        if col not in out:
            out[col] = np.nan
    return out

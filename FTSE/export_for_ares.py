r"""Build the complete FTSE ex-dividend event file consumed by Ares.

Run this with the research environment (pandas/openpyxl/pyarrow are required):

    C:\repo\ex_div_dates_research\.venv\Scripts\python.exe C:\repo\ex_div_dates_research\FTSE\export_for_ares.py

The export is deliberately separated from the Ares run because Ares uses its own
Python environment. This step reconstructs drops with dated iShares holdings,
labels the long/short rule, estimates a strictly lagged 60-session DAX beta, and
resolves the actual FT/FDAX/Mini-DAX/Micro-DAX archive contracts for every event.

The beta reproduces the two-window notebook: FTSE 08:00--10:00 London is paired
with DAX 09:00--11:00 Berlin. The event date is never used in its own beta.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ftse_weights as fw
from fx_rates import DEFAULT_CACHE as FX_CACHE
from fx_rates import DEFAULT_TICK_ROOT as MARKET_TICK_ROOT
from fx_rates import eurgbp_history

ROOT = Path(__file__).resolve().parent
DIV_FILE = ROOT / "FTSE_div_dates_updatd.xlsx"
OUT = ROOT / "ares_inputs.json"
AUDIT_OUT = ROOT / "ares_events_audit.csv"
FTSE_PANEL = ROOT / "panel_with_mid.parquet"
DAX_PANEL = ROOT.parent / "DAX" / "dax_panel_with_mid.parquet"
TICK_ROOT = Path(r"D:\tick_data\100_ms")

# Signal definition -- kept in the payload so every Ares run is auditable.
WEIGHTS_SOURCE = "updated"
ORIGINAL_SHEET = "final table"
ANALYSIS_START = pd.Timestamp("2022-03-01")
ANALYSIS_END = None
SMALL_MAX_DROP_BPS = 5.0
LARGE_MIN_DROP_BPS = 5.0
LONG_LOTS = 20
SHORT_LOTS = 20

# Hedge definition. Pre-analysis Thursdays are excluded because the old dividend
# calendar is not complete enough to identify every event there.
HEDGE_WINDOW = 60
HEDGE_MIN_OBS = 40
# The FTSE cache is indexed from 01:00 London, not midnight.
FTSE_OPEN_M = 7 * 60       # 08:00 London
FTSE_EXIT_M = 9 * 60       # 10:00 London
DAX_OPEN_M = 9 * 60
DAX_EXIT_M = 11 * 60

# Dividend reconstruction settings shared with the research notebook.
FIX_FOREIGN_CCY_DIVIDENDS = True
EXCLUDE_DIVIDENDS = [("AV", "2022-05-16")]
SHARE_EVENTS = {"AV": "2022-05-17", "GSK": "2022-07-18"}
GBPUSD = 1.35
GBPEUR = 1.17

ASSETS = ("FT", "DAX", "FDXM", "FDXS")


def expected_drops() -> tuple[pd.Series, fw.DropResult]:
    """Ex-date -> expected index drop in bps, using the notebook's weighting code."""
    result = fw.expected_drops(
        WEIGHTS_SOURCE,
        start=ANALYSIS_START,
        end=ANALYSIS_END,
        div_file=DIV_FILE,
        original_sheet=ORIGINAL_SHEET,
        gbpusd=GBPUSD,
        gbpeur=GBPEUR,
        fix_foreign_ccy=FIX_FOREIGN_CCY_DIVIDENDS,
        exclude=EXCLUDE_DIVIDENDS,
        share_events=SHARE_EVENTS,
    )
    for note in result.notes:
        print("  " + note)
    if len(result.specials):
        print("  possible special dividends retained in the source:")
        print("    " + result.specials[
            ["ex_date", "code", "div_pence", "contrib_bps"]
        ].round({"div_pence": 2, "contrib_bps": 2}).to_string(index=False).replace(
            "\n", "\n    "
        ))
    return result.exdiv["drop_bps"].sort_index(), result


def minute_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.pivot_table(index="date", columns="m", values="mid").sort_index()


def lagged_beta_60(drops: pd.Series) -> pd.Series:
    """Lagged FTSE~DAX OLS slope, trained only on prior non-event observations."""
    ftse = minute_panel(FTSE_PANEL)
    dax = minute_panel(DAX_PANEL)
    common = ftse.index.intersection(dax.index).sort_values()
    frame = pd.DataFrame(index=common)
    frame["ftse_bps"] = (
        ftse.loc[common, FTSE_EXIT_M] / ftse.loc[common, FTSE_OPEN_M] - 1
    ) * 1e4
    frame["dax_bps"] = (
        dax.loc[common, DAX_EXIT_M] / dax.loc[common, DAX_OPEN_M] - 1
    ) * 1e4
    frame["drop"] = drops.reindex(common).fillna(0.0)
    frame["training_normal"] = np.where(
        frame.index < ANALYSIS_START,
        frame.index.dayofweek != 3,
        frame["drop"].eq(0.0),
    )

    eligible = (
        frame["training_normal"]
        & frame[["ftse_bps", "dax_bps"]].notna().all(axis=1)
    )
    beta = pd.Series(np.nan, index=frame.index, name="beta_60", dtype=float)
    for i, day in enumerate(frame.index):
        hist = frame.iloc[:i].loc[
            eligible.iloc[:i], ["ftse_bps", "dax_bps"]
        ].tail(HEDGE_WINDOW)
        if len(hist) < HEDGE_MIN_OBS or hist["dax_bps"].var(ddof=1) <= 0:
            continue
        beta.loc[day] = (
            hist["ftse_bps"].cov(hist["dax_bps"])
            / hist["dax_bps"].var(ddof=1)
        )
    return beta


def largest_archive_contract(asset: str, day: pd.Timestamp) -> str | None:
    """Most actively recorded contract in an archive directory for one date."""
    folder = TICK_ROOT / asset / day.strftime("%Y%m%d")
    files = list(folder.glob("FUT_*.csv")) if folder.exists() else []
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_size).stem.removeprefix("FUT_")


def contracts_for_day(day: pd.Timestamp) -> dict[str, str] | None:
    """Resolve one common DAX expiry plus the active FT contract from the archive."""
    ft = largest_archive_contract("FT", day)
    dax = largest_archive_contract("DAX", day)
    if ft is None or dax is None or not dax.startswith("DAX"):
        return None
    expiry = dax.removeprefix("DAX")
    contracts = {
        "FT": ft,
        "DAX": f"DAX{expiry}",
        "FDXM": f"FDXM{expiry}",
        "FDXS": f"FDXS{expiry}",
    }
    for asset, contract in contracts.items():
        path = (TICK_ROOT / asset / day.strftime("%Y%m%d")
                / f"FUT_{contract}.csv")
        if not path.exists() or path.stat().st_size == 0:
            return None
    return contracts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-lots", type=int, default=LONG_LOTS)
    parser.add_argument("--short-lots", type=int, default=SHORT_LOTS)
    args = parser.parse_args()
    if args.long_lots <= 0 or args.short_lots <= 0:
        parser.error("--long-lots and --short-lots must be positive integers")
    return args


def main() -> None:
    args = parse_args()
    long_lots = args.long_lots
    short_lots = args.short_lots
    print(f"WEIGHTS_SOURCE = {WEIGHTS_SOURCE!r}")
    drops, result = expected_drops()
    beta_sessions = lagged_beta_60(drops)
    fx = eurgbp_history(
        drops.index,
        cache=FX_CACHE,
        tick_root=MARKET_TICK_ROOT,
        entry_time=pd.Timestamp("08:00").time(),
        exit_time=pd.Timestamp("10:00").time(),
    )
    # If an event session is absent from either minute-panel cache, carry the most
    # recently calculated beta forward.  That ratio was fully known before the event
    # and lets Ares use its available raw tick files without introducing look-ahead.
    beta = (beta_sessions.reindex(beta_sessions.index.union(drops.index))
            .sort_index().ffill().reindex(drops.index))

    rows: list[dict] = []
    events: dict[str, dict] = {}
    front_months = {asset: {} for asset in ASSETS}
    omitted: list[tuple[str, str]] = []

    for day, drop in drops.items():
        if not np.isfinite(drop) or drop <= 0:
            continue
        direction = "long" if drop < SMALL_MAX_DROP_BPS else "short"
        lots = long_lots if direction == "long" else short_lots
        b = beta.get(day, np.nan)
        if not np.isfinite(b) or b <= 0:
            omitted.append((day.date().isoformat(), "no lagged beta"))
            continue
        fx_row = fx.loc[day]
        eurgbp_entry = fx_row.get("eurgbp_entry", np.nan)
        eurgbp_exit = fx_row.get("eurgbp_exit", np.nan)
        if not np.isfinite(eurgbp_entry) or not np.isfinite(eurgbp_exit):
            reason = str(fx_row.get("fx_error", "missing 6E/6B cross rate"))
            omitted.append((day.date().isoformat(), reason))
            continue
        contracts = contracts_for_day(day)
        if contracts is None:
            omitted.append((day.date().isoformat(), "missing FT/DAX archive contract"))
            continue

        key = day.date().isoformat()
        events[key] = {
            "drop_bps": round(float(drop), 8),
            "direction": direction,
            "ftse_lots": int(lots),
            "beta_60": round(float(b), 10),
            "eurgbp_entry": round(float(eurgbp_entry), 10),
            "eurgbp_exit": round(float(eurgbp_exit), 10),
            "fx_contract": str(fx_row["fx_contract"]),
        }
        for asset, contract in contracts.items():
            front_months[asset][key] = contract
        rows.append({
            "date": key,
            **events[key],
            **{f"contract_{asset.lower()}": contract
               for asset, contract in contracts.items()},
        })

    if not events:
        raise SystemExit("no executable Ares events after beta/archive validation")

    payload = {
        "schema_version": 3,
        "assets": list(ASSETS),
        "events": events,
        "front_months": front_months,
        "meta": {
            "weights_source": WEIGHTS_SOURCE,
            "div_file": DIV_FILE.name,
            "dividend_sheet": "all ex divs",
            "holdings_file": "isf_holdings_history.xlsx",
            "analysis_start": ANALYSIS_START.date().isoformat(),
            "analysis_end": (pd.Timestamp(ANALYSIS_END).date().isoformat()
                             if ANALYSIS_END is not None else None),
            "small_max_drop_bps": SMALL_MAX_DROP_BPS,
            "large_min_drop_bps": LARGE_MIN_DROP_BPS,
            "long_lots": long_lots,
            "short_lots": short_lots,
            "hedge": "lagged OLS FTSE~DAX beta",
            "hedge_window": HEDGE_WINDOW,
            "hedge_min_obs": HEDGE_MIN_OBS,
            "return_window": "FTSE 08:00-10:00 London / DAX 09:00-11:00 Berlin",
            "gbpusd": GBPUSD,
            "gbpeur": GBPEUR,
            "eurgbp_source": "same-expiry CME 6E/6B as-of midpoint cross",
            "eurgbp_entry_time": "08:00 Europe/London",
            "eurgbp_exit_time": "10:00 Europe/London",
            "fix_foreign_ccy_dividends": FIX_FOREIGN_CCY_DIVIDENDS,
            "n_dividends_ccy_converted": int(len(result.converted)),
            "excluded_dividends": [[c, str(pd.Timestamp(d).date())]
                                     for c, d in EXCLUDE_DIVIDENDS],
            "share_events": SHARE_EVENTS,
            "n_events": len(events),
            "n_long": sum(e["direction"] == "long" for e in events.values()),
            "n_short": sum(e["direction"] == "short" for e in events.values()),
            "omitted": [{"date": d, "reason": reason} for d, reason in omitted],
        },
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(AUDIT_OUT, index=False)

    first, last = next(iter(events)), next(reversed(events))
    print(f"wrote {OUT.name} and {AUDIT_OUT.name}")
    print(f"  events: {len(events)} ({payload['meta']['n_long']} long, "
          f"{payload['meta']['n_short']} short), {first} .. {last}")
    print(f"  beta: median {np.median([e['beta_60'] for e in events.values()]):.3f}")
    print(f"  omitted: {len(omitted)}")
    for day, reason in omitted:
        print(f"    {day}: {reason}")


if __name__ == "__main__":
    main()

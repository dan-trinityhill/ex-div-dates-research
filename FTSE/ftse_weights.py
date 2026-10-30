r"""Expected FTSE 100 index drop on each ex-dividend date, two ways.

Used by ftse_exdiv_two_windows.ipynb and export_for_ares.py so both get identical
numbers. Pick the method with WEIGHTS_SOURCE:

  "original"  The workbook's own constituent list, index weights and Sheet4 prices
              (today's), applied to every historical date. This reproduces the numbers
              the notebook produced before.

  "updated"   The index as it actually stood on each ex-date, from the iShares Core
              FTSE 100 (ISF) holdings in isf_holdings_history.xlsx. For an ex-date T the
              latest report dated before T is used. Dividends come from the "all ex divs"
              sheet, so companies that have since left the index are included. Starts at
              the first report, 28 Feb 2022.

How "updated" turns holdings into an index drop
------------------------------------------------
ISF fully replicates the index, so the number of shares it holds in each company is
proportional to that company's free-float shares in the index. A dividend D (GBP per
share) on N fund shares removes D x N from the fund's equity value, and the index drops
by the same fraction:

    drop_bps(T) = 1e4 x sum_i  D_i x N_i  /  MV(T-1)

    MV(T-1) = MV(report) x level(T-1 16:35) / level(report day 16:35)

MV(report) is the fund's total equity value in the report. The FTSE 100 future's 16:35
mid rolls it forward to the evening before the ex-date, which is when the index is
valued before the 08:00 drop. This is the same thing as weight_i x D_i / P_i, but it
uses the index level on the day instead of share prices that are months (or, in the
original method, years) out of date. The future's basis and quarterly roll move the
ratio by well under 1%, so the drop estimate by well under 1% of itself.

Dividend currency
-----------------
The workbook mostly quotes dividends in pence, but some are in USD or EUR (e.g. Glencore,
Antofagasta, Coca-Cola HBC every time, and recent unconverted HSBC / BP / Rio rows). With
FIX_FOREIGN_CCY_DIVIDENDS=True a dividend whose single-payment yield is below
MIN_PENCE_YIELD (0.04%, impossible for a real pence dividend on these stocks) is treated as
USD (or EUR, from isf_company_codes.csv) and converted at GBPUSD / GBPEUR. Every converted
row is listed. Set it False to reproduce the old numbers exactly.

Share-count events
------------------
Between two reports the holding is fixed at the earlier report's count. A share
consolidation or big buyback inside the gap (Aviva May 2022, GSK July 2022) then leaves the
old count against the new per-share dividend. `share_events={"AV": "2022-05-17"}` makes
dividends from that date use the next report's holding, scaled back by the fund's overall
growth between the two reports.

Special dividends
-----------------
The workbook does not say which dividends are specials or capital returns (Aviva's 101.69p
in May 2022 and Mondi's 136p in Jan 2024 look like one-off returns of capital). Any single payment
yielding more than 7% is returned in `specials` so it can be checked, and `exclude` takes a
list of (code, ex-date) pairs to leave out.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LONDON = ZoneInfo("Europe/London")
EPOCH = pd.Timestamp("1970-01-01")

DEFAULTS = dict(
    div_file=ROOT / "FTSE_div_dates_updatd.xlsx",
    holdings_file=ROOT / "isf_holdings_history.xlsx",
    code_map_file=ROOT / "isf_company_codes.csv",
    panel_cache=ROOT / "panel_with_mid.parquet",
    tick_dir=Path(r"D:\tick_data\100_ms\FT"),
    original_sheet="Sheet1",
    updated_sheet="all ex divs",
    gbpusd=1.35,
    gbpeur=1.17,
    fix_foreign_ccy=True,
    min_pence_yield=0.0004,
    level_time="16:35",
    exclude=(),               # [("AV", "2022-05-16"), ...] dividends to leave out, e.g. specials
    special_yield=0.07,       # single-payment yield above this is listed as a possible special
    share_events={},          # {"AV": "2022-05-17"}: from this date use the NEXT report's share count
)
FIRST_REPORT = pd.Timestamp("2022-02-28")


@dataclass
class DropResult:
    method: str
    contrib: pd.DataFrame            # one row per dividend used
    exdiv: pd.DataFrame              # index ex_date -> n_stocks, drop_bps
    converted: pd.DataFrame = field(default_factory=pd.DataFrame)   # currency fixes
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)    # rows not used, with reason
    specials: pd.DataFrame = field(default_factory=pd.DataFrame)    # unusually large single payments
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)    # per report: weight with no dividend data
    notes: list = field(default_factory=list)


# --------------------------------------------------------------------- helpers
def _convert_foreign(df, price_pence_col, ccy_col, cfg, notes, units_col="dividend_units"):
    """Convert dividends that are not in pence. Returns (df, converted).

    Two cases: companies whose dividends the workbook always quotes in USD/EUR (the
    `dividend_units` column of isf_company_codes.csv), and odd rows for other companies
    whose single-payment yield is impossibly small for pence (recent unconverted rows)."""
    df = df.copy()
    df["div_pence"] = df["Dividend"].astype(float)
    df["ccy_converted"] = ""
    if not cfg["fix_foreign_ccy"]:
        return df, df.iloc[0:0]
    units = df.get(units_col, pd.Series("", index=df.index)).fillna("")
    always = units.isin(["USD", "EUR"])
    y = df["div_pence"] / df[price_pence_col]
    odd = ~always & (y < cfg["min_pence_yield"]) & df["div_pence"].gt(0) & y.notna()
    ccy = df.get(ccy_col, pd.Series("USD", index=df.index)).fillna("USD").replace("", "USD")
    ccy = ccy.where(~always, units)
    bad = always | odd
    fx = np.where(ccy.eq("EUR"), cfg["gbpeur"], cfg["gbpusd"])
    df.loc[bad, "div_pence"] = df.loc[bad, "Dividend"] * 100.0 / fx[bad.to_numpy()]
    df.loc[bad, "ccy_converted"] = ccy[bad]
    conv = df.loc[bad, ["ex_date", "code", "Company", "Dividend", "ccy_converted", "div_pence"]]
    notes.append(f"{int(bad.sum())} dividend rows were not in pence and were converted "
                 f"({int(always.sum())} from always-USD/EUR companies, {int(odd.sum())} odd rows; "
                 f"GBPUSD {cfg['gbpusd']}, GBPEUR {cfg['gbpeur']})")
    return df, conv


def _apply_exclusions(df, cfg):
    """Drop user-excluded (code, ex_date) dividends. Returns (kept, dropped)."""
    keys = {(str(c).strip(), pd.Timestamp(d).normalize()) for c, d in cfg["exclude"]}
    if not keys:
        return df, df.iloc[0:0]
    hit = pd.Series([(c, d.normalize()) in keys for c, d in zip(df["code"], df["ex_date"])],
                    index=df.index)
    return df[~hit].copy(), df[hit].assign(reason="excluded by EXCLUDE_DIVIDENDS")


def _possible_specials(df, cfg):
    cols = [c for c in ("ex_date", "code", "Company", "Dividend", "div_pence", "div_yield",
                        "contrib_bps") if c in df.columns]
    return df.loc[df["div_yield"] > cfg["special_yield"], cols].sort_values("ex_date")


def _uk_epoch(day, when):
    local = dt.datetime.combine(pd.Timestamp(day).date(), when, tzinfo=LONDON)
    return (pd.Timestamp(local).tz_convert("UTC").tz_localize(None) - EPOCH) / pd.Timedelta(seconds=1)


def _scan_level(day, tick_dir, when):
    """Last FT front-month mid at or before `when` London, read straight from ticks."""
    d = Path(tick_dir) / pd.Timestamp(day).strftime("%Y%m%d")
    files = sorted(d.glob("FUT_FT*.csv"), key=lambda f: f.stat().st_size) if d.exists() else []
    if not files:
        return np.nan
    t = pd.read_csv(files[-1], skiprows=2, header=None, usecols=[1, 2, 4, 5],
                    names=["UnixTs", "MsgType", "Bid", "Ask"], low_memory=False)
    q = t[(t["MsgType"] == "Q") & t["Bid"].notna() & t["Ask"].notna()
          & (t["UnixTs"] <= _uk_epoch(day, when))]
    return float((q["Bid"].iloc[-1] + q["Ask"].iloc[-1]) / 2) if len(q) else np.nan


class IndexLevels:
    """FTSE 100 future mid at a fixed London time, from the notebook's panel cache,
    falling back to the tick files for any day the cache does not hold."""

    def __init__(self, cfg):
        self.cfg = cfg
        h, m = (int(x) for x in cfg["level_time"].split(":"))
        self.when = dt.time(h, m)
        self.m = h * 60 + m - 60                     # panel minute index (01:00 = 0)
        self.levels = pd.Series(dtype=float)
        p = Path(cfg["panel_cache"])
        if p.exists():
            panel = pd.read_parquet(p, columns=["date", "m", "mid"])
            panel = panel[panel["m"] == self.m]
            self.levels = (panel.assign(date=pd.to_datetime(panel["date"]))
                           .set_index("date")["mid"].sort_index())
        self.days = pd.DatetimeIndex(self.levels.index)

    def level(self, day):
        day = pd.Timestamp(day).normalize()
        v = self.levels.get(day, np.nan)
        if not np.isfinite(v):
            v = _scan_level(day, self.cfg["tick_dir"], self.when)
            self.levels.loc[day] = v
        return v

    def trading_day_on_or_before(self, day):
        day = pd.Timestamp(day).normalize()
        known = self.days[self.days <= day]
        if len(known):
            return known[-1]
        return day if day.weekday() < 5 else day - pd.offsets.BDay(1)

    def trading_day_before(self, day):
        day = pd.Timestamp(day).normalize()
        known = self.days[self.days < day]
        if len(known) and (day - known[-1]).days <= 5:
            return known[-1]
        return day - pd.offsets.BDay(1)


def _summarise(exdiv):
    return (f"ex-dates: {len(exdiv)} ({exdiv.index.min().date()} .. {exdiv.index.max().date()}); "
            f"expected drop mean {exdiv['drop_bps'].mean():.2f}, median "
            f"{exdiv['drop_bps'].median():.2f}, max {exdiv['drop_bps'].max():.2f} bps")


# ------------------------------------------------------------------- original
def original_drops(start=None, end=None, **kw) -> DropResult:
    """The workbook's current weights and Sheet4 prices -- the notebook's original logic."""
    cfg = {**DEFAULTS, **kw}
    notes = []
    sheet = cfg["original_sheet"]
    div = pd.read_excel(cfg["div_file"], sheet_name=sheet)
    s4 = pd.read_excel(cfg["div_file"], sheet_name="Sheet4").rename(
        columns={"Market cap (m)": "mcap_sheet4"})
    if "FTSE?" in div.columns:
        div = div[div["FTSE?"].astype(str).str.strip().str.lower().eq("yes")].copy()
    div = div.rename(columns={"Column2": "code", "Column3": "market cap",
                              "Column4": "share of ftse 100"})
    div = div.drop(columns=[c for c in ("Column1",) if c in div.columns])
    div = div.rename(columns={"share of ftse 100": "weight", "market cap": "mcap_workbook"})
    missing = [c for c in ("Company", "Ex-Dividend Date", "Dividend", "code",
                           "weight", "mcap_workbook") if c not in div.columns]
    if missing:
        raise KeyError(f"'{sheet}' is missing expected columns: {missing}")
    for c in ("Dividend", "weight", "mcap_workbook"):
        div[c] = pd.to_numeric(div[c], errors="coerce")
    div["ex_date"] = pd.to_datetime(div["Ex-Dividend Date"], errors="coerce")
    invalid = div[["ex_date", "Dividend", "weight", "mcap_workbook"]].isna().any(axis=1)
    notes.append(f"source sheet: {sheet}; rows dropped for missing date/dividend/weight/mcap: "
                 f"{int(invalid.sum())}")
    div = div.loc[~invalid].copy()
    lo = pd.Timestamp(start) if start else div["ex_date"].min()
    hi = pd.Timestamp(end) if end else div["ex_date"].max()
    div = div[div["ex_date"].between(lo, hi)].copy()

    s4["price_pence"] = np.where(s4["Currency"].eq("USD"),
                                 s4["Price"] / cfg["gbpusd"] * 100.0, s4["Price"])
    mg = div.merge(s4[["Code", "Currency", "mcap_sheet4", "price_pence"]],
                   left_on="code", right_on="Code", how="left")
    mg["mcap_gap"] = (mg["mcap_workbook"] - mg["mcap_sheet4"]).abs()
    mg = (mg.sort_values("mcap_gap")
            .drop_duplicates(subset=["Company", "ex_date", "Dividend"], keep="first")
            .sort_values(["ex_date", "Company"]).reset_index(drop=True))
    codes = pd.read_csv(cfg["code_map_file"], keep_default_na=False) if Path(cfg["code_map_file"]).exists() else None
    ccy_map = dict(zip(codes["code"], codes["dividend_ccy_if_not_pence"])) if codes is not None else {}
    mg["ccy_if_not_pence"] = mg["code"].map(ccy_map)
    unit_map = dict(zip(codes["code"], codes["dividend_units"])) if codes is not None and "dividend_units" in codes else {}
    mg["dividend_units"] = mg["code"].map(unit_map).fillna("")
    mg, conv = _convert_foreign(mg, "price_pence", "ccy_if_not_pence", cfg, notes)
    mg["div_yield"] = mg["div_pence"] / mg["price_pence"]
    mg["contrib_bps"] = mg["weight"] * mg["div_yield"] * 1e4
    mg, dropped = _apply_exclusions(mg, cfg)
    uw = mg.drop_duplicates("code")
    notes.append(f"distinct constituents carrying a dividend: {len(uw)}; weights sum to "
                 f"{uw['weight'].sum():.4f} (today's constituents, today's weights and prices)")
    exdiv = (mg.groupby("ex_date").agg(n_stocks=("code", "size"), drop_bps=("contrib_bps", "sum"))
               .sort_index())
    notes.append(_summarise(exdiv))
    return DropResult("original", mg, exdiv, converted=conv, excluded=dropped,
                      specials=_possible_specials(mg, cfg), notes=notes)


# -------------------------------------------------------------------- updated
def load_holdings(holdings_file):
    h = pd.read_excel(holdings_file, sheet_name="Holdings")
    h["as_of"] = pd.to_datetime(h["as_of"])
    h = h[h["in_index"].astype(bool)]
    # Evraz Feb 2022 has two lines for the same shares: count the shares once, sum the value
    snap = (h.groupby(["as_of", "company"], as_index=False)
              .agg(holding_shares=("holding_shares", "max"),
                   fair_value_gbp000=("fair_value_gbp000", "sum")))
    snap["mv_gbp000"] = snap.groupby("as_of")["fair_value_gbp000"].transform("sum")
    snap["weight"] = snap["fair_value_gbp000"] / snap["mv_gbp000"]
    snap["price_pence"] = snap["fair_value_gbp000"] * 1000.0 / snap["holding_shares"] * 100.0
    return snap


def updated_drops(start=None, end=None, **kw) -> DropResult:
    """Constituents, share counts and index value as at each ex-date (from the ISF reports)."""
    cfg = {**DEFAULTS, **kw}
    notes = []
    snap = load_holdings(cfg["holdings_file"])
    codes = pd.read_csv(cfg["code_map_file"], keep_default_na=False)
    unmapped = sorted(set(snap["company"]) - set(codes["company"]))
    if unmapped:
        raise KeyError(f"companies in {Path(cfg['holdings_file']).name} with no code in "
                       f"{Path(cfg['code_map_file']).name}: {unmapped}")
    if "dividend_units" not in codes:
        codes["dividend_units"] = ""
    snap = snap.merge(codes[["company", "code", "dividend_ccy_if_not_pence", "dividend_units"]],
                      on="company")
    reports = sorted(snap["as_of"].unique())
    first_report = pd.Timestamp(reports[0])

    lv = IndexLevels(cfg)
    rep = pd.DataFrame({"as_of": reports})
    rep["valuation_day"] = rep["as_of"].map(lv.trading_day_on_or_before)
    rep["level_report"] = rep["valuation_day"].map(lv.level)
    rep["mv_gbp000"] = rep["as_of"].map(snap.groupby("as_of")["mv_gbp000"].first())
    if rep["level_report"].isna().any():
        raise ValueError(f"no FTSE future level for report day(s) "
                         f"{rep.loc[rep['level_report'].isna(), 'valuation_day'].dt.date.tolist()}")

    a = pd.read_excel(cfg["div_file"], sheet_name=cfg["updated_sheet"])
    a = a.dropna(subset=["Company"]).rename(columns={"Column2": "code"})
    a["ex_date"] = pd.to_datetime(a["Formatted date"], errors="coerce")
    a["ex_date"] = a["ex_date"].fillna(pd.to_datetime(a["Ex-Dividend Date"], errors="coerce"))
    a["Dividend"] = pd.to_numeric(a["Dividend"], errors="coerce")
    a = a[a["code"].isin(set(codes["code"]))]
    lo = max(pd.Timestamp(start) if start else first_report, first_report + pd.Timedelta(days=1))
    if start and pd.Timestamp(start) <= first_report:
        notes.append(f"start moved to {lo.date()}: the first ISF report is {first_report.date()}")
    hi = pd.Timestamp(end) if end else a["ex_date"].max()
    a = a[a["ex_date"].between(lo, hi)]
    excluded = [a[a["Dividend"].isna() | a["Dividend"].le(0)].assign(reason="no dividend amount")]
    a = a[a["Dividend"].gt(0)]
    a = a.drop_duplicates(["code", "ex_date", "Dividend"]).copy()

    # latest report strictly before the ex-date
    a = a.sort_values("ex_date")
    a = pd.merge_asof(a, rep[["as_of"]].assign(key=rep["as_of"]),
                      left_on="ex_date", right_on="key", allow_exact_matches=False,
                      direction="backward").drop(columns="key")
    a = a.merge(snap[["as_of", "code", "company", "holding_shares", "fair_value_gbp000",
                      "weight", "price_pence", "dividend_ccy_if_not_pence", "dividend_units"]],
                on=["as_of", "code"], how="left")
    out_of_index = a["holding_shares"].isna()
    excluded.append(a[out_of_index].assign(reason="not in the index at the latest report before the ex-date"))
    a = a[~out_of_index].copy()

    a, conv = _convert_foreign(a, "price_pence", "dividend_ccy_if_not_pence", cfg, notes)

    # share-count events (consolidations, large buybacks/issues) between two reports: from the
    # given date use the next report's holding, scaled back by the fund's overall growth
    a["shares_from"] = "report"
    if cfg["share_events"]:
        nxt = {pd.Timestamp(x): pd.Timestamp(y) for x, y in zip(reports[:-1], reports[1:])}
        for code, when in cfg["share_events"].items():
            when = pd.Timestamp(when)
            # only dividends after the event that still use a report from before it
            hit = a["code"].eq(code) & a["ex_date"].ge(when) & a["as_of"].lt(when)
            for rep_date in a.loc[hit, "as_of"].unique():
                rep_date = pd.Timestamp(rep_date)
                if rep_date not in nxt:
                    continue
                A = snap[snap["as_of"] == rep_date].set_index("company")["holding_shares"]
                B = snap[snap["as_of"] == nxt[rep_date]].set_index("company")["holding_shares"]
                k = (B / A).dropna().median()
                nb = snap[(snap["as_of"] == nxt[rep_date]) & (snap["code"] == code)]["holding_shares"]
                if len(nb):
                    rows = hit & a["as_of"].eq(rep_date)
                    a.loc[rows, "holding_shares"] = float(nb.iloc[0]) / k
                    a.loc[rows, "shares_from"] = f"next report ({nxt[rep_date].date()})"
    a["t_minus_1"] = a["ex_date"].map(lv.trading_day_before)
    a = a.merge(rep[["as_of", "valuation_day", "level_report", "mv_gbp000"]], on="as_of")
    a["level_t1"] = a["t_minus_1"].map(lv.level)
    no_level = a["level_t1"].isna()
    if no_level.any():
        excluded.append(a[no_level].assign(reason="no FTSE future level on T-1"))
        a = a[~no_level].copy()
    a["mv_t1_gbp"] = a["mv_gbp000"] * 1000.0 * a["level_t1"] / a["level_report"]
    a["contrib_bps"] = 1e4 * (a["div_pence"] / 100.0) * a["holding_shares"] / a["mv_t1_gbp"]
    a["div_yield"] = a["contrib_bps"] / 1e4 / a["weight"]
    a, dropped = _apply_exclusions(a, cfg)
    excluded.append(dropped)
    a = a.rename(columns={"as_of": "report_as_of"}).sort_values(["ex_date", "Company"]).reset_index(drop=True)

    exdiv = (a.groupby("ex_date").agg(n_stocks=("code", "size"), drop_bps=("contrib_bps", "sum"))
               .sort_index())

    # which part of each report's index has no dividend history at all in the workbook
    have = set(pd.read_excel(cfg["div_file"], sheet_name=cfg["updated_sheet"], usecols=["Column2"])
               ["Column2"].dropna().astype(str))
    cov = snap.assign(has_dividend_rows=snap["code"].isin(have))
    note = dict(zip(codes["code"], codes["note"]))
    missing = (cov[~cov["has_dividend_rows"]]
               .assign(note=lambda x: x["code"].map(note))
               .pivot_table(index=["company", "code", "note"], columns="as_of",
                            values="weight", aggfunc="sum"))
    if len(missing):
        missing.columns = [pd.Timestamp(c).date() for c in missing.columns]

    notes.append(f"reports used: {', '.join(str(pd.Timestamp(r).date()) for r in reports)}")
    notes.append(_summarise(exdiv))
    return DropResult("updated", a, exdiv, converted=conv,
                      excluded=pd.concat(excluded, ignore_index=True), coverage=missing,
                      specials=_possible_specials(a, cfg), notes=notes)


def expected_drops(source="updated", start=None, end=None, **kw) -> DropResult:
    source = str(source).strip().lower()
    if source == "updated":
        return updated_drops(start, end, **kw)
    if source == "original":
        return original_drops(start, end, **kw)
    raise ValueError("WEIGHTS_SOURCE must be 'updated' or 'original'")

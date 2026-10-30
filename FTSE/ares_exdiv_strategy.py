r"""Full fill-simulated FTSE ex-dividend strategy with a DAX hedge.

Generate the input first with the research environment, then run this file with
Ares' interpreter:

    C:\repo\ex_div_dates_research\.venv\Scripts\python.exe C:\repo\ex_div_dates_research\FTSE\export_for_ares.py
    cd C:\repo\ares
    C:\repo\ex_div_dates_research\.venv\Scripts\python.exe C:\repo\ex_div_dates_research\FTSE\ares_exdiv_strategy.py

Rules exported by ``export_for_ares.py``:

* long FTSE when ``0 < expected drop <= 5 bps``;
* short FTSE when ``expected drop > 5 bps``;
* reverse-direction DAX hedge using a beta estimated from the previous 60 eligible
  sessions only; and
* updated historical FTSE weights from the dated iShares holdings reports.

The hedge is an actual multi-leg Ares simulation.  FDAX and Mini-DAX are used for
the coarsest low-contract-count hedge.  If their nearest combination misses the
target notional by more than ``HEDGE_ERROR_LIMIT``, Micro-DAX refines the residual.
All legs are worked and filled independently by the Ares queue model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from ares import (
    Backtest, BacktestConfig, ExchangeConfig, FrontMonths, StaticData, Strategy,
    StrategyConfig, register_strategy,
)
from ares.actions import Action, OrderSide, TimeInForce
from ares.objects import Fill, TimerEvent
from ares.stream import MarketDataStream

ROOT = Path(r"C:\repo\ex_div_dates_research\FTSE")
INPUTS = ROOT / "ares_inputs.json"
OUT_DIR = ROOT / "ares_runs" / "full_hedged"

ASSETS = ("FT", "DAX", "FDXM", "FDXS")
DAX_ASSETS = ("DAX", "FDXM", "FDXS")
POINT_VALUE = {"FT": 10.0, "DAX": 25.0, "FDXM": 5.0, "FDXS": 1.0}
TICK_SIZE = {"FT": 0.5, "DAX": 1.0, "FDXM": 1.0, "FDXS": 1.0}

# GBP per EUR fallback. Normal runs use the event-specific same-expiry 6E/6B
# cross exported in ares_inputs.json. This remains for manual/legacy inputs and
# Ares' engine-level equity accounting, whose StaticData FX map is not time-varying.
EURGBP_FALLBACK = 0.87

# Per-lot, per-side fees in each contract's native currency. FT is GBP; the DAX
# family is EUR. Consolidated event P&L converts the latter with event-specific FX.
# Sources: eurex.com DAX/Mini-DAX/Micro-DAX product transaction-fee tables.
FEE_PER_LOT_NATIVE = {
    "FT": 0.50,
    "DAX": 0.80,
    "FDXM": 0.20,
    "FDXS": 0.08,
}
FEE_PER_LOT_GBP_FALLBACK = {
    asset: fee if asset == "FT" else fee * EURGBP_FALLBACK
    for asset, fee in FEE_PER_LOT_NATIVE.items()
}

# 08:00 London entry and 10:00 London exit, matching the research notebook.
OPEN_LOCAL = time(8, 0)
ENTRY_MIN = 5
EXIT_START_MIN = 115
FLAT_BY_MIN = 120
PAD_MIN = 2
CROSS_LAST_SEC = 30
EVERY = timedelta(seconds=1)
WARMUP = timedelta(minutes=5)

# Contract-selection rule discussed in the research: use Micro-DAX only when the
# nearest FDAX/Mini combination breaches this percentage of target hedge notional.
USE_DAX_HEDGE = True
HEDGE_ERROR_LIMIT = 0.05
SLICE_LOTS = {"FT": 5, "DAX": 1, "FDXM": 2, "FDXS": 5}

SEND_LATENCY = timedelta(milliseconds=1)
RECV_LATENCY = timedelta(microseconds=150)
TOUCH_QUEUE_MULTIPLE = 2.0
LONDON = timezone.utc  # replaced below after importing zoneinfo without dateutil
try:
    from zoneinfo import ZoneInfo
    LONDON = ZoneInfo("Europe/London")
except ImportError:  # pragma: no cover - Python 3.14 always has zoneinfo
    pass


# A few DAX archive files contain quote heartbeat rows with every price/size field
# blank, usually well outside the requested window.  MarketDataStream parses before
# applying the window and its stock parser raises on those rows.  Install a narrow
# guard for this pipeline: only structurally incomplete T/Q rows are ignored; every
# valid row still goes through Ares' original trade/quote constructors and ordering.
SKIPPED_INCOMPLETE_ROWS: dict[str, int] = {}


def _complete_events(self, handle, asset, contract):
    trade, quote = self._trade, self._quote
    for line in handle:
        if not line or line[0] in {"#", "\n"}:
            continue
        row = line.split(",")
        if len(row) < 12:
            continue
        message_type = row[2]
        complete = (
            message_type == "T" and row[1] and row[7] and row[8]
        ) or (
            message_type == "Q" and row[1] and row[3] and row[4] and row[5] and row[6]
        )
        if not complete:
            if message_type in {"T", "Q"}:
                SKIPPED_INCOMPLETE_ROWS[asset] = SKIPPED_INCOMPLETE_ROWS.get(asset, 0) + 1
            continue
        if message_type == "T":
            yield trade(row, asset, contract)
        elif message_type == "Q":
            yield quote(row, asset, contract)


MarketDataStream._events = _complete_events


def nearest_int(x: float) -> int:
    """Round a non-negative contract target half-up, not bankers' rounding."""
    return int(math.floor(x + 0.5))


def hedge_contracts(
    target_gbp: float,
    dax_level: float,
    eurgbp: float = EURGBP_FALLBACK,
    error_limit: float = HEDGE_ERROR_LIMIT,
) -> dict:
    """Min-contract DAX mix, adding micros only when Mini granularity is too coarse."""
    if target_gbp <= 0 or dax_level <= 0 or eurgbp <= 0:
        raise ValueError("hedge notional, DAX level and EURGBP must be positive")

    micro_value = dax_level * eurgbp
    mini_value = 5 * micro_value
    mini_units = nearest_int(target_gbp / mini_value)
    mini_notional = mini_units * mini_value
    mini_error = abs(mini_notional - target_gbp) / target_gbp

    use_micro = mini_error > error_limit
    if use_micro:
        total_micro_units = nearest_int(target_gbp / micro_value)
    else:
        total_micro_units = mini_units * 5

    fdax = total_micro_units // 25
    residual = total_micro_units % 25
    fdxm = residual // 5
    fdxs = residual % 5
    actual_gbp = total_micro_units * micro_value
    signed_error_gbp = actual_gbp - target_gbp
    return {
        "DAX": fdax,
        "FDXM": fdxm,
        "FDXS": fdxs,
        "used_micro": bool(use_micro),
        "target_gbp": target_gbp,
        "actual_gbp": actual_gbp,
        "signed_error_gbp": signed_error_gbp,
        "error_pct": abs(signed_error_gbp) / target_gbp * 100,
        "mini_only_error_pct": mini_error * 100,
    }


@dataclass(kw_only=True)
class ExDivHedgedParams(StrategyConfig):
    events: dict[str, dict] = field(default_factory=dict)
    use_dax_hedge: bool = USE_DAX_HEDGE
    hedge_error_limit: float = HEDGE_ERROR_LIMIT
    eurgbp: float = EURGBP_FALLBACK
    entry_min: int = ENTRY_MIN
    exit_start_min: int = EXIT_START_MIN
    flat_by_min: int = FLAT_BY_MIN
    cross_last_sec: int = CROSS_LAST_SEC
    passive_entry: bool = True
    passive_exit: bool = True
    slice_lots: dict[str, int] = field(default_factory=lambda: dict(SLICE_LOTS))
    every: timedelta = EVERY


@register_strategy("ftse_exdiv_dax_hedged", ExDivHedgedParams)
class FtseExDivDaxHedged(Strategy):
    """Combined long/short FTSE rule with an opposite-direction DAX hedge."""

    def __init__(self, static_data: StaticData, config: ExDivHedgedParams):
        super().__init__(static_data, config)
        self.plans: dict[str, dict] = {}
        self.fill_execution: list[dict] = []
        self.add_timer(config.every, self.step)

    def on_exchange_message(self, message) -> list[Action]:
        """Record the top of book visible when each fill reaches the strategy."""
        if isinstance(message, Fill):
            book = self.book_of(message.asset)
            self.fill_execution.append({
                "ts": message.ts.isoformat(),
                "date": message.ts.date().isoformat(),
                "contract": message.contract,
                "asset": message.asset,
                "order_id": message.order_id,
                "side": message.side.name,
                "qty": message.qty,
                "price": message.price,
                "is_maker": int(message.is_maker),
                "observed_bid": book.bid if book is not None else float("nan"),
                "observed_ask": book.ask if book is not None else float("nan"),
                "observed_mid": book.mid if book is not None else float("nan"),
            })
        return []

    def _capture_reference(self, plan: dict, phase: str) -> None:
        """Snapshot executable BBOs at the scheduled entry or exit timestamp."""
        key = f"{phase}_bbo"
        if key in plan:
            return
        required = [asset for asset, target in plan["targets"].items() if target]
        books = {asset: self.book_of(asset) for asset in required}
        if any(book is None for book in books.values()):
            return
        plan[f"{phase}_reference_ts"] = self.now.isoformat()
        plan[key] = {
            asset: {"bid": book.bid, "ask": book.ask, "mid": book.mid}
            for asset, book in books.items()
        }

    def _marks(self) -> tuple[datetime, datetime, datetime, datetime]:
        start = datetime.combine(self.now.date(), self.config.trade_from)
        return (
            start,
            start + timedelta(minutes=self.config.entry_min),
            start + timedelta(minutes=self.config.exit_start_min),
            start + timedelta(minutes=self.config.flat_by_min),
        )

    def _event(self) -> dict | None:
        return self.config.events.get(self.now.date().isoformat())

    def _build_plan(self, event: dict) -> dict | None:
        key = self.now.date().isoformat()
        if key in self.plans:
            return self.plans[key]

        ft_book = self.book_of("FT")
        dax_book = self.book_of("DAX")
        if ft_book is None or (self.config.use_dax_hedge and dax_book is None):
            return None

        direction = str(event["direction"]).lower()
        ft_sign = 1 if direction == "long" else -1
        ft_lots = int(event["ftse_lots"])
        targets = {asset: 0 for asset in self.assets}
        targets["FT"] = ft_sign * ft_lots

        ft_mid = ft_book.mid
        dax_mid = dax_book.mid if dax_book is not None else float("nan")
        plan = {
            "date": key,
            "direction": direction,
            "drop_bps": float(event["drop_bps"]),
            "beta_60": float(event["beta_60"]),
            "eurgbp_entry": float(event.get("eurgbp_entry", self.config.eurgbp)),
            "eurgbp_exit": float(event.get("eurgbp_exit", self.config.eurgbp)),
            "fx_contract": str(event.get("fx_contract", "fallback")),
            "ftse_lots": ft_lots,
            "ftse_reference": ft_mid,
            "dax_reference": dax_mid,
            "targets": targets,
            "used_micro": False,
            "target_hedge_gbp": 0.0,
            "actual_hedge_gbp": 0.0,
            "hedge_error_pct": 0.0,
            "mini_only_error_pct": 0.0,
        }

        if self.config.use_dax_hedge:
            target_gbp = (
                float(event["beta_60"]) * ft_lots * ft_mid * POINT_VALUE["FT"]
            )
            mix = hedge_contracts(
                target_gbp, dax_mid, plan["eurgbp_entry"],
                self.config.hedge_error_limit,
            )
            hedge_sign = -ft_sign
            for asset in DAX_ASSETS:
                targets[asset] = hedge_sign * int(mix[asset])
            plan.update({
                "used_micro": mix["used_micro"],
                "target_hedge_gbp": mix["target_gbp"],
                "actual_hedge_gbp": mix["actual_gbp"],
                "hedge_error_pct": mix["error_pct"],
                "mini_only_error_pct": mix["mini_only_error_pct"],
            })

        self.plans[key] = plan
        hedge_text = ", ".join(
            f"{asset} {targets[asset]:+d}" for asset in DAX_ASSETS
            if targets[asset]
        ) or "unhedged"
        print(
            f"  plan {key}: {direction} FT {abs(targets['FT'])} | {hedge_text} | "
            f"beta {plan['beta_60']:.3f} | EURGBP {plan['eurgbp_entry']:.4f} "
            f"({plan['fx_contract']}) | hedge error {plan['hedge_error_pct']:.2f}%"
        )
        return plan

    def step(self, timer: TimerEvent) -> list[Action]:
        if not self.in_session:
            return self.pull_all()

        event = self._event()
        if event is None:
            return self._work_targets({asset: 0 for asset in self.assets}, cross=True)

        start, entry_by, unwind_at, flat_by = self._marks()
        if self.now < start:
            return []

        plan = self._build_plan(event)
        if plan is None:
            return []

        if self.now >= unwind_at:
            targets = {asset: 0 for asset in self.assets}
            deadline = flat_by
            passive = self.config.passive_exit
        else:
            targets = plan["targets"]
            deadline = entry_by
            passive = self.config.passive_entry

        # Do not put on only one side of the hedge while another required book has
        # not arrived.  During flattening, work every position for which a book exists.
        if self.now < unwind_at:
            required = [a for a, target in targets.items() if target]
            if any(self.book_of(asset) is None for asset in required):
                return []
            self._capture_reference(plan, "entry")

        if self.now >= flat_by:
            self._capture_reference(plan, "exit")

        cross = (not passive) or (
            self.now >= deadline - timedelta(seconds=self.config.cross_last_sec)
        )
        return self._work_targets(targets, cross=cross)

    def _work_targets(self, targets: dict[str, int], cross: bool) -> list[Action]:
        actions: list[Action] = []
        for asset in self.assets:
            target = int(targets.get(asset, 0))
            book = self.book_of(asset)
            if book is None:
                continue
            actions += self._go_to(asset, target, book, cross)
        return actions

    def _go_to(self, asset: str, target: int, book, cross: bool) -> list[Action]:
        need = target - self.position(asset)
        if need == 0:
            return self.pull(asset)
        max_qty = int(self.config.slice_lots.get(asset, abs(need)))
        if not cross:
            return self.work_to(asset, target, book.bid, book.ask, max_qty=max_qty)
        side = OrderSide.BUY if need > 0 else OrderSide.SELL
        price = book.ask if need > 0 else book.bid
        qty = min(abs(need), max_qty)
        return self.pull(asset, side) + self.quote(
            asset, side, price, qty, tif=TimeInForce.IOC
        )


def london_open_in_utc(day: date) -> time:
    local = datetime.combine(day, OPEN_LOCAL, tzinfo=LONDON)
    return local.astimezone(timezone.utc).replace(tzinfo=None).time()


def load_inputs() -> tuple[dict, dict, FrontMonths]:
    blob = json.loads(INPUTS.read_text(encoding="utf-8"))
    if blob.get("schema_version") != 3:
        raise ValueError("ares_inputs.json is stale; rerun export_for_ares.py")
    events = dict(blob["events"])
    assets = tuple(blob["assets"])
    if assets != ASSETS:
        raise ValueError(f"expected assets {ASSETS}, got {assets}")
    mapping = {
        (asset, date.fromisoformat(day)): contract
        for asset, by_day in blob["front_months"].items()
        for day, contract in by_day.items()
    }
    return blob, events, FrontMonths(mapping)


def filter_dates(events: dict, args) -> list[date]:
    days = sorted(date.fromisoformat(k) for k in events)
    if args.start:
        days = [d for d in days if d >= date.fromisoformat(args.start)]
    if args.end:
        days = [d for d in days if d <= date.fromisoformat(args.end)]
    if args.max_events is not None:
        days = days[:args.max_events]
    return days


def leg_stats(fills, static: StaticData) -> dict[tuple[date, str], dict]:
    grouped: dict[tuple[date, str], dict] = {}
    for fill in fills:
        row = grouped.setdefault(
            (fill.ts.date(), fill.asset),
            {"buy_qty": 0, "buy_ntl": 0.0, "sell_qty": 0,
             "sell_ntl": 0.0, "maker": 0, "n_fills": 0},
        )
        if fill.side is OrderSide.BUY:
            row["buy_qty"] += fill.qty
            row["buy_ntl"] += fill.qty * fill.price
        else:
            row["sell_qty"] += fill.qty
            row["sell_ntl"] += fill.qty * fill.price
        row["maker"] += int(fill.is_maker)
        row["n_fills"] += 1

    for (day, asset), row in grouped.items():
        multiplier = static.point_values[asset] * static.fx_rates.get(asset, 1.0)
        native_multiplier = static.point_values[asset]
        row["gross_native"] = (
            row["sell_ntl"] - row["buy_ntl"]
        ) * native_multiplier
        row["fees_native"] = (
            row["buy_qty"] + row["sell_qty"]
        ) * FEE_PER_LOT_NATIVE.get(asset, 0.0)
        row["gross_gbp"] = (row["sell_ntl"] - row["buy_ntl"]) * multiplier
        row["fees_gbp"] = (
            row["buy_qty"] + row["sell_qty"]
        ) * static.all_in_fees.get(asset, 0.0)
        row["net_gbp"] = row["gross_gbp"] - row["fees_gbp"]
        row["closed"] = row["buy_qty"] == row["sell_qty"]
    return grouped


def per_date_pnl(fills, plans: dict[str, dict], static: StaticData) -> list[dict]:
    legs = leg_stats(fills, static)
    rows: list[dict] = []
    for key, plan in sorted(plans.items()):
        day = date.fromisoformat(key)
        ft = legs.get((day, "FT"))
        if not ft:
            continue
        ft_target = int(plan["targets"]["FT"])
        if ft_target > 0:
            entry = ft["buy_ntl"] / ft["buy_qty"] if ft["buy_qty"] else float("nan")
            exit_px = ft["sell_ntl"] / ft["sell_qty"] if ft["sell_qty"] else float("nan")
        else:
            entry = ft["sell_ntl"] / ft["sell_qty"] if ft["sell_qty"] else float("nan")
            exit_px = ft["buy_ntl"] / ft["buy_qty"] if ft["buy_qty"] else float("nan")
        roundtrip_lots = min(ft["buy_qty"], ft["sell_qty"])
        ft_notional = roundtrip_lots * entry * POINT_VALUE["FT"]

        active_pairs = [(asset, legs[(day, asset)]) for asset in ASSETS
                        if (day, asset) in legs]
        active_legs = [row for _, row in active_pairs]
        fx_entry = float(plan["eurgbp_entry"])
        fx_exit = float(plan["eurgbp_exit"])
        fx_fee = (fx_entry + fx_exit) / 2.0
        converted = {
            asset: {
                "gross_gbp": row["gross_native"] * (1.0 if asset == "FT" else fx_exit),
                "fees_gbp": row["fees_native"] * (1.0 if asset == "FT" else fx_fee),
            }
            for asset, row in active_pairs
        }
        total_gross = sum(x["gross_gbp"] for x in converted.values())
        total_fees = sum(x["fees_gbp"] for x in converted.values())
        total_net = total_gross - total_fees
        dax_gross = sum(converted.get(a, {}).get("gross_gbp", 0.0) for a in DAX_ASSETS)
        dax_fees = sum(converted.get(a, {}).get("fees_gbp", 0.0) for a in DAX_ASSETS)
        n_fills = sum(x["n_fills"] for x in active_legs)
        maker = sum(x["maker"] for x in active_legs)
        closed = all(x["closed"] for x in active_legs)

        rows.append({
            "date": key,
            "direction": plan["direction"],
            "drop_bps": plan["drop_bps"],
            "beta_60": plan["beta_60"],
            "eurgbp_entry": fx_entry,
            "eurgbp_exit": fx_exit,
            "fx_contract": plan["fx_contract"],
            "ftse_lots": roundtrip_lots,
            "ftse_entry": entry,
            "ftse_exit": exit_px,
            "fdax_lots": abs(int(plan["targets"].get("DAX", 0))),
            "mini_dax_lots": abs(int(plan["targets"].get("FDXM", 0))),
            "micro_dax_lots": abs(int(plan["targets"].get("FDXS", 0))),
            "used_micro": plan["used_micro"],
            "target_hedge_gbp": plan["target_hedge_gbp"],
            "actual_hedge_gbp": plan["actual_hedge_gbp"],
            "hedge_error_pct": plan["hedge_error_pct"],
            "mini_only_error_pct": plan["mini_only_error_pct"],
            "ftse_gross_gbp": ft["gross_gbp"],
            "ftse_fees_gbp": ft["fees_gbp"],
            "dax_gross_gbp": dax_gross,
            "dax_fees_gbp": dax_fees,
            "gross_gbp": total_gross,
            "fees_gbp": total_fees,
            "net_gbp": total_net,
            "net_bps": total_net / ft_notional * 1e4 if ft_notional else float("nan"),
            "maker_pct": maker / n_fills * 100 if n_fills else float("nan"),
            "n_fills": n_fills,
            "closed": closed,
        })
    return rows


def summarise(rows: list[dict]) -> None:
    closed = [r for r in rows if r["closed"] and math.isfinite(r["net_bps"])]
    if not closed:
        print("no closed round-trip events to summarise")
        return
    net = [r["net_bps"] for r in closed]
    gbp = [r["net_gbp"] for r in closed]
    n = len(net)
    years = max((date.fromisoformat(closed[-1]["date"])
                 - date.fromisoformat(closed[0]["date"])).days / 365.25,
                1 / 365.25)
    per_year = n / years
    mean = sum(net) / n
    sd = (sum((x - mean) ** 2 for x in net) / (n - 1)) ** 0.5 if n > 1 else float("nan")
    print("\n" + "=" * 82)
    print("FULL ARES COMBINED FTSE EX-DIV + 60-SESSION DAX HEDGE")
    print("=" * 82)
    print(f"closed events {n}/{len(rows)} | hit rate "
          f"{sum(x > 0 for x in net) / n * 100:.1f}% | "
          f"mean {mean:+.2f} bps | total GBP {sum(gbp):+,.0f}")
    print(f"fees GBP {sum(r['fees_gbp'] for r in closed):,.0f} | "
          f"micro used on {sum(r['used_micro'] for r in closed)} events | "
          f"mean hedge mismatch {sum(r['hedge_error_pct'] for r in closed) / n:.2f}%")
    if n > 1 and sd > 0:
        print(f"annualised event Sharpe: {mean / sd * math.sqrt(per_year):.2f} "
              f"({per_year:.1f} trades/year)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="first event date, YYYY-MM-DD")
    parser.add_argument("--end", help="last event date, YYYY-MM-DD")
    parser.add_argument("--max-events", type=int, help="run only the first N selected events")
    parser.add_argument("--no-hedge", action="store_true", help="run FTSE legs without DAX")
    parser.add_argument("--ft-child", type=int, default=SLICE_LOTS["FT"])
    parser.add_argument("--fdax-child", type=int, default=SLICE_LOTS["DAX"])
    parser.add_argument("--mini-child", type=int, default=SLICE_LOTS["FDXM"])
    parser.add_argument("--micro-child", type=int, default=SLICE_LOTS["FDXS"])
    parser.add_argument("--cross-last-sec", type=int, default=CROSS_LAST_SEC,
                        help="cross any residual during the final N seconds")
    parser.add_argument("--aggressive-entry", action="store_true",
                        help="cross immediately instead of entering passively")
    parser.add_argument("--aggressive-exit", action="store_true",
                        help="cross immediately instead of exiting passively")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Ares result directory")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    child_values = [args.ft_child, args.fdax_child, args.mini_child, args.micro_child]
    if any(value <= 0 for value in child_values):
        parser.error("all child-order quantities must be positive integers")
    if args.cross_last_sec < 0:
        parser.error("--cross-last-sec must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    blob, events, front = load_inputs()
    traded = filter_dates(events, args)
    if not traded:
        raise SystemExit("no exported events in the requested date range")

    groups: dict[time, list[date]] = {}
    for day in traded:
        groups.setdefault(london_open_in_utc(day), []).append(day)

    static = StaticData(
        point_values=dict(POINT_VALUE),
        tick_sizes=dict(TICK_SIZE),
        fx_rates={"FT": 1.0, "DAX": EURGBP_FALLBACK,
                  "FDXM": EURGBP_FALLBACK, "FDXS": EURGBP_FALLBACK},
        all_in_fees=dict(FEE_PER_LOT_GBP_FALLBACK),
    )
    exchange = ExchangeConfig(
        send_latency=SEND_LATENCY,
        recv_latency=RECV_LATENCY,
        touch_queue_multiple=TOUCH_QUEUE_MULTIPLE,
    )

    print(f"input schema {blob['schema_version']} | weights {blob['meta']['weights_source']} | "
          f"{len(traded)} events in {len(groups)} UTC-offset group(s)")
    print(f"rule: long <= {blob['meta']['small_max_drop_bps']:.2f} bps; "
          f"short > {blob['meta']['large_min_drop_bps']:.2f} bps; "
          f"DAX hedge {'OFF' if args.no_hedge else 'ON'}")

    all_fills = []
    all_fill_execution: list[dict] = []
    all_plans: dict[str, dict] = {}
    grand_equity = 0.0
    for open_utc, days in sorted(groups.items()):
        until = (datetime.combine(date(2000, 1, 1), open_utc)
                 + timedelta(minutes=FLAT_BY_MIN + PAD_MIN)).time()
        selected_events = {d.isoformat(): events[d.isoformat()] for d in days}
        print(f"\n=== {open_utc} UTC (= 08:00 London), {len(days)} dates "
              f"{days[0]} .. {days[-1]} ===")
        strategy = FtseExDivDaxHedged(static, ExDivHedgedParams(
            assets=ASSETS,
            trade_from=open_utc,
            trade_until=until,
            warmup=WARMUP,
            events=selected_events,
            use_dax_hedge=not args.no_hedge,
            passive_entry=not args.aggressive_entry,
            passive_exit=not args.aggressive_exit,
            cross_last_sec=args.cross_last_sec,
            slice_lots={"FT": args.ft_child, "DAX": args.fdax_child,
                        "FDXM": args.mini_child, "FDXS": args.micro_child},
        ))
        state = Backtest(BacktestConfig(
            exchange_config=exchange,
            strategy=strategy,
            dates=days,
            front_months=front,
            out_dir=str(Path(args.out_dir) / f"utc{open_utc.strftime('%H%M')}"),
        )).run(show_progress=not args.no_progress)

        pf = state.portfolio
        maker = sum(fill.is_maker for fill in state.fills)
        print(f"  fills {len(state.fills)} ({maker} maker, "
              f"{len(state.fills) - maker} taker) | realised {pf.realized():,.0f} | "
              f"fees {pf.fees():,.0f} | equity {pf.equity():,.0f}")
        left = {contract: pos.qty for contract, pos in pf.positions.items() if pos.qty}
        if left:
            print(f"  WARNING not flat at the end: {left}")
        grand_equity += pf.equity()
        all_fills.extend(state.fills)
        all_fill_execution.extend(strategy.fill_execution)
        all_plans.update(strategy.plans)

    rows = per_date_pnl(all_fills, all_plans, static)
    output = Path(args.out_dir) / "per_date_pnl.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nper-date P&L -> {output}")

    execution_output = Path(args.out_dir) / "execution_fills.csv"
    execution_output.parent.mkdir(parents=True, exist_ok=True)
    execution_fields = [
        "ts", "date", "contract", "asset", "order_id", "side", "qty",
        "price", "is_maker", "observed_bid", "observed_ask", "observed_mid",
    ]
    with execution_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=execution_fields)
        writer.writeheader()
        writer.writerows(all_fill_execution)

    reference_output = Path(args.out_dir) / "execution_references.csv"
    reference_fields = ["date", "phase", "ts", "asset", "target", "bid", "ask", "mid"]
    reference_rows = []
    for day, plan in sorted(all_plans.items()):
        for phase in ("entry", "exit"):
            for asset, book in plan.get(f"{phase}_bbo", {}).items():
                reference_rows.append({
                    "date": day,
                    "phase": phase,
                    "ts": plan.get(f"{phase}_reference_ts"),
                    "asset": asset,
                    "target": int(plan["targets"].get(asset, 0)),
                    **book,
                })
    with reference_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reference_fields)
        writer.writeheader()
        writer.writerows(reference_rows)
    print(f"execution fills -> {execution_output}")
    print(f"scheduled BBO references -> {reference_output}")
    print(f"TOTAL Ares engine equity across UTC groups (static FX fallback): "
          f"GBP {grand_equity:,.0f}")
    print("Authoritative consolidated GBP P&L below uses each event's 6E/6B cross.")
    summarise(rows)
    if SKIPPED_INCOMPLETE_ROWS:
        print("incomplete raw rows skipped before Ares parsing: " + ", ".join(
            f"{asset}={count}" for asset, count in sorted(SKIPPED_INCOMPLETE_ROWS.items())
        ))
    if any(not row["closed"] for row in rows):
        print("WARNING: at least one event was not fully flat; inspect its fills before use")


if __name__ == "__main__":
    main()

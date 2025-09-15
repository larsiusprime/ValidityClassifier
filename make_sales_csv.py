#!/usr/bin/env python3
"""
Synthetic property sales data generator that writes CSV instead of returning a DataFrame.

Spec recap:
- key_primary: 10-digit unique ID per row
- sale_date: mm/dd/yyyy
- sale_price: [0, 1_200_000], with ~15% "very low" (<= 999) values; much higher chance
  of very low when buyer/seller surnames match
- buyer_name: person or company; buyer_surname blank when company
- seller_name: person or company; seller_surname blank when company
- market_area: from a pool of about 50 options; prices cluster within ~±30% of an
  area-specific median setpoint

Usage examples:
  python make_sales_csv.py --rows 10000 --out sales_data.csv
  python make_sales_csv.py --rows 10000  # writes ./sales_data.csv by default
  python make_sales_csv.py --out -       # writes to stdout

This script only requires numpy and pandas (both are already in requirements.txt).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd


# ---------------------
# Defaults
# ---------------------
DEFAULT_ROWS = 10_000
DEFAULT_SEED = 42
DEFAULT_DATE_START = datetime(2015, 1, 1)
DEFAULT_DATE_END = datetime(2025, 9, 15)
DEFAULT_AREAS = 50
DEFAULT_OUT = "sales_data.csv"


FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"
]

SURNAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"
]

COMPANY_ROOTS = [
    "Acme", "Globex", "Umbrella", "Initech", "Soylent", "Stark", "Wayne", "Wonka",
    "Hooli", "Vehement", "Massive Dynamic", "Cyberdyne", "Gekko", "Tyrell", "Oscorp",
    "Vandelay", "Duff", "Paper Street", "Monarch", "Gringotts", "Prestige", "Dunder Mifflin",
    "Blue Sun", "Octan", "MomCorp"
]
COMPANY_SUFFIXES = ["LLC", "Inc", "Ltd", "PLC", "Corp", "GmbH", "S.A.", "S.p.A."]


def random_company_name(rng: np.random.Generator, size: int) -> np.ndarray:
    roots = rng.choice(COMPANY_ROOTS, size=size)
    sufs = rng.choice(COMPANY_SUFFIXES, size=size)
    return np.char.add(np.char.add(roots, " "), sufs)


def random_individual_name(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    first = rng.choice(FIRST_NAMES, size=size)
    last = rng.choice(SURNAMES, size=size)
    full = np.char.add(np.char.add(first, " "), last)
    return full, last


def random_dates(rng: np.random.Generator, n: int, start: datetime, end: datetime) -> np.ndarray:
    span_days = (end - start).days
    offsets = rng.integers(0, span_days + 1, size=n)
    dates = np.array([start + timedelta(days=int(d)) for d in offsets])
    return np.array([dt.strftime("%m/%d/%Y") for dt in dates])


def unique_10_digit_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate n unique 10-digit zero-padded IDs in O(n) memory.

    Strategy: choose a random base in [0, 10^10 - n], then take base + perm(range(n)).
    This yields n unique integers in [0, 10^10) without allocating huge arrays.
    Leading zeros are allowed and preserved via formatting.
    """
    limit = 10_000_000_000  # 10^10
    if n > limit:
        raise ValueError("Requested more IDs than the 10-digit space allows")

    # Pick a random contiguous window of length n
    base = int(rng.integers(0, limit - n + 1))

    # Create [0..n-1] and shuffle to avoid monotone IDs
    seq = np.arange(n, dtype=np.int64)
    rng.shuffle(seq)

    ids = base + seq
    return np.array([f"{x:010d}" for x in ids])


def build_dataframe(
    n_rows: int,
    seed: int,
    n_areas: int,
    date_start: datetime,
    date_end: datetime,
    base_low_prob: float = 0.15,
    match_low_prob: float = 0.60,
    very_low_max: int = 999,
    area_median_min: int = 120_000,
    area_median_max: int = 900_000,
    mult_mean: float = 1.0,
    mult_std: float = 0.10,
    mult_clip_low: float = 0.70,
    mult_clip_high: float = 1.30,
    buyer_company_prob: float = 0.45,
    seller_company_prob: float = 0.45,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    market_areas = np.array([f"MA-{i:03d}" for i in range(1, n_areas + 1)])
    area_medians = rng.integers(area_median_min, area_median_max + 1, size=n_areas)
    median_by_area = dict(zip(market_areas, area_medians))

    area = rng.choice(market_areas, size=n_rows)
    sale_date = random_dates(rng, n_rows, date_start, date_end)

    buyer_is_company = rng.random(n_rows) < buyer_company_prob
    seller_is_company = rng.random(n_rows) < seller_company_prob

    buyer_names = np.empty(n_rows, dtype=object)
    buyer_surnames = np.empty(n_rows, dtype=object)

    buyer_names[buyer_is_company] = random_company_name(rng, int(buyer_is_company.sum()))
    buyer_surnames[buyer_is_company] = ""

    b_full, b_last = random_individual_name(rng, int((~buyer_is_company).sum()))
    buyer_names[~buyer_is_company] = b_full
    buyer_surnames[~buyer_is_company] = b_last

    seller_names = np.empty(n_rows, dtype=object)
    seller_surnames = np.empty(n_rows, dtype=object)

    seller_names[seller_is_company] = random_company_name(rng, int(seller_is_company.sum()))
    seller_surnames[seller_is_company] = ""

    s_full, s_last = random_individual_name(rng, int((~seller_is_company).sum()))
    seller_names[~seller_is_company] = s_full
    seller_surnames[~seller_is_company] = s_last

    non_blank = (buyer_surnames != "") & (seller_surnames != "")
    surnames_match = non_blank & (buyer_surnames == seller_surnames)

    p_low = np.full(n_rows, base_low_prob)
    p_low[surnames_match] = match_low_prob
    very_low_flag = rng.random(n_rows) < p_low

    medians_for_rows = np.vectorize(median_by_area.get)(area).astype(float)
    mult = rng.normal(mult_mean, mult_std, size=n_rows)
    mult = np.clip(mult, mult_clip_low, mult_clip_high)
    clustered_prices = medians_for_rows * mult

    very_low_prices = rng.integers(0, very_low_max + 1, size=n_rows)
    prices = np.where(very_low_flag, very_low_prices, clustered_prices)
    prices = np.clip(np.round(prices).astype(int), 0, 1_200_000)

    df = pd.DataFrame({
        "key_primary": unique_10_digit_ids(rng, n_rows),
        "sale_date": sale_date,
        "sale_price": prices,
        "buyer_name": buyer_names,
        "buyer_surname": buyer_surnames,
        "seller_name": seller_names,
        "seller_surname": seller_surnames,
        "market_area": area,
    })
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic property sales CSV.")
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Number of rows to generate (default: 10000)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    p.add_argument("--areas", type=int, default=DEFAULT_AREAS, help="Number of market areas (default: 50)")
    p.add_argument("--date-start", type=str, default=DEFAULT_DATE_START.strftime("%Y-%m-%d"), help="Start date YYYY-MM-DD (default: 2015-01-01)")
    p.add_argument("--date-end", type=str, default=DEFAULT_DATE_END.strftime("%Y-%m-%d"), help="End date YYYY-MM-DD (default: 2025-09-15)")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output CSV path or '-' for stdout (default: sales_data.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        date_start = datetime.strptime(args.date_start, "%Y-%m-%d")
        date_end = datetime.strptime(args.date_end, "%Y-%m-%d")
    except ValueError as e:
        print(f"Invalid date format: {e}", file=sys.stderr)
        sys.exit(2)
    if date_end < date_start:
        print("date-end must be >= date-start", file=sys.stderr)
        sys.exit(2)

    df = build_dataframe(
        n_rows=args.rows,
        seed=args.seed,
        n_areas=args.areas,
        date_start=date_start,
        date_end=date_end,
    )

    if args.out == "-":
        df.to_csv(sys.stdout, index=False)
    else:
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()

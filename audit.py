import argparse
import sqlite3
from contextlib import closing

import pandas as pd

DB = "data/prices.db"
SAMPLE_SIZE = 10
RANDOM_SEED = 42
MIN_GAP = 0.20
MAX_GAP = 0.50
SPEC_FIELDS = {
    "ram_gb": "RAM",
    "storage_gb": "storage",
    "cpu_model": "CPU model",
    "screen_in": "screen size",
}


def export_review_template(sample, path):
    review = sample[
        [
            "snapshot_date",
            "item_id",
            "title",
            "price",
            "estimated_asking_price",
            "price_gap_pct",
            "condition",
            "ram_gb",
            "storage_gb",
            "cpu_model",
            "screen_in",
            "url",
        ]
    ].copy()
    review["manual_review"] = ""
    review["verified_specs"] = ""
    review["observed_comparable_price"] = ""
    review["notes"] = ""
    review.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Sample candidate listings for manual review.")
    parser.add_argument(
        "--export",
        metavar="CSV_PATH",
        help="write the sampled listings and blank review fields to a CSV file",
    )
    args = parser.parse_args()

    with closing(sqlite3.connect(DB)) as con:
        listings = pd.read_sql("SELECT * FROM scored_listings", con)

    if listings.empty:
        raise SystemExit("No scored listings found. Run clean.py and model.py first.")

    latest_date = listings.snapshot_date.max()
    latest = listings[listings.snapshot_date == latest_date]
    candidates = latest[
        latest.price_gap_pct.between(MIN_GAP, MAX_GAP, inclusive="both")
        & latest.url.notna()
    ]

    if candidates.empty:
        raise SystemExit(f"No 20%-50% gap candidates found for snapshot {latest_date}.")

    sample = candidates.sample(
        n=min(SAMPLE_SIZE, len(candidates)),
        random_state=RANDOM_SEED,
    )
    print(f"Manual audit sample: {len(sample)} of {len(candidates)} candidates")
    print(f"Latest snapshot: {latest_date}")
    print("These are unverified candidates; inspect each listing before judging it.\n")

    for number, (_, row) in enumerate(sample.iterrows(), start=1):
        missing = [
            label for column, label in SPEC_FIELDS.items()
            if pd.isna(row[column])
        ]
        condition = row.condition if pd.notna(row.condition) else "Unknown"
        print(f"{number}. {row.title}")
        print(f"   Asking: ${row.price:.0f} | Model estimate: ${row.estimated_asking_price:.0f} | Gap: {row.price_gap_pct:.0%}")
        print(f"   Condition: {condition} | Seller feedback: {row.seller_feedback:.1f}%")
        print(f"   Missing parsed specs: {', '.join(missing) if missing else 'None'}")
        print(f"   Listing: {row.url}")

    if args.export:
        export_review_template(sample, args.export)
        print(f"\nReview template saved to {args.export}")


if __name__ == "__main__":
    main()

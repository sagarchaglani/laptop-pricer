# Laptop Listing Research

An eBay used-laptop listing monitor. It estimates typical **asking prices** from collected listing data and surfaces listings for human review. It does not verify that a listing is safe, working, or worth buying.

## Run locally

Requires Python 3.11. Install the pinned dependencies and prepare the local SQLite database:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Set `EBAY_ID` and `EBAY_SECRET` in your shell (or in an untracked `.env` file loaded by your own tooling), then run:

```bash
python collector.py
python clean.py
python model.py
streamlit run app.py
```

The collector calls eBay Browse for used, fixed-price listings. These are asking prices, not completed sales. The database is `data/prices.db`; the pipeline writes raw, cleaned, scored, and validation-metric tables there.

To sample candidates and keep a manual review record, run:

```bash
python audit.py --export data/manual_audit.csv
```

Inspect the listings and fill in `manual_review`, `verified_specs`, `observed_comparable_price`, and `notes` in the CSV. This is a human review aid, not completed-sale ground truth; review files contain listing information and should be handled consistently with eBay's terms.

## Deploy the dashboard

The simplest demo deployment is Streamlit Community Cloud backed by this GitHub repository and its daily GitHub Actions workflow:

1. Push the project and an initialized `data/prices.db` (with `clean_listings`, `scored_listings`, and `model_metrics` tables) to your GitHub default branch. To initialize it locally, configure eBay credentials and run `python collector.py`, `python clean.py`, and `python model.py`, then push the database.
2. In GitHub, add repository Actions secrets named `EBAY_ID` and `EBAY_SECRET`. In **Settings → Actions → General**, allow GitHub Actions to write repository contents.
3. In Streamlit Community Cloud, create an app from the repository's default branch with `app.py` as the entry point. The dashboard does not need the eBay credentials; keep those only in GitHub Actions secrets.
4. Run the `collect` workflow once using **Actions → collect → Run workflow**. It fetches listings, updates the database, and commits the snapshot. The workflow also runs daily at 06:00 UTC. Streamlit Cloud can then redeploy from the updated branch.

The demo stores the SQLite database in Git, so each daily update adds another database version to repository history. This is convenient for a small portfolio project, but it is not a scalable or multi-user storage design. For a larger deployment, move snapshots to durable managed database/object storage and add a migration/backup plan. Do not commit credentials or local `.env` files.

## Validation and trust

The model predicts log asking price from title-parsed RAM, storage, CPU family/model and generation, screen size, brand, and seller feedback. Listings missing RAM, brand, or CPU are not scored.

- With only one collection date, the pipeline reports **item-grouped cross-validation within that period**. It keeps repeated observations of one listing in the same fold, but this is not a test of future-period performance.
- After earlier and later snapshots exist, it trains on earlier dates and validates against the newest date. New item IDs are preferred for this test; if there are no new IDs, the dashboard labels the repeated-item holdout explicitly.
- Error is also reported by brand, asking-price range, and whether key specs were parsed. Small segment counts make those results unstable.
- An empirical error band is shown only when at least 20 later-snapshot, previously unseen listings are available. It uses the 90th percentile of absolute log errors from that holdout; it is descriptive, not a guaranteed confidence interval.

The reported MAPE and error band only describe performance against eBay advertised prices in this dataset. They are not confirmed sale-price accuracy, a buying recommendation, or a guarantee of future performance. This project does not collect completed-sale prices; a suitable, policy-compliant sold-price source would be needed to validate transaction value. A large gap is only a reason to investigate—not proof of a bargain.

Always open each listing and verify exact specifications, condition, activation or firmware locks, battery health, accessories, seller history, shipping, taxes, and return terms. Title parsing is imperfect; sellers may omit defects or disclose them only in photos or descriptions. Listings may change or disappear after collection. Check eBay's terms before publishing collected listing data.

## Tests

Run the focused pipeline tests with:

```bash
python -m unittest discover -s tests -v
```

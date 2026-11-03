import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import model
from audit import export_review_template
from clean import parse
from collector import request_headers


def listings(snapshot_date, prefix, count, brand="Dell"):
    rows = []
    for index in range(count):
        ram = [8, 16, 32][index % 3]
        price = 250 + ram * 20 + index * 3
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "item_id": f"{prefix}-{index}",
                "title": f"{brand} laptop {ram}GB RAM",
                "price": price,
                "currency": "USD",
                "condition": "Used",
                "seller_feedback": 99.0,
                "location": "US",
                "url": f"https://example.test/{prefix}-{index}",
                "ram_gb": ram,
                "storage_gb": 256 + (index % 2) * 256,
                "cpu": "i5",
                "cpu_model": f"i5-{11000 + index % 4}",
                "cpu_gen": 11,
                "screen_in": 14.0,
                "brand": brand,
                "not_laptop": False,
                "for_parts": False,
            }
        )
    return pd.DataFrame(rows)


class PipelineTests(unittest.TestCase):
    def test_title_parser_extracts_specs_and_rejects_parts(self):
        specs = parse("Lenovo ThinkPad i5-1135G7 16GB RAM 512GB SSD 14 inch")
        self.assertEqual(specs["ram_gb"], 16)
        self.assertEqual(specs["storage_gb"], 512)
        self.assertEqual(specs["cpu_model"], "i5-1135g7")
        self.assertEqual(specs["brand"], "lenovo")
        self.assertFalse(specs["for_parts"])
        self.assertTrue(parse("Laptop for parts only")["for_parts"])

    def test_collector_uses_bearer_token(self):
        self.assertEqual(
            request_headers("sample-token")["Authorization"],
            "Bearer sample-token",
        )

    def test_audit_export_includes_manual_review_fields(self):
        sample = listings("2026-11-03", "audit", 1)
        sample["estimated_asking_price"] = 500.0
        sample["price_gap_pct"] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            export_review_template(sample, path)
            review = pd.read_csv(path)
        self.assertIn("manual_review", review.columns)
        self.assertIn("observed_comparable_price", review.columns)
        self.assertTrue(pd.isna(review.loc[0, "manual_review"]))

    def test_temporal_validation_scores_latest_and_persists_metrics(self):
        clean = pd.concat(
            [
                listings("2026-11-02", "old", 30),
                listings("2026-11-03", "new", 20, brand="Asus"),
            ],
            ignore_index=True,
        )
        con = sqlite3.connect(":memory:")
        estimator = HistGradientBoostingRegressor(
            categorical_features="from_dtype",
            max_iter=5,
            random_state=42,
        )
        with patch("model.new_model", return_value=estimator):
            model.score_listings(clean, con)

        scored = pd.read_sql("SELECT * FROM scored_listings", con)
        metrics = pd.read_sql("SELECT * FROM model_metrics", con)
        overall = metrics[metrics.segment_type == "overall"].iloc[0]
        self.assertEqual(len(scored), 20)
        self.assertEqual(overall.validation_method, "temporal_holdout_new_items")
        self.assertEqual(overall.sample_count, 20)
        self.assertTrue(np.isfinite(overall.mape))
        self.assertTrue(scored.estimate_low.notna().all())
        self.assertTrue(scored.estimate_high.notna().all())
        con.close()

    def test_single_snapshot_uses_grouped_cv_without_error_band(self):
        clean = listings("2026-11-03", "only", 10)
        con = sqlite3.connect(":memory:")
        estimator = HistGradientBoostingRegressor(
            categorical_features="from_dtype",
            max_iter=5,
            random_state=42,
        )
        with patch("model.new_model", return_value=estimator):
            model.score_listings(clean, con)

        scored = pd.read_sql("SELECT * FROM scored_listings", con)
        metrics = pd.read_sql("SELECT * FROM model_metrics", con)
        overall = metrics[metrics.segment_type == "overall"].iloc[0]
        self.assertEqual(overall.validation_method, "item_grouped_cv_same_period")
        self.assertEqual(len(scored), 10)
        self.assertTrue(scored.estimate_low.isna().all())
        con.close()


if __name__ == "__main__":
    unittest.main()

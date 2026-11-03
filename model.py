import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

DB = "data/prices.db"
SPEC_FIELDS = ["ram_gb", "storage_gb", "cpu_model", "screen_in"]
CATEGORICAL_FIELDS = ["brand", "cpu", "cpu_model"]
NUMERIC_FIELDS = ["ram_gb", "storage_gb", "screen_in", "seller_feedback", "cpu_gen"]


def new_model():
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        max_iter=300,
        random_state=42,
    )


def features_for(frame, category_levels=None):
    features = frame[NUMERIC_FIELDS + CATEGORICAL_FIELDS].copy()
    for column in CATEGORICAL_FIELDS:
        if category_levels is None:
            features[column] = features[column].astype("category")
        else:
            known_values = features[column].where(
                features[column].isin(category_levels[column])
            )
            features[column] = pd.Categorical(
                known_values,
                categories=category_levels[column],
            )
    return features


def metric_rows(actual, predictions, validation_method, validation_date):
    actual = actual.reset_index(drop=True).copy()
    predictions = np.asarray(predictions, dtype=float)
    if len(actual) != len(predictions):
        raise ValueError("Prediction count must match the validation rows.")

    def summarize(indices, segment_type, segment):
        prices = actual.iloc[indices].price.to_numpy(dtype=float)
        estimates = predictions[indices]
        percentage_errors = np.abs(prices - estimates) / prices
        log_errors = np.abs(np.log(prices / estimates))
        return {
            "snapshot_date": validation_date,
            "validation_method": validation_method,
            "segment_type": segment_type,
            "segment": segment,
            "sample_count": len(indices),
            "mape": float(percentage_errors.mean()),
            "median_ape": float(np.median(percentage_errors)),
            "p90_absolute_log_error": float(np.quantile(log_errors, 0.9)),
        }

    rows = [summarize(np.arange(len(actual)), "overall", "All listings")]
    segments = [
        ("brand", actual.brand.fillna("Unknown").astype(str)),
        (
            "asking_price",
            pd.cut(
                actual.price,
                bins=[0, 500, 1000, 2000, np.inf],
                labels=["<= $500", "$500-$1,000", "$1,000-$2,000", "> $2,000"],
                include_lowest=True,
            ).astype("object").fillna("Unknown"),
        ),
        (
            "spec_completeness",
            actual[SPEC_FIELDS].notna().all(axis=1).map(
                {True: "All key specs parsed", False: "Some key specs missing"}
            ),
        ),
    ]
    for segment_type, values in segments:
        for segment in values.unique():
            indices = np.flatnonzero(values.to_numpy() == segment)
            rows.append(summarize(indices, segment_type, str(segment)))
    return rows


def _write_metrics(con, rows):
    pd.DataFrame(
        rows,
        columns=[
            "snapshot_date",
            "validation_method",
            "segment_type",
            "segment",
            "sample_count",
            "mape",
            "median_ape",
            "p90_absolute_log_error",
        ],
    ).to_sql("model_metrics", con, if_exists="replace", index=False)


def score_listings(clean, con):
    eligible = clean.dropna(subset=["item_id", "ram_gb", "brand", "cpu"]).copy()
    eligible = eligible[eligible.price.gt(0)]
    if eligible.empty:
        raise ValueError("No clean listings have the required model features.")

    latest_date = eligible.snapshot_date.max()
    latest = eligible[eligible.snapshot_date == latest_date].copy()
    prior = eligible[eligible.snapshot_date < latest_date].copy()
    distinct_prior_items = prior.item_id.nunique()
    error_band_log = None

    if not prior.empty and distinct_prior_items >= 2:
        validation = latest[~latest.item_id.isin(prior.item_id)].copy()
        validation_method = "temporal_holdout_new_items"
        if validation.empty:
            validation = latest.copy()
            validation_method = "temporal_holdout_repeat_items"

        model = new_model()
        train_features = features_for(prior)
        category_levels = {
            column: train_features[column].cat.categories
            for column in CATEGORICAL_FIELDS
        }
        model.fit(train_features, np.log(prior.price))
        predictions = np.exp(
            model.predict(features_for(latest, category_levels))
        )
        validation_predictions = np.exp(
            model.predict(features_for(validation, category_levels))
        )
        rows = metric_rows(
            validation,
            validation_predictions,
            validation_method,
            latest_date,
        )
        overall = next(row for row in rows if row["segment_type"] == "overall")
        if (
            validation_method == "temporal_holdout_new_items"
            and overall["sample_count"] >= 20
        ):
            error_band_log = overall["p90_absolute_log_error"]
    else:
        group_count = eligible.item_id.nunique()
        if group_count < 2:
            raise ValueError(
                "At least two eligible listing IDs are required for grouped validation."
            )
        folds = min(5, group_count)
        validation_method = "item_grouped_cv_same_period"
        predictions = np.exp(
            cross_val_predict(
                new_model(),
                features_for(eligible),
                np.log(eligible.price),
                groups=eligible.item_id,
                cv=GroupKFold(n_splits=folds),
            )
        )
        validation = eligible
        rows = metric_rows(
            validation,
            predictions,
            validation_method,
            latest_date,
        )
        latest_positions = np.flatnonzero(
            eligible.snapshot_date.to_numpy() == latest_date
        )
        latest = eligible.iloc[latest_positions].copy()
        predictions = predictions[latest_positions]

    latest["estimated_asking_price"] = predictions
    latest["price_gap_pct"] = (
        latest.estimated_asking_price - latest.price
    ) / latest.estimated_asking_price
    latest["estimate_low"] = (
        latest.estimated_asking_price * np.exp(-error_band_log)
        if error_band_log is not None
        else np.nan
    )
    latest["estimate_high"] = (
        latest.estimated_asking_price * np.exp(error_band_log)
        if error_band_log is not None
        else np.nan
    )
    latest.to_sql("scored_listings", con, if_exists="replace", index=False)
    _write_metrics(con, rows)

    overall = next(row for row in rows if row["segment_type"] == "overall")
    print(
        f"Validation: {validation_method} | "
        f"{overall['sample_count']} listings | "
        f"MAPE: {overall['mape']:.1%} | "
        f"median absolute percentage error: {overall['median_ape']:.1%}"
    )
    print(f"Scored {len(latest)} of {len(clean)} clean rows in snapshot {latest_date}")
    if error_band_log is None:
        print("No empirical estimate range shown; at least 20 later new listings are needed.")
    else:
        print(
            "Estimate ranges use the 90th percentile of held-out absolute log errors; "
            "they are not guaranteed confidence intervals."
        )


def main():
    with closing(sqlite3.connect(DB)) as con:
        clean = pd.read_sql("SELECT * FROM clean_listings", con)
        if clean.empty:
            raise SystemExit("No clean listings found. Run collector.py and clean.py first.")
        with con:
            score_listings(clean, con)


if __name__ == "__main__":
    main()

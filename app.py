import sqlite3
from contextlib import closing
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Laptop Listing Research", layout="wide")

with closing(sqlite3.connect("data/prices.db")) as con:
    scored = pd.read_sql("SELECT * FROM scored_listings", con)
    clean = pd.read_sql("SELECT * FROM clean_listings", con)
    metrics = pd.read_sql("SELECT * FROM model_metrics", con)

if scored.empty:
    st.error("No listings have model estimates yet. Run clean.py and model.py first.")
    st.stop()

latest_snapshot = scored.snapshot_date.max()
latest = scored[scored.snapshot_date == latest_snapshot].copy()
latest_clean_count = int((clean.snapshot_date == latest_snapshot).sum())
overall_metrics = metrics[
    (metrics.snapshot_date == latest_snapshot)
    & (metrics.segment_type == "overall")
]

st.title("Laptop Listing Research")
st.caption(f"Latest captured snapshot: {latest_snapshot}. Availability can change after capture.")
snapshot_age = (date.today() - date.fromisoformat(latest_snapshot)).days
if snapshot_age > 2:
    st.warning(f"Data is {snapshot_age} days old. Treat listing availability and estimates as stale.")
st.warning(
    "Model estimates are based on advertised asking prices, not completed sales. "
    "Titles can omit defects, locks, battery condition, or missing accessories. "
    "Open and verify every listing before treating it as a candidate."
)
if overall_metrics.empty:
    st.warning("No validation metrics are available for this snapshot.")
else:
    validation = overall_metrics.iloc[0]
    method_labels = {
        "item_grouped_cv_same_period": "Item-grouped cross-validation within this collection period",
        "temporal_holdout_new_items": "Later-snapshot holdout of listings not seen in earlier snapshots",
        "temporal_holdout_repeat_items": "Later-snapshot holdout (listings also seen earlier)",
    }
    method_label = method_labels.get(
        validation.validation_method,
        str(validation.validation_method),
    )
    st.info(
        f"Validation: {method_label}. "
        f"MAPE {validation.mape:.1%}; median absolute percentage error "
        f"{validation.median_ape:.1%} on {int(validation.sample_count)} listings. "
        "These metrics describe asking-price estimates in this collected data; "
        "they are not proof of future accuracy or transaction value."
    )

brand = st.multiselect("Brand", sorted(latest.brand.dropna().unique()))
max_price = st.slider("Maximum asking price", 100, 3000, 1000)
min_gap = st.slider("Minimum gap below model estimate (%)", 0, 70, 20) / 100
results = latest[(latest.price <= max_price) & (latest.price_gap_pct >= min_gap)].copy()
if brand:
    results = results[results.brand.isin(brand)]

results["review_status"] = results.price_gap_pct.apply(
    lambda gap: "Large gap: investigate first" if gap > 0.5 else "Candidate: inspect listing"
)
spec_fields = ["ram_gb", "storage_gb", "cpu_model", "screen_in"]
spec_labels = {
    "ram_gb": "RAM",
    "storage_gb": "storage",
    "cpu_model": "CPU model",
    "screen_in": "screen size",
}
results["missing_specs"] = results[spec_fields].apply(
    lambda row: ", ".join(spec_labels[column] for column in spec_fields if pd.isna(row[column])) or "None",
    axis=1,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clean listings in latest snapshot", latest_clean_count)
c2.metric("Candidates to inspect", len(results))
c3.metric(
    "Validation MAPE",
    f"{overall_metrics.iloc[0].mape:.1%}" if not overall_metrics.empty else "Unavailable",
)
median_gap_pct = round(latest.price_gap_pct.median() * 100)
c4.metric("Median gap vs estimate", f"{median_gap_pct}%")
st.caption(f"Not scored because required specs were missing: {max(0, latest_clean_count - len(latest))} listings.")

if (
    not overall_metrics.empty
    and overall_metrics.iloc[0].validation_method == "temporal_holdout_new_items"
    and overall_metrics.iloc[0].sample_count >= 20
):
    st.caption(
        "Estimate ranges are based on the 90th percentile of absolute log errors "
        "in the later-snapshot holdout. They are empirical error bands, not "
        "guaranteed confidence intervals."
    )
else:
    st.caption(
        "No empirical error range is shown. Collect later snapshots to evaluate "
        "future-period performance; a point estimate alone should not be treated as precise."
    )

segment_metrics = metrics[
    (metrics.snapshot_date == latest_snapshot)
    & (metrics.segment_type != "overall")
]
if not segment_metrics.empty:
    st.subheader("Validation error by listing group")
    st.caption("Small groups have unstable error estimates; sample counts are shown.")
    st.dataframe(
        segment_metrics.rename(
            columns={
                "segment_type": "Group",
                "segment": "Segment",
                "sample_count": "Listings",
                "mape": "MAPE",
                "median_ape": "Median absolute percentage error",
            }
        )[["Group", "Segment", "Listings", "MAPE", "Median absolute percentage error"]],
        column_config={
            "MAPE": st.column_config.NumberColumn(format="percent"),
            "Median absolute percentage error": st.column_config.NumberColumn(format="percent"),
        },
        hide_index=True,
    )

display_columns = [
    "title",
    "price",
    "estimated_asking_price",
    "price_gap_pct",
    "review_status",
    "missing_specs",
    "condition",
    "seller_feedback",
    "snapshot_date",
    "url",
]
if latest[["estimate_low", "estimate_high"]].notna().all(axis=1).any():
    results["historical_error_low"] = results.estimate_low
    results["historical_error_high"] = results.estimate_high
    display_columns[3:3] = ["historical_error_low", "historical_error_high"]

st.dataframe(
    results.sort_values("price_gap_pct", ascending=False)[display_columns],
    column_config={
        "price": st.column_config.NumberColumn("Asking price", format="$%.0f"),
        "estimated_asking_price": st.column_config.NumberColumn("Model estimate", format="$%.0f"),
        "historical_error_low": st.column_config.NumberColumn("Historical error band low", format="$%.0f"),
        "historical_error_high": st.column_config.NumberColumn("Historical error band high", format="$%.0f"),
        "price_gap_pct": st.column_config.NumberColumn("Gap vs estimate", format="percent"),
        "url": st.column_config.LinkColumn("Open listing"),
    },
    hide_index=True,
)

trend = clean.groupby(["snapshot_date", "brand"]).price.median().reset_index()
if trend.snapshot_date.nunique() < 2:
    st.info("Price history will appear after at least two collection snapshots.")
else:
    st.plotly_chart(
        px.line(trend, x="snapshot_date", y="price", color="brand",
                title="Median observed asking price over time"),
        use_container_width=True,
    )
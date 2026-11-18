from __future__ import annotations

import numpy as np
import pandas as pd

from citibike_pipeline.stage3_models import (
    FEATURE_NAMES, build_numeric_features, canonical_hour_utc, demand_metrics,
)


def test_ambiguous_fall_hour_is_not_mixed() -> None:
    values = pd.Series(pd.to_datetime(["2025-11-02 01:00:00", "2025-11-02 02:00:00"]))
    converted = canonical_hour_utc(values)
    assert pd.isna(converted.iloc[0])
    assert str(converted.iloc[1]) == "2025-11-02 07:00:00+00:00"


def test_fixed_date_feature_builder_has_no_demand_lags() -> None:
    row = {
        "local_hour": 8, "iso_day_of_week": 4, "panel_month": 11,
        "profile_out": 3.0, "profile_in": 2.0,
        "temperature_2m_c": 10.0, "apparent_temperature_c": 9.0,
        "relative_humidity_2m_pct": 70.0, "precipitation_mm": 0.0,
        "snowfall_cm": 0.0, "wind_speed_10m_max_ms": 3.0,
        "wind_gusts_10m_max_ms": 5.0, "cloud_cover_pct": 40.0,
        "is_weekend": False, "is_workday": False,
        "is_federal_holiday": True, "is_thanksgiving": True,
        "is_thanksgiving_eve": False, "is_day_after_thanksgiving": False,
        "days_from_thanksgiving": 0,
    }
    result = build_numeric_features(pd.DataFrame([row]))
    assert result.shape == (1, len(FEATURE_NAMES))
    assert not any("lag" in name or "previous" in name for name in FEATURE_NAMES)


def test_wape_and_net_metrics() -> None:
    result = demand_metrics(
        np.array([2.0, 4.0]), np.array([1.0, 3.0]),
        np.array([3.0, 3.0]), np.array([1.0, 2.0]),
    )
    assert result["ride_out"]["mae"] == 1.0
    assert np.isclose(result["ride_out"]["wape"], 2 / 6)
    assert result["net_outflow"]["mae"] == 0.5

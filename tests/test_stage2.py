from datetime import date

from scripts.build_model_features import build_calendar, thanksgiving
from scripts.download_trip_history import iter_months


def test_month_iteration_is_inclusive():
    assert list(iter_months("2023-11", "2024-02")) == ["202311", "202312", "202401", "202402"]


def test_thanksgiving_dates():
    assert thanksgiving(2023) == date(2023, 11, 23)
    assert thanksgiving(2024) == date(2024, 11, 28)
    assert thanksgiving(2025) == date(2025, 11, 27)
    assert thanksgiving(2026) == date(2026, 11, 26)


def test_dst_calendar_removes_spring_gap_and_flags_fall_repeat():
    _, hours = build_calendar("2024-03-10", "2024-11-03")
    spring = hours[hours["hour_local"].eq("2024-03-10 02:00:00")]
    fall = hours[hours["hour_local"].eq("2024-11-03 01:00:00")]
    assert spring.empty
    assert len(fall) == 1
    assert bool(fall.iloc[0]["dst_ambiguous_collapsed"])

from datetime import datetime, timezone

from citibike_pipeline.gbfs import FetchResult, station_status_frame, validate_snapshot


def test_time_error_and_operating_flags():
    now = datetime(2026, 11, 18, 0, 0, tzinfo=timezone.utc)
    payload = {
        "last_updated": int(now.timestamp()) - 30,
        "data": {"stations": [{
            "station_id": "abc", "num_bikes_available": 2, "num_docks_available": 3,
            "num_bikes_disabled": 0, "num_docks_disabled": 0,
            "is_installed": 1, "is_renting": 1, "is_returning": 1,
            "last_reported": 86400,
        }]},
    }
    frame = station_status_frame(FetchResult(payload, now, now, "https://example.test"))
    assert bool(frame.loc[0, "operating"])
    assert bool(frame.loc[0, "station_time_error"])
    assert not bool(frame.loc[0, "feed_time_error"])


def test_validation_detects_id_mismatch():
    import pandas as pd

    info = pd.DataFrame({"station_id": pd.Series(["a"], dtype="string")})
    status = pd.DataFrame({
        "station_id": pd.Series(["b"], dtype="string"),
        "num_bikes_available": [1], "num_docks_available": [2],
        "operating": [True], "feed_time_error": [False], "station_time_error": [False],
    })
    report = validate_snapshot(info, status)
    assert report["status_ids_missing_from_info"] == 1
    assert report["info_ids_missing_from_status"] == 1


# Time splits and feature availability

## Initial comparison split

- `train_initial`: 2023-01-01 through 2024-09-30.
- `validation_regular`: 2024-10-01 through 2025-06-30, excluding the seven-day Thanksgiving holdout window.
- `test_regular`: 2025-07-01 through the latest published trip month, excluding Thanksgiving holdout windows.
- `thanksgiving_holdout`: Thanksgiving day plus three days before and after in 2024 and 2025.

The split is chronological. It is not a random row split. Model selection must not use a holdout year's Thanksgiving results before scoring that scenario. After the experiment design is frozen, later rolling-origin fits may train on earlier years' Thanksgiving observations, but never on the target year's future results.

## Feature availability

Features safe for a fixed-date forecast issued well in advance:

- calendar fields, federal-holiday indicators, and Thanksgiving-relative dates;
- station coordinates and station attributes known at the issue time;
- historical demand summaries calculated only from observations before the issue time;
- a weather forecast vintage that existed at the issue time.

Features restricted to a real-time rolling forecast:

- lagged ride counts and trailing means;
- current or lagged GBFS bikes/docks;
- recent neighbouring-station demand.

Historical Open-Meteo archive weather is valid for training and explanatory analysis. It is not a valid input for the 2026-11-26 future prediction unless replaced by a forecast vintage retrieved before the forecast target. Inventory empty/full indicators are outcome labels or real-time state features, not long-ahead fixed-date predictors.

The Citi Bike trip timestamps do not include a UTC offset. The repeated fall-back civil hour is collapsed and flagged as `dst_ambiguous_collapsed`; the nonexistent spring-forward hour is removed. Inventory and weather retain UTC timestamps in their source tables before being collapsed to the same civil-hour convention for joining.


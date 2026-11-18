clear all
set more off

* Run from the repository root. Large yearly files remain local and are not in Git.
local years 2023 2024 2025 2026
local first 1
foreach y of local years {
    import delimited using "data/processed/exports/stata_csv/citibike_station_hour_`y'.csv", clear varnames(1) encoding(utf8)
    gen double hour_dt = clock(hour_local, "YMDhms")
    format hour_dt %tc
    foreach v in zero_filled_active_hour dst_ambiguous_collapsed is_federal_holiday is_weekend is_workday is_thanksgiving is_thanksgiving_eve is_day_after_thanksgiving is_thanksgiving_window_3d empty_any full_any {
        capture confirm string variable `v'
        if !_rc {
            tempvar numeric_flag
            gen byte `numeric_flag' = cond(lower(`v') == "true", 1, cond(lower(`v') == "false", 0, .))
            drop `v'
            rename `numeric_flag' `v'
        }
    }
    compress
    if `first' {
        save "data/processed/exports/citibike_station_hour_panel.dta", replace
        local first 0
    }
    else {
        append using "data/processed/exports/citibike_station_hour_panel.dta"
        save "data/processed/exports/citibike_station_hour_panel.dta", replace
    }
}

isid station_id hour_dt
describe
summarize ride_out ride_in net_outflow

clear all
set more off

* Run from the repository root. Large yearly files remain local and are not in Git.
local years 2023 2024 2025 2026
local first 1
foreach y of local years {
    import delimited using "data/processed/exports/stata_csv/citibike_station_hour_`y'.csv", clear varnames(1) encoding(utf8)
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

isid station_id hour_local
describe
summarize ride_out ride_in net_outflow

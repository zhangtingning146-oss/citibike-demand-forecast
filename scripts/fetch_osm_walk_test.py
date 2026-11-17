from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Manhattan walk network and test subway access")
    parser.add_argument("--gbfs-info", type=Path, required=True)
    parser.add_argument("--entrances", type=Path, default=Path("data/raw/mta/MTA_Subway_Entrances_and_Exits_2024.csv"))
    parser.add_argument("--sample-stations", type=int, default=12)
    args = parser.parse_args()

    raw = Path("data/raw/osm")
    raw.mkdir(parents=True, exist_ok=True)
    graph_path = raw / "manhattan_walk.graphml"
    boundary_path = raw / "manhattan_boundary.geojson"
    place = "Manhattan, New York County, New York, USA"

    boundary = ox.geocode_to_gdf(place).to_crs(4326)
    boundary.to_file(boundary_path, driver="GeoJSON")
    if graph_path.exists():
        graph = ox.load_graphml(graph_path)
    else:
        graph = ox.graph_from_polygon(boundary.geometry.iloc[0], network_type="walk", simplify=True)
        ox.save_graphml(graph, graph_path)

    info = pd.read_parquet(args.gbfs_info)
    stations = gpd.GeoDataFrame(
        info, geometry=gpd.points_from_xy(info["lon"], info["lat"]), crs=4326
    )
    stations = stations[stations.geometry.within(boundary.geometry.iloc[0])].copy()
    manhattan_station_count = len(stations)
    stations = stations.sort_values("capacity", ascending=False).head(args.sample_stations)

    entrances = pd.read_csv(args.entrances, dtype={"complex_id": "string", "station_id": "string"})
    entrances = entrances[
        entrances["borough"].astype(str).str.upper().eq("M")
        & entrances["entry_allowed"].astype(str).str.upper().eq("YES")
    ].dropna(subset=["entrance_longitude", "entrance_latitude"]).copy()

    entrance_nodes = ox.distance.nearest_nodes(
        graph, X=entrances["entrance_longitude"].to_numpy(), Y=entrances["entrance_latitude"].to_numpy()
    )
    entrances["walk_node"] = entrance_nodes
    station_nodes = ox.distance.nearest_nodes(
        graph, X=stations["lon"].to_numpy(), Y=stations["lat"].to_numpy()
    )
    stations["walk_node"] = station_nodes

    # Route only to the 20 closest entrances by great-circle distance, then keep the best three walk paths.
    output: list[dict] = []
    e_lat = entrances["entrance_latitude"].to_numpy()
    e_lon = entrances["entrance_longitude"].to_numpy()
    for _, station in stations.iterrows():
        straight = ox.distance.great_circle(station["lat"], station["lon"], e_lat, e_lon)
        candidate_idx = np.argsort(straight)[:20]
        routed: list[tuple[float, int]] = []
        for idx in candidate_idx:
            try:
                distance = nx.shortest_path_length(
                    graph, station["walk_node"], entrances.iloc[idx]["walk_node"], weight="length"
                )
                routed.append((float(distance), int(idx)))
            except nx.NetworkXNoPath:
                continue
        for rank, (distance, idx) in enumerate(sorted(routed)[:3], start=1):
            entrance = entrances.iloc[idx]
            output.append({
                "gbfs_station_id": station["station_id"],
                "citibike_short_name": station["short_name"],
                "citibike_station_name": station["name"],
                "rank": rank,
                "walk_distance_m": round(distance, 1),
                "subway_complex_id": entrance["complex_id"],
                "subway_stop_name": entrance["stop_name"],
                "subway_routes": entrance["daytime_routes"],
                "entrance_type": entrance["entrance_type"],
                "entrance_latitude": entrance["entrance_latitude"],
                "entrance_longitude": entrance["entrance_longitude"],
            })

    out = Path("data/processed/citibike_subway_walk_sample.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(out, index=False)
    print(f"Manhattan stations in current GBFS: {manhattan_station_count:,}")
    print(f"Entry-allowed Manhattan subway entrances: {len(entrances):,}")
    print(f"Walking-distance rows written: {len(output):,}")


if __name__ == "__main__":
    main()


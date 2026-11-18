from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


WEATHER_COLUMNS = [
    "temperature_2m_c", "apparent_temperature_c", "relative_humidity_2m_pct",
    "precipitation_mm", "snowfall_cm", "wind_speed_10m_max_ms",
    "wind_gusts_10m_max_ms", "cloud_cover_pct",
]
FEATURE_NAMES = [
    "log_profile_out", "log_profile_in",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    *WEATHER_COLUMNS,
    "is_weekend", "is_workday", "is_federal_holiday", "is_thanksgiving",
    "is_thanksgiving_eve", "is_day_after_thanksgiving", "days_from_thanksgiving_clipped",
]
BASIC_FEATURES = {
    "log_profile_out", "log_profile_in", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "is_workday",
}


def seed_everything(seed: int = 20261118) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def canonical_hour_utc(hour_local: pd.Series) -> pd.Series:
    values = pd.to_datetime(hour_local)
    return values.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")


def build_numeric_features(frame: pd.DataFrame) -> np.ndarray:
    hour = frame["local_hour"].to_numpy(dtype=np.float32)
    dow = frame["iso_day_of_week"].to_numpy(dtype=np.float32)
    month = frame["panel_month"].to_numpy(dtype=np.float32)
    features = {
        "log_profile_out": np.log1p(frame["profile_out"].to_numpy(dtype=np.float32)),
        "log_profile_in": np.log1p(frame["profile_in"].to_numpy(dtype=np.float32)),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * (dow - 1) / 7),
        "dow_cos": np.cos(2 * np.pi * (dow - 1) / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
    }
    for column in WEATHER_COLUMNS:
        features[column] = frame[column].fillna(0).to_numpy(dtype=np.float32)
    for column in [
        "is_weekend", "is_workday", "is_federal_holiday", "is_thanksgiving",
        "is_thanksgiving_eve", "is_day_after_thanksgiving",
    ]:
        features[column] = frame[column].fillna(False).astype(np.float32).to_numpy()
    features["days_from_thanksgiving_clipped"] = (
        frame["days_from_thanksgiving"].fillna(30).clip(-7, 7).to_numpy(dtype=np.float32) / 7
    )
    return np.column_stack([features[name] for name in FEATURE_NAMES]).astype(np.float32)


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray | None = None) -> "Standardizer":
        subset = values if mask is None else values[mask]
        mean = np.nanmean(subset, axis=0).astype(np.float32)
        scale = np.nanstd(subset, axis=0).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.nan_to_num((values - self.mean) / self.scale, nan=0.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "Standardizer":
        return cls(np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["scale"], dtype=np.float32))


class DemandMLP(nn.Module):
    def __init__(self, n_stations: int, n_features: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(n_stations, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(n_features + embedding_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Softplus(),
        )

    def forward(self, numeric: torch.Tensor, station: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([numeric, self.station_embedding(station)], dim=-1))


class DemandGRU(nn.Module):
    def __init__(self, n_stations: int, n_features: int, embedding_dim: int = 16, hidden_size: int = 64) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(n_stations, embedding_dim)
        self.gru = nn.GRU(n_features + embedding_dim, hidden_size, batch_first=True, num_layers=1)
        self.head = nn.Sequential(nn.Linear(hidden_size, 2), nn.Softplus())

    def forward(self, numeric: torch.Tensor, station: torch.Tensor) -> torch.Tensor:
        embedding = self.station_embedding(station).unsqueeze(1).expand(-1, numeric.shape[1], -1)
        output, _ = self.gru(torch.cat([numeric, embedding], dim=-1))
        return self.head(output)


def poisson_deviance_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.clamp_min(1e-4)
    return torch.mean(prediction - target * torch.log(prediction))


def train_mlp(
    x: np.ndarray,
    station: np.ndarray,
    y: np.ndarray,
    n_stations: int,
    output: Path,
    epochs: int = 4,
    batch_size: int = 8192,
    seed: int = 20261118,
) -> tuple[DemandMLP, list[dict]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemandMLP(n_stations, x.shape[1]).to(device)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(station.astype(np.int64)), torch.from_numpy(y.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for numeric, station_id, target in loader:
            numeric = numeric.to(device, non_blocking=True)
            station_id = station_id.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(numeric, station_id)
            loss = poisson_deviance_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(numeric)
            total_rows += len(numeric)
        record = {"epoch": epoch, "poisson_objective": total_loss / total_rows, "device": str(device)}
        history.append(record)
        print(f"MLP epoch {epoch}: {record['poisson_objective']:.6f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    return model, history


def train_gru(
    x: np.ndarray,
    station: np.ndarray,
    y: np.ndarray,
    n_stations: int,
    output: Path,
    epochs: int = 3,
    batch_size: int = 512,
    seed: int = 20261118,
) -> tuple[DemandGRU, list[dict]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemandGRU(n_stations, x.shape[2]).to(device)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(station.astype(np.int64)), torch.from_numpy(y.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for numeric, station_id, target in loader:
            numeric = numeric.to(device, non_blocking=True)
            station_id = station_id.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(numeric, station_id)
            loss = poisson_deviance_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * numeric.shape[0] * numeric.shape[1]
            total_rows += numeric.shape[0] * numeric.shape[1]
        record = {"epoch": epoch, "poisson_objective": total_loss / total_rows, "device": str(device)}
        history.append(record)
        print(f"GRU epoch {epoch}: {record['poisson_objective']:.6f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    return model, history


@torch.inference_mode()
def predict_mlp(model: DemandMLP, x: np.ndarray, station: np.ndarray, batch_size: int = 32768) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        numeric = torch.from_numpy(x[start:start + batch_size]).to(device)
        station_id = torch.from_numpy(station[start:start + batch_size].astype(np.int64)).to(device)
        outputs.append(model(numeric, station_id).cpu().numpy())
    return np.concatenate(outputs)


@torch.inference_mode()
def predict_gru(model: DemandGRU, x: np.ndarray, station: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        numeric = torch.from_numpy(x[start:start + batch_size]).to(device)
        station_id = torch.from_numpy(station[start:start + batch_size].astype(np.int64)).to(device)
        outputs.append(model(numeric, station_id).cpu().numpy())
    return np.concatenate(outputs)


def save_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def demand_metrics(actual_out: np.ndarray, actual_in: np.ndarray, pred_out: np.ndarray, pred_in: np.ndarray) -> dict:
    def one(actual: np.ndarray, pred: np.ndarray) -> dict:
        error = pred - actual
        return {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "wape": float(np.sum(np.abs(error)) / max(np.sum(np.abs(actual)), 1.0)),
        }
    result = {"ride_out": one(actual_out, pred_out), "ride_in": one(actual_in, pred_in)}
    result["net_outflow"] = one(actual_out - actual_in, pred_out - pred_in)
    return result


"""Room 5 ramp-up detection and duration classification utilities.

This module focuses only on Room 5 (library served by an AHU) and uses
chilled water energy and off-coil air temperature to:
1) detect ramp-up intervals
2) derive event-level features
3) train a simple classifier to estimate ramp-up duration classes
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, TypedDict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RampUpInterval:
    """Represents a single ramp-up interval."""

    start_idx: int
    end_idx: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_minutes: float


def _prepare_room5_frame(
    df: pd.DataFrame,
    timestamp_col: str,
    temp_col: str,
    setpoint_col: str,
    energy_col: str,
) -> pd.DataFrame:
    frame = df[[timestamp_col, temp_col, setpoint_col, energy_col]].dropna().copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
    frame = frame.sort_values(timestamp_col).reset_index(drop=True)
    return frame


def detect_ramp_up_intervals(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    temp_col: str = "offcoil_air_temp",
    setpoint_col: str = "offcoil_temp_setpoint",
    energy_col: str = "chilled_water_energy",
    temp_drop_start: float = 0.5,
    end_drop_threshold: float = 0.05,
    energy_min_active: float = 0.0,
    min_duration_minutes: float = 1.0,
) -> List[RampUpInterval]:
    """Detect ramp-up intervals for Room 5.

    Start: when chilled water energy becomes active (> energy_min_active) OR when
    off-coil temperature drops by more than temp_drop_start in one step.
    End: when temperature drop slows (>= -end_drop_threshold) AND chilled water
    energy is no longer active.
    """

    data = _prepare_room5_frame(df, timestamp_col, temp_col, setpoint_col, energy_col)
    if data.empty:
        return []

    temp = data[temp_col].astype(float)
    setpoint = data[setpoint_col].astype(float)
    energy = data[energy_col].astype(float)
    temp_diff = temp.diff().fillna(0.0)
    step_minutes = (
        data[timestamp_col].diff().dt.total_seconds().dropna().median() / 60.0
        if len(data) > 1
        else 1.0
    )
    if not np.isfinite(step_minutes) or step_minutes <= 0:
        step_minutes = 1.0

    intervals: List[RampUpInterval] = []
    i = 1

    while i < len(data):
        dropping_start = temp_diff.iloc[i] <= -temp_drop_start
        energy_active = energy.iloc[i] > energy_min_active

        if energy_active or dropping_start:
            start_idx = i
            end_idx = start_idx
            j = end_idx + 1
            while j < len(data):
                ongoing_drop = temp_diff.iloc[j] < -end_drop_threshold
                energy_still_active = energy.iloc[j] > energy_min_active
                if ongoing_drop or energy_still_active:
                    end_idx = j
                    j += 1
                    continue
                break

            duration_minutes = (
                data[timestamp_col].iloc[end_idx] - data[timestamp_col].iloc[start_idx]
            ).total_seconds() / 60.0 + step_minutes

            if duration_minutes >= min_duration_minutes:
                intervals.append(
                    RampUpInterval(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        start_time=data[timestamp_col].iloc[start_idx],
                        end_time=data[timestamp_col].iloc[end_idx],
                        duration_minutes=duration_minutes,
                    )
                )
            i = end_idx + 1
        else:
            i += 1

    return intervals


def intervals_to_feature_frame(
    df: pd.DataFrame,
    intervals: Sequence[RampUpInterval],
    *,
    timestamp_col: str = "timestamp",
    temp_col: str = "offcoil_air_temp",
    setpoint_col: str = "offcoil_temp_setpoint",
    energy_col: str = "chilled_water_energy",
) -> pd.DataFrame:
    """Convert intervals to an event-level feature frame."""

    data = _prepare_room5_frame(df, timestamp_col, temp_col, setpoint_col, energy_col)

    if not intervals:
        return pd.DataFrame(
            columns=[
                "duration_minutes",
                "temp_gap_start",
                "temp_drop_rate",
                "mean_energy",
                "energy_slope",
            ]
        )

    rows = []
    for interval in intervals:
        segment = data.iloc[interval.start_idx : interval.end_idx + 1]
        temp_start = float(segment[temp_col].iloc[0])
        setpoint_start = float(segment[setpoint_col].iloc[0])
        temp_end = float(segment[temp_col].iloc[-1])
        energy_values = segment[energy_col].astype(float)
        if len(energy_values) < 2:
            energy_slope = 0.0
        else:
            energy_slope = (energy_values.iloc[-1] - energy_values.iloc[0]) / max(
                interval.duration_minutes, 1.0
            )

        rows.append(
            {
                "duration_minutes": interval.duration_minutes,
                "temp_gap_start": temp_start - setpoint_start,
                "temp_drop_rate": (temp_start - temp_end) / max(interval.duration_minutes, 1.0),
                "mean_energy": energy_values.mean(),
                "energy_slope": energy_slope,
            }
        )

    return pd.DataFrame(rows)


def train_duration_classifier(
    events_df: pd.DataFrame,
    *,
    duration_threshold: float = 20.0,
    test_size: float = 0.25,
    random_state: int = 0,
) -> tuple[Pipeline, float]:
    """Train a classifier that estimates whether ramp-up duration is long/short."""

    if events_df.empty:
        raise ValueError("No ramp-up events available for training.")

    events_df = events_df.copy()
    events_df["target_long"] = (events_df["duration_minutes"] >= duration_threshold).astype(int)

    if events_df["target_long"].nunique() < 2:
        raise ValueError("Need both long and short events to train the classifier.")

    feature_cols = ["temp_gap_start", "temp_drop_rate", "mean_energy", "energy_slope"]
    X = events_df[feature_cols]
    y = events_df["target_long"]

    stratify_labels = y
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=stratify_labels, random_state=random_state
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=None, random_state=random_state
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=300, class_weight="balanced"),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    f1 = f1_score(y_val, predictions)
    return model, f1


class RampUpModelResult(TypedDict):
    intervals: List[RampUpInterval]
    features: pd.DataFrame
    model: Pipeline
    f1: float


def build_room5_ramp_up_model(
    df: pd.DataFrame,
    *,
    duration_threshold: float = 20.0,
    detection_kwargs: Mapping[str, Any] | None = None,
) -> RampUpModelResult:
    """End-to-end helper for Room 5 ramp-up modeling."""

    detection_kwargs = dict(detection_kwargs) if detection_kwargs else {}
    timestamp_col = detection_kwargs.get("timestamp_col", "timestamp")
    temp_col = detection_kwargs.get("temp_col", "offcoil_air_temp")
    setpoint_col = detection_kwargs.get("setpoint_col", "offcoil_temp_setpoint")
    energy_col = detection_kwargs.get("energy_col", "chilled_water_energy")

    intervals = detect_ramp_up_intervals(df, **detection_kwargs)
    feature_frame = intervals_to_feature_frame(
        df,
        intervals,
        timestamp_col=timestamp_col,
        temp_col=temp_col,
        setpoint_col=setpoint_col,
        energy_col=energy_col,
    )
    model, f1 = train_duration_classifier(
        feature_frame, duration_threshold=duration_threshold
    )

    return {"intervals": intervals, "features": feature_frame, "model": model, "f1": f1}

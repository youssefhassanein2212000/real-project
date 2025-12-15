import unittest

import numpy as np
import pandas as pd

from room5_rampup import (
    detect_ramp_up_intervals,
    intervals_to_feature_frame,
    train_duration_classifier,
    detect_ramp_up_intervals_room5,
    keep_first_ramp_per_day_intervals,
    build_ramp_events_df,
    ramp_duration_lodo_cv,
    train_duration_model,
)


def _build_synthetic_room5_data() -> pd.DataFrame:
    timestamps = []
    temps = []
    setpoints = []
    energies = []

    start = pd.Timestamp("2024-01-01 00:00:00")
    setpoint_value = 23.0

    def append_segment(temp_values, energy_values):
        for temp, energy in zip(temp_values, energy_values):
            timestamps.append(start + pd.Timedelta(minutes=len(timestamps)))
            temps.append(temp)
            setpoints.append(setpoint_value)
            energies.append(energy)

    # Baseline before first ramp
    append_segment([26.0] * 5, [0.0] * 5)

    # Ramp 1: quick drop to setpoint over 10 minutes
    ramp1_temps = [26.0 - 0.3 * i for i in range(1, 11)]
    append_segment(ramp1_temps, [5.0] * 10)

    # Steady state between ramps
    append_segment([23.0] * 10, [0.0] * 10)

    # Ramp 2: slower drop to setpoint over 40 minutes
    ramp2_temps = [27.0 - 0.1 * i for i in range(0, 41)]
    append_segment(ramp2_temps, [4.0] * 41)

    # Steady state after ramp
    append_segment([23.0] * 5, [0.0] * 5)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "offcoil_air_temp": temps,
            "offcoil_temp_setpoint": setpoints,
            "chilled_water_energy": energies,
        }
    )


def _build_multiday_room5_data() -> pd.DataFrame:
    rows = []
    ts = pd.Timestamp("2024-01-01 00:00:00")

    def add_ramp(day_offset: int, start_hour: int, temps: list[float]):
        nonlocal ts
        base = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_offset, hours=start_hour)
        for i, t in enumerate(temps):
            rows.append(
                {
                    "timestamp": base + pd.Timedelta(minutes=i),
                    "offcoil_air_temp": t,
                    "offcoil_temp_setpoint": 23.0,
                    "chilled_water_energy": 1.0,
                }
            )

    # Day 1 two ramps; only first should be kept
    add_ramp(0, 6, [26.0, 25.3, 24.5, 23.0, 22.5])
    add_ramp(0, 9, [25.0, 24.7, 24.0, 23.0])

    # Day 2 one ramp
    add_ramp(1, 6, [26.5, 25.8, 25.0, 23.0, 22.8])
    return pd.DataFrame(rows)


def _build_synthetic_event_frame() -> pd.DataFrame:
    # Four short events and four long events with separable features
    rows = [
        {"duration_minutes": 12, "temp_gap_start": 2.0, "temp_drop_rate": 0.35, "mean_energy": 6.0, "energy_slope": 0.08},
        {"duration_minutes": 14, "temp_gap_start": 2.5, "temp_drop_rate": 0.32, "mean_energy": 5.8, "energy_slope": 0.05},
        {"duration_minutes": 16, "temp_gap_start": 2.8, "temp_drop_rate": 0.30, "mean_energy": 5.5, "energy_slope": 0.02},
        {"duration_minutes": 18, "temp_gap_start": 3.0, "temp_drop_rate": 0.28, "mean_energy": 5.2, "energy_slope": 0.0},
        {"duration_minutes": 30, "temp_gap_start": 4.0, "temp_drop_rate": 0.14, "mean_energy": 4.0, "energy_slope": -0.02},
        {"duration_minutes": 35, "temp_gap_start": 4.3, "temp_drop_rate": 0.12, "mean_energy": 3.8, "energy_slope": -0.05},
        {"duration_minutes": 40, "temp_gap_start": 4.5, "temp_drop_rate": 0.10, "mean_energy": 3.5, "energy_slope": -0.07},
        {"duration_minutes": 45, "temp_gap_start": 4.8, "temp_drop_rate": 0.09, "mean_energy": 3.3, "energy_slope": -0.1},
    ]
    return pd.DataFrame(rows)


class Room5RampUpTests(unittest.TestCase):
    def test_detect_ramp_up_intervals(self):
        df = _build_synthetic_room5_data()
        intervals = detect_ramp_up_intervals(df, temp_drop_start=0.5, end_drop_threshold=0.05)

        self.assertEqual(len(intervals), 2)
        durations = sorted(round(i.duration_minutes) for i in intervals)
        self.assertEqual(durations, [10, 41])

    def test_training_pipeline_reaches_target_f1(self):
        event_frame = _build_synthetic_event_frame()
        model, f1 = train_duration_classifier(
            event_frame, duration_threshold=25.0, test_size=0.25, random_state=1
        )

        self.assertIsNotNone(model)
        self.assertGreaterEqual(f1, 0.8)

    def test_feature_conversion(self):
        df = _build_synthetic_room5_data()
        intervals = detect_ramp_up_intervals(df, temp_drop_start=0.5, end_drop_threshold=0.05)
        feature_frame = intervals_to_feature_frame(df, intervals)

        self.assertEqual(len(feature_frame), len(intervals))
        self.assertTrue({"duration_minutes", "mean_energy"}.issubset(feature_frame.columns))

    def test_new_detection_and_first_per_day(self):
        df = _build_multiday_room5_data()
        labels, intervals = detect_ramp_up_intervals_room5(df)
        self.assertGreaterEqual(len(intervals), 2)
        kept = keep_first_ramp_per_day_intervals(intervals, df["timestamp"])
        self.assertEqual(len(kept), 2)
        # Ensure labels mark ramp portions
        self.assertTrue(labels.sum() > 0)

    def test_event_frame_and_loocv(self):
        df = _build_multiday_room5_data()
        _, intervals = detect_ramp_up_intervals_room5(df)
        kept = keep_first_ramp_per_day_intervals(intervals, df["timestamp"])
        events = build_ramp_events_df(df, kept)
        self.assertEqual(len(events), 2)
        preds, truths, mae, med_ae = ramp_duration_lodo_cv(events)
        self.assertEqual(len(preds), len(truths))
        self.assertTrue(all(p >= 0 for p in preds))
        self.assertTrue(mae >= 0 or np.isnan(mae))
        model, cols = train_duration_model(events)
        self.assertIsNotNone(model)
        self.assertGreater(len(cols), 0)


if __name__ == "__main__":
    unittest.main()

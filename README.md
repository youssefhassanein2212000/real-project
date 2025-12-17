# real-project

Room 5 (library served by an AHU) ramp-up detection and modeling utilities.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
import pandas as pd
from room5_rampup import (
    build_room5_ramp_up_model,
)

df = pd.read_csv("ROBOD/combined_Room5_imputed.csv")
# Classification helper (long/short)
results = build_room5_ramp_up_model(df, duration_threshold=20.0)
print("Detected intervals:", len(results["intervals"]))  # multiple ramps per day supported
print("Validation F1-score:", results["f1"])

# Optional: duration regression + per-day filtering
from room5_rampup import (
    detect_ramp_up_intervals_room5,
    keep_first_ramp_per_day_intervals,
    build_ramp_events_df,
    ramp_duration_lodo_cv,
    train_duration_model,
)

labels, intervals = detect_ramp_up_intervals_room5(df, drop_threshold=0.5, max_gap=2, require_chw=False)
# keep_first_ramp_per_day_intervals is optional; omit it to allow multiple ramps per day
intervals = keep_first_ramp_per_day_intervals(intervals, df["timestamp"])
events = build_ramp_events_df(df, intervals)
preds, truths, mae, med_ae = ramp_duration_lodo_cv(events)
duration_model, duration_features = train_duration_model(events)
```

## Tests

```bash
python -m unittest discover tests
```

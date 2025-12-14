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
results = build_room5_ramp_up_model(
    df,
    duration_threshold=20.0,  # classify long vs short ramp-up durations
    detection_kwargs={
        "drop_threshold": 0.05,
        "margin": 0.05,
    },
)

print("Detected intervals:", len(results["intervals"]))
print("Validation F1-score:", results["f1"])
```

## Tests

```bash
python -m unittest discover tests
```

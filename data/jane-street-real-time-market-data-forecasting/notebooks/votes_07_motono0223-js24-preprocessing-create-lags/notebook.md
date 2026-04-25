# JS24: Preprocessing (Create Lags)

- **Author:** motono0223
- **Votes:** 691
- **Ref:** motono0223/js24-preprocessing-create-lags
- **URL:** https://www.kaggle.com/code/motono0223/js24-preprocessing-create-lags
- **Last run:** 2024-10-28 10:50:44.703000

---

# Libraries

```python
import pandas as pd
import polars as pl
import numpy as np
import gc
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import StratifiedGroupKFold
```

# Configurations

```python
class CONFIG:
    target_col = "responder_6"
    lag_cols_original = ["date_id", "symbol_id"] + [f"responder_{idx}" for idx in range(9)]
    lag_cols_rename = { f"responder_{idx}" : f"responder_{idx}_lag_1" for idx in range(9)}
    valid_ratio = 0.05
    start_dt = 1100
```

# Load training data

```python
# Use last 2 parquets
train = pl.scan_parquet(
    f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet"
).select(
    pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"),
    pl.all(),
).with_columns(
    (pl.col(CONFIG.target_col)*2).cast(pl.Int32).alias("label"),
).filter(
    pl.col("date_id").gt(CONFIG.start_dt)
)
```

# Create Lags data from training data

```python
lags = train.select(pl.col(CONFIG.lag_cols_original))
lags = lags.rename(CONFIG.lag_cols_rename)
lags = lags.with_columns(
    date_id = pl.col('date_id') + 1,  # lagged by 1 day
    )
lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last()  # pick up last record of previous date
lags
```

# Merge training data and lags data

```python
train = train.join(lags, on=["date_id", "symbol_id"],  how="left")
train
```

# Split training data and validation data

```python
len_train   = train.select(pl.col("date_id")).collect().shape[0]
valid_records = int(len_train * CONFIG.valid_ratio)
len_ofl_mdl = len_train - valid_records
last_tr_dt  = train.select(pl.col("date_id")).collect().row(len_ofl_mdl)[0]

print(f"\n len_train = {len_train}")
print(f"\n len_ofl_mdl = {len_ofl_mdl}")
print(f"\n---> Last offline train date = {last_tr_dt}\n")

training_data = train.filter(pl.col("date_id").le(last_tr_dt))
validation_data   = train.filter(pl.col("date_id").gt(last_tr_dt))
```

```python
validation_data
```

# Save data as parquets

```python
training_data.collect().\
write_parquet(
    f"training.parquet", partition_by = "date_id",
)
```

```python
validation_data.collect().\
write_parquet(
    "validation.parquet", partition_by = "date_id",
)
```
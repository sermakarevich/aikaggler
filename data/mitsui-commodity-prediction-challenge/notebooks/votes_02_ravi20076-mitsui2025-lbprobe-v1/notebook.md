# MITSUI2025|LBProbe|V1

- **Author:** Ravi Ramakrishnan
- **Votes:** 521
- **Ref:** ravi20076/mitsui2025-lbprobe-v1
- **URL:** https://www.kaggle.com/code/ravi20076/mitsui2025-lbprobe-v1
- **Last run:** 2025-07-26 13:24:28.510000

---

# **FOREWORD**

This kernel is a test to probe into the public leaderboard and ascertain that the **test set available now is the last 90 days in the training period**. Refer the data page [here](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data) where the host writes- <br>

### Dataset Description
This competition dataset consists of multiple financial time series data obtained from markets around the world. The dataset various financial instruments such as metals, futures, US stocks, and foreign exchange. Participants are challenged to develop models that predict the returns of multiple target financial time series.

### Competition Phases and Data Updates
The competition will proceed in two phases:

A model training phase with a test set of roughly three months of historical data. Because these prices are publicly available leaderboard scores during this phase are not meaningful.
A forecasting phase with a test set to be collected after submissions close. You should expect this test set to be about the same size as the test set in the first phase.
During the forecasting phase the evaluation API will serve test data from the beginning of the public set to the end of the private set.

### What I do here

I probe into the public leaderboard and check using the dummy submission as below- <br>
- Try and merge the date available in the API with the train labels
- If the date is available, then simply borrow the results from the train labels (ground truth)
- Else use the dummy submission

If all the train dates are repeated here, then my score will be an infinitely high number. Else it will match the results from the dummy submission kernel [here](https://www.kaggle.com/code/sohier/mitsui-demo-submission)

# **IMPORTS**

```python
import pandas as pd, polars as pl, numpy as np
import os
from warnings import filterwarnings 
filterwarnings("ignore")

pd.set_option(
    'display.max_rows' , 30, 
    'display.max_columns' , 35 ,
    'display.max_colwidth',  100,
    'display.precision' , 4,
    'display.float_format', '{:,.4f}'.format
) 

NUM_TARGET_COLUMNS = 424
```

# **PROBING**

```python
%%time 

train_labels = pd.read_csv(
    f"/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv"
)

sel_cols = train_labels.columns.tolist()

train_labels["date_id"] = train_labels["date_id"].astype(np.uint16)
display(train_labels.head(10))
```

```python
%%time 

import kaggle_evaluation.mitsui_inference_server
NUM_TARGET_COLUMNS = 424


def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:

    Xtest      = test.to_pandas()
    date_id    = Xtest["date_id"][0]
    test_preds = train_labels.loc[date_id, sel_cols[1:]].transpose().fillna(0).to_dict()
   
    predictions = pl.DataFrame(test_preds).select(pl.all().cast(pl.Float64))
    print(f"Captured ground truth | {date_id}")
        
    assert isinstance(predictions, (pd.DataFrame, pl.DataFrame))
    assert len(predictions) == 1
    return predictions
```

```python
%%time 

inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))
```
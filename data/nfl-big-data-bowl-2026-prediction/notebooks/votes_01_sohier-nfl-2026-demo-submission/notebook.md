# NFL 2026 Demo Submission

- **Author:** Sohier Dane
- **Votes:** 362
- **Ref:** sohier/nfl-2026-demo-submission
- **URL:** https://www.kaggle.com/code/sohier/nfl-2026-demo-submission
- **Last run:** 2025-10-24 17:23:02.170000

---

```python
"""
The evaluation API requires that you set up a server which will respond to inference requests.
We have already defined the server; you just need write the predict function.
When we evaluate your submission on the hidden test set the client defined in `nfl_gateway` will run in a different container
with direct access to the hidden test set and hand off the data timestep by timestep.
Your code will always have access to the published copies of the copmetition files.
"""

import os

import pandas as pd
import polars as pl

import kaggle_evaluation.nfl_inference_server


def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Replace this function with your inference code.
    You can return either a Pandas or Polars dataframe, though Polars is recommended for performance.
    Each batch of predictions (except the very first) must be returned within 5 minutes of the batch features being provided.
    """
    predictions = pl.DataFrame({'x': [0.0] * len(test), 'y': [0.0] * len(test)})

    assert isinstance(predictions, (pd.DataFrame, pl.DataFrame))
    assert len(predictions) == len(test)
    return predictions


# When your notebook is run on the hidden test set, inference_server.serve must be called within 10 minutes of the notebook starting
# or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very
# first `predict` call, which does not have the usual 5 minute response deadline.
inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/nfl-big-data-bowl-2026-prediction/',))
```
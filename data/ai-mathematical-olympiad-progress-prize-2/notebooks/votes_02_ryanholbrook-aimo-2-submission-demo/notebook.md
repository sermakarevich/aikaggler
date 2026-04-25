# AIMO 2 Submission Demo

- **Author:** Ryan Holbrook
- **Votes:** 1157
- **Ref:** ryanholbrook/aimo-2-submission-demo
- **URL:** https://www.kaggle.com/code/ryanholbrook/aimo-2-submission-demo
- **Last run:** 2024-10-23 14:02:58.857000

---

```python
import os

import pandas as pd
import polars as pl

import kaggle_evaluation.aimo_2_inference_server
```

The evaluation API requires that you set up a server which will respond to inference requests. We have already defined the server; you just need write the `predict` function. When we evaluate your submission on the hidden test set the client defined in `aimo_2_gateway` will run in a different container with direct access to the hidden test set and hand off each question one at a time, in random order.

Your code will always have access to the published copies of the files.

```python
# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    id_ = id_.item(0)
    question = question.item(0)
    # Make a prediction
    prediction = 0  # model.predict(question)
    return pl.DataFrame({'id': id_, 'answer': 0})
```

When your notebook is run on the hidden test set, `inference_server.serve()` must be called within 15 minutes of the notebook starting or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very first `predict` call, which does not have the usual 30 minute response deadline.

```python
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
        )
    )
```
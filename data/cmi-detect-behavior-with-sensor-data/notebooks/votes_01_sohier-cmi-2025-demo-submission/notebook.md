# CMI 2025 Demo Submission

- **Author:** Sohier Dane
- **Votes:** 875
- **Ref:** sohier/cmi-2025-demo-submission
- **URL:** https://www.kaggle.com/code/sohier/cmi-2025-demo-submission
- **Last run:** 2025-05-30 02:45:44.417000

---

```python
import os

import pandas as pd
import polars as pl

import kaggle_evaluation.cmi_inference_server
```

The evaluation API requires that you set up a server which will respond to inference requests. We have already defined the server; you just need write the predict function. When we evaluate your submission on the hidden test set the client defined in the gateway will run in a different container with direct access to the hidden test set and hand off the one sequence at a time.

Your code will always have access to the published copies of the files.

```python
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # Replace this function with your inference code.
    # You can return either a Pandas or Polars dataframe, though Polars is recommended.
    # Each prediction (except the very first) must be returned within 30 minutes of the batch features being provided.
    return 'Text on phone'
```

```python
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )
```
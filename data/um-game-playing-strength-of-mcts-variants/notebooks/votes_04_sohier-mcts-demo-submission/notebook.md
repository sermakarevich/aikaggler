# MCTS Demo Submission

- **Author:** Sohier Dane
- **Votes:** 435
- **Ref:** sohier/mcts-demo-submission
- **URL:** https://www.kaggle.com/code/sohier/mcts-demo-submission
- **Last run:** 2024-09-06 16:40:52.003000

---

```python
import os

import polars as pl

import kaggle_evaluation.mcts_inference_server
```

The evaluation API requires that you set up a server which will respond to inference requests. We have already defined the server; you just need write the `predict` function. When we evaluate your submission on the hidden test set the client defined in `mcts_gateway` will run in a different container with direct access to the hidden test set and hand off the data in batches of 100. 

Your code will always have access to the published copies of the files.

```python
def predict(test: pl.DataFrame, sample_sub: pl.DataFrame):
    # Replace this function with your inference code.
    # You can return either a Pandas or Polars dataframe, though Polars is recommended.
    # Each batch of predictions (except the very first) must be returned within 10 minutes of the batch features being provided.
    return sample_sub.with_columns(pl.col('utility_agent1') + 0.123)
```

When your notebook is run on the hidden test set, `inference_server.serve` must be called within 15 minutes of the notebook starting or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very first `predict` call, which does not have the usual 10 minute response deadline.

```python
inference_server = kaggle_evaluation.mcts_inference_server.MCTSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv',
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv'
        )
    )
```

Note that nothing past `inference_server.serve()` will be run when your submission is evaluated on the hidden test set.
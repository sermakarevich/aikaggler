# MCTS Variants - Getting Started

- **Author:** inversion
- **Votes:** 581
- **Ref:** inversion/mcts-variants-getting-started
- **URL:** https://www.kaggle.com/code/inversion/mcts-variants-getting-started
- **Last run:** 2024-09-09 13:42:00.103000

---

## Code submission example

This competition is different than a standard Code competition. It uses a backend to serve chunks of the test data, which your model must run inference on before the next chunk is served. Because of this, you have to include some required code for your submission to work.

The evaluation API requires that you set up a server which will respond to inference requests. We have already defined the server; you just need write the predict function. When we evaluate your submission on the hidden test set the client defined in `mcts_gateway` will run in a different container with direct access to the hidden test set and hand off the data in batches of 100.

Your code will always have access to the published copies of the files.

```python
import os
import sys

import polars as pl
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

from pathlib import Path
comp_path = Path('/kaggle/input/um-game-playing-strength-of-mcts-variants')
```

### Critical import

```python
import kaggle_evaluation.mcts_inference_server
```

### Building a model

This example trains a model inline, but you will probably want to train a model offline, import it, and just use the notebook for inference.

It's wrapped in a function because this Code format requires you to get to the `predict` function within 15 minutes. So, if you have something expensive to do up-front (e.g., training a model, loading a model, etc.), and it will take longer than 15 minutes, those operations need to happen in the first call of `predict` so the gateway server doesn't time out.

```python
target = 'utility_agent1'

def train_model():

    # we'll use these in each call of `predict`
    global obj_cols, enc, rf

    train = pl.read_csv(comp_path / 'train.csv')
    y_train = train[target]

    cols_to_drop = ['num_draws_agent1', 'num_losses_agent1', 'num_wins_agent1', target]
    train = train.drop(cols_to_drop)
    
    obj_cols =  train.select(pl.col(pl.String)).columns
    
    enc = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-999, encoded_missing_value=-9999)
    enc.fit(train[obj_cols])
    train_transformed = enc.transform(train[obj_cols])
    for e, c in enumerate(obj_cols):
        train = train.with_columns(pl.Series(c, train_transformed[:, e]))
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1)
    rf.fit(train, y_train)
```

### Inference should be in a function named `predict` as similar to the following:

If you're not doing anything "expensive" in the first `predict` call, you don't technically need the counter code.

```python
counter = 0
def predict(test, submission):
    global counter
    if counter == 0:
        # Perform any additional slow steps in the first call to `predict`
        train_model()
    counter += 1    
    test_transformed = enc.transform(test[obj_cols])
    for e, c in enumerate(obj_cols):
        test = test.with_columns(pl.Series(c, test_transformed[:, e]))
    return submission.with_columns(pl.Series(target, rf.predict(test)))
```

### Calling the gateway server

You must run the cell below within 15 minutes of the notebook re-run or the gateway will throw an error. If you need more than 15 minutes time to load your model, train your model, etc, you should do it during the very first `predict` call, as shown previously.

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
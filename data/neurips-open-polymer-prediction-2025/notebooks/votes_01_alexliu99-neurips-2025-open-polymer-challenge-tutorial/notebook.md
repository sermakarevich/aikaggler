# NeurIPS 2025 Open Polymer Challenge Tutorial

- **Author:** Alex Liu
- **Votes:** 987
- **Ref:** alexliu99/neurips-2025-open-polymer-challenge-tutorial
- **URL:** https://www.kaggle.com/code/alexliu99/neurips-2025-open-polymer-challenge-tutorial
- **Last run:** 2025-06-13 02:06:33.977000

---

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

# Introduction

Polymers are macromolecules composed of many repeating units. There are several ways to represent polymers. In this tutorial, we focus on the single repeating unit, or monomer, with its polymerization points. You can represent a polymer in three main ways:

* **SMILES (Simplified Molecular Input Line Entry System):** One simple extension of standard SMILES uses the `*` character to mark polymerization points. This format is supported by RDKit. [Learn more](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) or see the [RDKit documentation](https://www.rdkit.org/docs/GettingStartedInPython.html#reading-and-writing-molecules).

<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*tuXqpPuKO9PBPwUx5jz5Nw.png" alt="SMILES example" width="50%"/>
</div>

* **Vector:** You can extract features from the polymer structure to create a vector. A common method is to use Morgan fingerprints. See [RDKit fingerprint documentation - rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html).

<div align="center">
<img src="https://www.researchgate.net/profile/Your-Profile-Name/publication/382398932/figure/fig3/AS:XXXXXXX/An-example-of-structure-based-representations-Aspirins-Morgan-fingerprints-are-a-binary.png" alt="Morgan Fingerprints Example" width="30%"/>
</div>

* **Graph:** A graph naturally represents polymers, where nodes are atoms and edges are bonds. Both atoms and bonds can have multiple attributes, which can improve model predictions.

<div align="center">
<img src="https://www.wolfram.com/language/12/molecular-structure-and-computation/assets.en/molecule-graphs/O_84.png" alt="Molecular Graph" width="30%"/>
</div>

In this tutorial, we provide baseline implementations for polymer modeling using these three representations. You are also encouraged to explore other formats, such as point clouds in 3D space, to predict polymer properties accurately.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = '/kaggle/input/open-polymer-challenge/train.csv'
train_df = pd.read_csv(csv_path)

# 1. split off 20% for dev_test
temp_df, dev_test = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,  # for reproducibility
    shuffle=True
)

# 2. split the remaining 80% into 75% train / 25% valid → 0.6 / 0.2 overall
dev_train, dev_val = train_test_split(
    temp_df,
    test_size=0.25,  # 0.25 * 0.8 = 0.2 of the original
    random_state=42,
    shuffle=True
)

# Verify sizes
print(f"Total rows:   {len(train_df)}")
print(f"Dev train:    {len(dev_train)} ({len(dev_train)/len(train_df):.2%})")
print(f"Dev valid:    {len(dev_val)} ({len(dev_val)/len(train_df):.2%})")
print(f"Dev test:     {len(dev_test)} ({len(dev_test)/len(train_df):.2%})")
print(f"Polymer example:{dev_train['SMILES'].to_list()[:3]}")
print(f"Columns:{dev_train.columns}")
```

# Modeling polymers as sequential strings with LSTM

LSTMs first process SMILES strings into character-based tokens using a predefined dictionary, such as:

```python
char_dic = {
    '<pad>': 0,
    '#': 1,   # Triple bond
    '%': 2,   # Two-digit ring closure (e.g., '%10')
    '(': 3,   # Branch opening
    ')': 4,   # Branch closing
    '*': 5,   # Wildcard atom (used in BigSMILES for polymer repeating units)
    '+': 6,   # Positive charge
    '-': 7,   # Negative charge
    '0': 8,   # Ring closure digit
    '1': 9,
    '2': 10,
    ...
}
```

Each character is assigned an index number as its token. Given a SMILES string as input, we obtain a sequence of tokens:

```python}
tokens = [char_dic[char] for char in SMILES_string]
```

We can then build an LSTM using these token embeddings to predict the desired property. We use `torch-molecule` to simplify the implementation. We can then build an LSTM using these token embeddings to predict the desired property. We use torch-molecule to simplify the implementation. Full model details are available at [this link](https://github.com/liugangcode/torch-molecule/blob/main/torch_molecule/predictor/lstm/modeling_lstm.py).

```python
from tqdm.notebook import tqdm as notebook_tqdm
import tqdm
tqdm.tqdm = notebook_tqdm
tqdm.trange = notebook_tqdm

from torch_molecule import LSTMMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec

search_parameters = {
    "output_dim": ParameterSpec(ParameterType.INTEGER, (8, 32)),
    "LSTMunits": ParameterSpec(ParameterType.INTEGER, (30, 120)),
    "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-4, 1e-2)),
}

lstm = LSTMMolecularPredictor(
    task_type="regression",
    num_task=5,
    batch_size=192,
    epochs=200,
    verbose=True
)

print("Model initialized successfully")
X_train = dev_train['SMILES'].to_list()
y_train = dev_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
X_val = dev_val['SMILES'].to_list()
y_val = dev_val[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
lstm.autofit(
    X_train = X_train,
    y_train = y_train,
    X_val = X_val,
    y_val = y_val,
    search_parameters=search_parameters,
    n_trials = 10 # number of times searching the best hyper-parameters
)
```

```python
import numpy as np
from sklearn.metrics import mean_squared_error

X_test = dev_test['SMILES'].to_list()
y_test = dev_test[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
y_predict = lstm.predict(X_test)['prediction']

task_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Compute MSE per task, skipping NaNs
mse_per_task = {}
for i, name in enumerate(task_names):
    mask = ~np.isnan(y_test[:, i])
    if mask.sum() > 0:
        mse = mean_squared_error(y_test[mask, i], y_predict[mask, i])
        mse_per_task[name] = mse
    else:
        mse_per_task[name] = np.nan  # no valid data

print("MSE per task:")
for name, mse in mse_per_task.items():
    print(f"  {name}: {mse:.4f}")

# Compute overall MSE across all tasks, skipping NaNs
mask_all = ~np.isnan(y_test)
y_true_flat = y_test[mask_all]
y_pred_flat = y_predict[mask_all]
mse_overall = mean_squared_error(y_true_flat, y_pred_flat)

print(f"Overall MSE: {mse_overall:.4f}")
```

# Modeling polymers as vectors with random forests

We can extract fingerprints using RDKit, then we implement the random forests with sklearn for each of the tasks.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def smiles_to_fp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))

# Convert SMILES to fingerprint features
X_train_feats = np.vstack([smiles_to_fp(s) for s in X_train])
X_val_feats   = np.vstack([smiles_to_fp(s) for s in X_val])
X_test = dev_test['SMILES'].to_list()
X_test_feats  = np.vstack([smiles_to_fp(s) for s in X_test])

# Combine train and validation sets
X_dev_feats = np.vstack([X_train_feats, X_val_feats])
y_dev = np.vstack([y_train, y_val])

# Test targets
y_test = dev_test[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()

task_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
models = {}
y_pred = np.zeros_like(y_test)

# Train one random forest per task
for idx, name in enumerate(task_names):
    print('Training random forest for the task:', name)
    y_col = y_dev[:, idx]
    mask  = ~np.isnan(y_col)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_dev_feats[mask], y_col[mask])
    models[name] = rf
    # Predict on test set
    y_pred[:, idx] = rf.predict(X_test_feats)

# Compute MSE per task, skipping NaNs
mse_per_task = {}
for i, name in enumerate(task_names):
    print('Predicting for the task:', name)
    mask = ~np.isnan(y_test[:, i])
    if mask.sum() > 0:
        mse = mean_squared_error(y_test[mask, i], y_pred[mask, i])
        mse_per_task[name] = mse
    else:
        mse_per_task[name] = np.nan

print("MSE per task:")
for name, mse in mse_per_task.items():
    print(f"  {name}: {mse:.4f}")

# Compute overall MSE across all tasks, skipping NaNs
mask_all = ~np.isnan(y_test)
y_true_flat = y_test[mask_all]
y_pred_flat = y_pred[mask_all]
mse_overall = mean_squared_error(y_true_flat, y_pred_flat)
print(f"Overall MSE: {mse_overall:.4f}")
```

# Modeling polymers as graphs with Graph Neural Networks

We can use GNNs for polymers. For each atom, the message passing mechanism in GNNs progogates its neighboring atoms information to the center to update the representation vetors. A pooling method (e.g., sum pooling) summarize atom-level representations to the graph (molecule) level. Finally we can make predict based on the representation. We still use `torch-molecule` to simplify the implementation of GNNs

```python
from tqdm.notebook import tqdm as notebook_tqdm
import tqdm
tqdm.tqdm = notebook_tqdm
tqdm.trange = notebook_tqdm

from torch_molecule import GNNMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec
import numpy as np
from sklearn.metrics import mean_squared_error

search_parameters = {
    'num_layer': ParameterSpec(
        param_type=ParameterType.INTEGER,
        value_range=(2, 5)
    ),
    'hidden_size': ParameterSpec(
        param_type=ParameterType.INTEGER,
        value_range=(64, 512)
    ),
    'learning_rate': ParameterSpec(
        param_type=ParameterType.LOG_FLOAT,
        value_range=(1e-4, 1e-2)
    ),
}

gnn = GNNMolecularPredictor(
    task_type="regression",
    num_task=5,
    batch_size=192,
    epochs=200,
    verbose=True
)

print("Model initialized successfully")
X_train = dev_train['SMILES'].to_list()
y_train = dev_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
X_val = dev_val['SMILES'].to_list()
y_val = dev_val[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
gnn.autofit(
    X_train = X_train,
    y_train = y_train,
    X_val = X_val,
    y_val = y_val,
    search_parameters=search_parameters,
    n_trials = 10 # number of times searching the best hyper-parameters
)
```

```python
X_test = dev_test['SMILES'].to_list()
y_test = dev_test[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
y_predict = gnn.predict(X_test)['prediction']

task_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Compute MSE per task, skipping NaNs
mse_per_task = {}
for i, name in enumerate(task_names):
    mask = ~np.isnan(y_test[:, i])
    if mask.sum() > 0:
        mse = mean_squared_error(y_test[mask, i], y_predict[mask, i])
        mse_per_task[name] = mse
    else:
        mse_per_task[name] = np.nan  # no valid data

print("MSE per task:")
for name, mse in mse_per_task.items():
    print(f"  {name}: {mse:.4f}")

# Compute overall MSE across all tasks, skipping NaNs
mask_all = ~np.isnan(y_test)
y_true_flat = y_test[mask_all]
y_pred_flat = y_predict[mask_all]
mse_overall = mean_squared_error(y_true_flat, y_pred_flat)

print(f"Overall MSE: {mse_overall:.4f}")
```

# Submission

**Note:** In this tutorial, we use mean squared error (MSE) as both the loss function and the evaluation metric on the development set. This choice does not exactly match the leaderboard evaluation metric, which is the weighted mean absolute error. Participants may explore different loss functions and development metrics for their submissions.

Based on the performance on the development set, we can use the best predictions or combine their predictions to create the final submission file. For example, we combine the prediction from GNNs and LSTMs.

```python
import pandas as pd

# Load sample submission
sample_sub = pd.read_csv('/kaggle/input/open-polymer-challenge/sample_submission.csv')
print(sample_sub.head())

# Load test set
test_df = pd.read_csv('/kaggle/input/open-polymer-challenge/test.csv')
print(test_df.head())

# Prepare test SMILES list
X_test = test_df['SMILES'].to_list()

# Predict using the trained LSTM model
lstm_preds = lstm.predict(X_test)['prediction']
gnn_preds = gnn.predict(X_test)['prediction']
preds = (lstm_preds + gnn_preds) / 2

# Build the submission DataFrame
submission_df = sample_sub.copy()
submission_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']] = preds

print('submission_df', submission_df)
```

```python
# save to CSV
submission_df.to_csv('submission.csv', index=False)
```
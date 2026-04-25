# JS-2024 v15-04-01 RMF-NN+Starter+TabM+Ridge

- **Author:** AI-Cat
- **Votes:** 509
- **Ref:** konstantinboyko/js-2024-v15-04-01-rmf-nn-starter-tabm-ridge
- **URL:** https://www.kaggle.com/code/konstantinboyko/js-2024-v15-04-01-rmf-nn-starter-tabm-ridge
- **Last run:** 2025-01-02 13:11:18.163000

---

# Libraries

```python
import polars as pl
!pip install rtdl_num_embeddings --no-index --find-links=/kaggle/input/jane-street-import/rtdl_num_embeddings
```

```python
import os, sys, gc
import enum
import datetime
import pickle
import dill
import numpy as np
import pandas as pd
import polars as pl

from sklearn.metrics import r2_score

import lightgbm as lgb
from lightgbm import LGBMRegressor, Booster
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule

sys.path.append("/kaggle/input/jane-street-real-time-market-data-forecasting")
import kaggle_evaluation.jane_street_inference_server

sys.path.append("/kaggle/input/src/tabm_reference")
from tanm_reference import Model, make_parameter_groups

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

@enum.unique
class DataEnum(enum.IntEnum):
    Train = 0
    Valid = 1
    Test = 2
    Infer = 3

is_debug = False
is_rerun = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', "") != "" 
is_local = os.environ.get("DOCKER_USING", "") == "LOCAL"
num_workers = 4
if is_rerun:
    is_debug = False

def load_from_dill(model_name, model_path=None, file_ext='.dill'):
    model_object = None
    with open(f"{model_path}/{model_name}{file_ext}", "rb") as file_handle:
        model_object = dill.load(file_handle)
    return model_object

# RMF-NN, Starter, TabM, Ridge
ensem_koef = dict(
    nn = 0.55,
    starter = 0.35,
    tabm = 0.3,
    ridge = 0.10,
)
sum_ensem_koef = 0.0
for value in ensem_koef.values():
    sum_ensem_koef += value 
for key, value in ensem_koef.items():
    ensem_koef[key] = value / sum_ensem_koef
print(ensem_koef)
```

# [yunsuxiaozi - JS2024 Starter v10 LB=0.0070 (0.0054)](https://www.kaggle.com/code/yunsuxiaozi/js2024-starter?scriptVersionId=206770572)

```python
model_path = "/kaggle/input/js-2024/13-01-11" if is_local else "/kaggle/input/js-2024-13-01-11"
final_feature = ['symbol_id','sin_time_id','cos_time_id','sin_time_id_halfday','cos_time_id_halfday'] + [f'feature_{i:02}' for i in range(79)]

lgb = Booster(model_file=f"{model_path}/lgb.model")
cat = CatBoostRegressor()
cat.load_model(f"{model_path}/cat.model")
xgb = XGBRegressor()
xgb.load_model(f"{model_path}/xgb.model")

def predict_starter(test,lags):
    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    test=test.to_pandas()
    test['sin_time_id']=np.sin(2*np.pi*test['time_id']/967)
    test['cos_time_id']=np.cos(2*np.pi*test['time_id']/967)
    test['sin_time_id_halfday']=np.sin(2*np.pi*test['time_id']/483)
    test['cos_time_id_halfday']=np.cos(2*np.pi*test['time_id']/483)
    test=test.fillna(-1)
    test=test[final_feature]
    eps=1e-10
    test_preds=0.55*lgb.predict(test)+0.2*cat.predict(test)+0.25*xgb.predict(test)
    test_preds=np.clip(test_preds,-5+eps,5-eps)
    predictions = predictions.with_columns(pl.Series('responder_6', test_preds.ravel()))
    return predictions
```

# [I2nfinit3y - Jane Street | TabM/FT-Transformer inference LB=0.0064 (0.0074)](https://www.kaggle.com/code/i2nfinit3y/jane-street-tabm-ft-transformer-inference?scriptVersionId=213715783)

```python
model_path = '/kaggle/input/js-2024' + ('/' if is_local else '-') + '19-02-1/last_tabm.pt'
stats_path = '/kaggle/input/js-2024' + ('/' if is_local else '-') + '19-02-1/data_stats.dill'
device = torch.device('cuda:0')

target_col = "responder_6"
necessary_cols = [target_col, 'weight']
feat_clear_categ = ["feature_09", "feature_10", "feature_11"]
feature_categ = feat_clear_categ + ['symbol_id', 'time_id']
feature_cols = [f"feature_{idx:02d}" for idx in range(79) if idx not in [9, 10, 11, 61]]
responder_cols = [f"responder_{idx}_lag_1" for idx in range(9)] 
feature_cont = feature_cols + responder_cols
dataset_cols = feature_cont + necessary_cols + feature_categ
std_feature = [i for i in feature_cont]

batch_size = 8192
n_cont_features = len(feature_cont)
n_cat_features = len(feature_categ)
n_classes = None
cat_cardinalities = [23, 10, 32, 40, 969]
# TabM
arch_type = 'tabm'
bins = None
model_koef = 32

print(n_cont_features, n_cat_features, len(dataset_cols))

category_mappings = {
    'feature_09': {2: 0, 4: 1, 9: 2, 11: 3, 12: 4, 14: 5, 15: 6, 25: 7, 26: 8, 30: 9, 
        34: 10, 42: 11, 44: 12, 46: 13, 49: 14, 50: 15, 57: 16, 64: 17, 68: 18, 70: 19, 81: 20, 82: 21},
    'feature_10': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 12: 8},
    'feature_11': {9: 0, 11: 1, 13: 2, 16: 3, 24: 4, 25: 5, 34: 6, 40: 7, 48: 8, 50: 9, 59: 10, 62: 11, 63: 12, 66: 13,
        76: 14, 150: 15, 158: 16, 159: 17, 171: 18, 195: 19, 214: 20, 230: 21, 261: 22, 297: 23, 336: 24, 376: 25, 388: 26, 410: 27, 522: 28, 534: 29, 539: 30},
    'symbol_id': {i : i for i in range(39)},
    'time_id' : {i : i for i in range(968)}
}

def standardize(df, feature_cols, means, stds):
    return df.with_columns([
        ((pl.col(col) - means[col]) / stds[col]).alias(col) for col in feature_cols
    ])

def encode_column(df, column, mapping):
    max_value = max(mapping.values())
    def encode_category(category):
        return mapping.get(category, max_value + 1)
    return df.with_columns(pl.col(column).map_elements(encode_category, return_dtype=pl.Int64).alias(column))

model_tabm = Model(
    n_num_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    n_classes=n_classes,
    backbone={
        'type': 'MLP',
        'n_blocks': 3 ,
        'd_block': 512,
        'dropout': 0.25,
    },
    bins=bins,
    num_embeddings=(
        None
        # {
        #     'type': 'PeriodicEmbeddings',
        #     'd_embedding': 16,
        #     'lite':True,
        # }
    ),
    arch_type=arch_type,
    k=model_koef,
).to(device)

with open(stats_path, "rb") as file_handle:
    data_stats = dill.load(file_handle)
means, stds = data_stats['means'], data_stats['stds']

checkpoint = torch.load(model_path, weights_only=True)
model_tabm.load_state_dict(checkpoint['model_state_dict'])
model_tabm.to(device)

lags_history = None

def predict_tabm(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    global lags_history
    for col in feature_categ:
        test = encode_column(test, col, category_mappings[col])
    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    time_id = test.select("time_id").to_numpy()[0]
    if time_id == 0:
        lags = lags.with_columns(pl.col('time_id').cast(pl.Int64))
        lags = lags.with_columns(pl.col('symbol_id').cast(pl.Int64))    
        lags_history = lags
    lags = lags_history.clone().group_by(["date_id", "symbol_id"], maintain_order=True).last() # pick up last record of previous date
    test = test.join(lags, on=["time_id", "symbol_id"], how="left")
    
    test = test.select(pl.all().forward_fill())
    test = test.with_columns([
        pl.col(col).fill_null(0) for col in feature_cont + feat_clear_categ
    ])
    test = standardize(test, std_feature, means, stds)

    model_tabm.eval()
    with torch.no_grad():
        X_cont = torch.tensor(test[feature_cont].to_numpy(), dtype=torch.float32).to(device)
        X_categ = torch.tensor(test[feature_categ].to_numpy(), dtype=torch.int64).to(device)
        outputs = model_tabm(X_cont, X_categ)
        # Assuming the model outputs a tensor of shape (batch_size, 1)
        preds = outputs.squeeze(-1).cpu().numpy()
        preds = preds.mean(1)
    
    predictions = \
        test.select('row_id').\
        with_columns(
            pl.Series(
                name   = 'responder_6', 
                values = np.clip(preds, a_min = -5, a_max = 5),
                dtype  = pl.Float64,
            )
        )
    
    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame)
    # with columns 'row_id', 'responer_6'
    assert list(predictions.columns) == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)
    return predictions
```

# [yunsuxiaozi - JS Ridge baseline v03 LB=0.0033](https://www.kaggle.com/code/yunsuxiaozi/js-ridge-baseline?scriptVersionId=202739388)

```python
rdg = load_from_dill(model_name='ridge', model_path="/kaggle/input/js-2024/16-01" if is_local else "/kaggle/input/js-2024-16-01")

def predict_ridge(test, lags):
    cols = [f'feature_{i:02}' for i in range(79)]
    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    test_preds = rdg.predict(test[cols].to_pandas().fillna(3).values)
    predictions = predictions.with_columns(pl.Series('responder_6', test_preds.ravel()))
    return predictions
```

# [Xiang Sheng - Jane Street RMF NN + XGB Lb=0.0076 (0.0056)](https://www.kaggle.com/code/voix97/jane-street-rmf-nn-xgb)

```python
class CONFIG:
    seed = 42
    target_col = "responder_6"
    feature_cols = [f"feature_{idx:02d}" for idx in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
    model_paths = [
        "/kaggle/input/js-xs-nn-trained-model",
        "/kaggle/input/js-with-lags-trained-xgb/result.pkl",
    ]

valid = pl.scan_parquet(f"/kaggle/input/js24-preprocessing-create-lags/validation.parquet/").collect().to_pandas()

xgb_model = None
model_path = CONFIG.model_paths[1]
with open( model_path, "rb") as fp:
    result = pickle.load(fp)
    xgb_model = result["model"]
xgb_feature_cols = ["symbol_id", "time_id"] + CONFIG.feature_cols

# Custom R2 metric for validation
def r2_val(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2

class NN(LightningModule):
    def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.BatchNorm1d(in_dim))
            if i > 0:
                layers.append(nn.SiLU())
            if i < len(dropouts):
                layers.append(nn.Dropout(dropouts[i]))
            layers.append(nn.Linear(in_dim, hidden_dim))
            # layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))  
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_step_outputs = []

    def forward(self, x):
        return 5 * self.model(x).squeeze(-1)  

    def training_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w  
        loss = loss.mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w
        loss = loss.mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.validation_step_outputs.append((y_hat, y, w))
        return loss

    def on_validation_epoch_end(self):
        """Calculate validation WRMSE at the end of the epoch."""
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
        if self.trainer.sanity_checking:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        else:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
            # r2_val
            val_r_square = r2_val(y, prob, weights)
            self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")

N_folds = 5
models = []
for fold in range(N_folds):
    checkpoint_path = f"{CONFIG.model_paths[0]}/nn_{fold}.model"
    model = NN.load_from_checkpoint(checkpoint_path)
    models.append(model.to("cuda:0"))

X_valid = valid[ xgb_feature_cols ]
y_valid = valid[ CONFIG.target_col ]
w_valid = valid[ "weight" ]
y_pred_valid_xgb = xgb_model.predict(X_valid)
valid_score = r2_score( y_valid, y_pred_valid_xgb, sample_weight=w_valid )
valid_score

X_valid = valid[ CONFIG.feature_cols ]
y_valid = valid[ CONFIG.target_col ]
w_valid = valid[ "weight" ]
X_valid = X_valid.fillna(method = 'ffill').fillna(0)
X_valid.shape, y_valid.shape, w_valid.shape

y_pred_valid_nn = np.zeros(y_valid.shape)
with torch.no_grad():
    for model in models:
        model.eval()
        y_pred_valid_nn += model(torch.FloatTensor(X_valid.values).to("cuda:0")).cpu().numpy() / len(models)
valid_score = r2_score( y_valid, y_pred_valid_nn, sample_weight=w_valid )
valid_score

y_pred_valid_ensemble = 0.5 * (y_pred_valid_xgb + y_pred_valid_nn)
valid_score = r2_score( y_valid, y_pred_valid_ensemble, sample_weight=w_valid )
valid_score

del valid, X_valid, y_valid, w_valid
gc.collect()

lags_ : pl.DataFrame | None = None

def predict_nn(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    global lags_
    if lags is not None:
        lags_ = lags

    predictions_14 = test.select('row_id', pl.lit(0.0).alias('responder_6'),)

    # add this part to reuse lags of previous date ids when rows have more than 0 time_ids.
    lags = lags_.clone().group_by(["date_id", "symbol_id"], maintain_order=True).last()
    test = test.join(lags, on=["date_id", "symbol_id"],  how="left")
    
    #if not lags is None:
    #    lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last() # pick up last record of previous date
    #    test = test.join(lags, on=["date_id", "symbol_id"],  how="left")
    #else:
    #    test = test.with_columns(( pl.lit(0.0).alias(f'responder_{idx}_lag_1') for idx in range(9) ))

    preds = np.zeros((test.shape[0],))
    preds += xgb_model.predict(test[xgb_feature_cols].to_pandas()) / 2
    test_input = test[CONFIG.feature_cols].to_pandas()
    test_input = test_input.fillna(method = 'ffill').fillna(0)
    test_input = torch.FloatTensor(test_input.values).to("cuda:0")
    with torch.no_grad():
        for i, nn_model in enumerate(models):
            nn_model.eval()
            preds += nn_model(test_input).cpu().numpy() / 10
    #print(f"predict> preds.shape =", preds.shape)

    predictions_14 = test.select('row_id').\
        with_columns(
            pl.Series(
                name   = 'responder_6', 
                values = np.clip(preds, a_min = -5, a_max = 5),
                dtype  = pl.Float64,
            )
        )
    return predictions_14
```

# Ensemble

```python
is_first = False

def predict(test:pl.DataFrame, lags:pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    global is_first

    pd_nn = predict_nn(test,lags).to_pandas()
    pd_starter = predict_starter(test,lags).to_pandas()
    pd_ridge = predict_ridge(test,lags).to_pandas()
    pd_tabm = predict_tabm(test,lags).to_pandas()

    pd_nn = pd_nn.rename(columns={'responder_6': 'col_nn'})
    pd_starter = pd_starter.rename(columns={'responder_6': 'col_starter'})
    pd_ridge = pd_ridge.rename(columns={'responder_6': 'col_ridge'})
    pd_tabm = pd_tabm.rename(columns={'responder_6': 'col_tabm'})

    pds = pd.merge(pd_nn, pd_starter, on=['row_id'])
    pds = pd.merge(pds, pd_ridge, on=['row_id'])
    pds = pd.merge(pds, pd_tabm, on=['row_id'])

    pds['responder_6'] = \
        pds['col_nn'] * ensem_koef['nn'] + \
        pds['col_starter'] * ensem_koef['starter'] + \
        pds['col_ridge'] * ensem_koef['ridge'] + \
        pds['col_tabm'] * ensem_koef['tabm'] 

    if not is_first:
        display(pds)
        is_first = True

    predictions = test.select('row_id', pl.lit(0.0).alias('responder_6'))
    pred = pds['responder_6'].to_numpy()
    predictions = predictions.with_columns(pl.Series('responder_6', pred.ravel()))
    return predictions

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )
```
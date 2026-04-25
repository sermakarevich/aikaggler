# JS2024 Starter

- **Author:** yunsuxiaozi
- **Votes:** 691
- **Ref:** yunsuxiaozi/js2024-starter
- **URL:** https://www.kaggle.com/code/yunsuxiaozi/js2024-starter
- **Last run:** 2024-11-12 11:33:28.153000

---

## Created by <a href="https://github.com/yunsuxiaozi/">yunsuxiaozi </a>  2024/11/12

### Here we will use lightgbm、xgboost、catboost and origin features to create a simple baseline.

- version1:LB:0.0043

- version2:LB:0.0045

- version3:LB:0.0045

- version4:LB:0.0045

- version5:LB:0.0045

- version6:failed

- version7:LB:0.0051

- version8:LB:0.0050

- version9:Failed

# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">Import Libraries</h1></span>

```python
source_file_path = '/kaggle/input/yunbase/Yunbase/baseline.py'
target_file_path = '/kaggle/working/baseline.py'
with open(source_file_path, 'r', encoding='utf-8') as file:
    content = file.read()
with open(target_file_path, 'w', encoding='utf-8') as file:
    file.write(content)
```

```python
!pip install -q --requirement /kaggle/input/yunbase/Yunbase/requirements.txt  \
--no-index --find-links file:/kaggle/input/yunbase/
```

```python
from baseline import Yunbase
import polars as pl#similar to pandas, but with better performance when dealing with large datasets.
import pandas as pd#read csv,parquet
import numpy as np#for scientific computation of matrices
#model
from  lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import os#Libraries that interact with the operating system
import gc#rubbish collection
#environment provided by competition hoster
import kaggle_evaluation.jane_street_inference_server

import random#provide some function to generate random_seed.
#set random seed,to make sure model can be recurrented.
def seed_everything(seed):
    np.random.seed(seed)#numpy's random seed
    random.seed(seed)#python built-in random seed
seed_everything(seed=2025)
```

# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">Load Train Data</h1></span>

I accidentally made a mistake by not filling in missing values in the training data and filling in -1 in the test data, resulting in a good LB (Those who know the reason can leave a message in the discussion forum)

```python
yunbase=Yunbase()
data=[]
for i in [6,7,8,9]:
    train=pl.read_parquet(f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet")
    train=train.to_pandas()
    train['sin_time_id']=np.sin(2*np.pi*train['time_id']/967)
    train['cos_time_id']=np.cos(2*np.pi*train['time_id']/967)
    train['sin_time_id_halfday']=np.sin(2*np.pi*train['time_id']/483)
    train['cos_time_id_halfday']=np.cos(2*np.pi*train['time_id']/483)
    #train=train.fillna(-1)
    train=yunbase.reduce_mem_usage(train,float16_as32=False)
    data.append(train)
train=pd.concat(data)
print(f"train.shape:{train.shape}")
del data
gc.collect()
final_feature=['symbol_id','sin_time_id','cos_time_id','sin_time_id_halfday','cos_time_id_halfday']+[f'feature_0{i}' if i<10 else f'feature_{i}' for i in range(79)]
train=train[['responder_6']+final_feature]
train.head()
```

# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">Model training</h1></span>

```python
lgb_params={"boosting_type": "gbdt","metric": 'rmse',
            'random_state': 2025,  "max_depth": 10,"learning_rate": 0.1,
            "n_estimators": 120,"colsample_bytree": 0.6,"colsample_bynode": 0.6,"verbose": -1,"reg_alpha": 0.2,
            "reg_lambda": 5,"extra_trees":True,'num_leaves':64,"max_bin":255,
            'device':'gpu','gpu_use_dp':True,
            }

cat_params={'task_type':'GPU',
           'random_state':2025,
           'eval_metric'         : 'RMSE',
           'bagging_temperature' : 0.50,
           'iterations'          : 200,
           'learning_rate'       : 0.1,
           'max_depth'           : 12,
           'l2_leaf_reg'         : 1.25,
           'min_data_in_leaf'    : 24,
           'random_strength'     : 0.25, 
           'verbose'             : 0,
          }
xgb_params={'random_state': 2025, 'n_estimators': 125, 
            'learning_rate': 0.1, 'max_depth': 10,
            'reg_alpha': 0.08, 'reg_lambda': 0.8, 
            'subsample': 0.95, 'colsample_bytree': 0.6, 
            'min_child_weight': 3,
            'tree_method':'gpu_hist',
           }
print("lgb")
lgb=LGBMRegressor(**lgb_params)
lgb.fit(train[final_feature].values,train['responder_6'].values)
print("cat")
cat=CatBoostRegressor(**cat_params)
cat.fit(train[final_feature].values,train['responder_6'].values)
print("xgb")
xgb=XGBRegressor(**xgb_params)
xgb.fit(train[final_feature].values,train['responder_6'].values)
```

# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">Model Inference</h1></span>

#### The following code is used in the training set to load more data for training, and not in the testing set to save time. I tried calling this function 100000 times and it would take about an hour.

```python
test=yunbase.reduce_mem_usage(test,float16_as32=False)
```

```python
def predict(test,lags):
    global lgb,cat,xgb
    
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
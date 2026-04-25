# MCTS: DeepTables NN

- **Author:** Vladimir Demidov
- **Votes:** 356
- **Ref:** yekenot/mcts-deeptables-nn
- **URL:** https://www.kaggle.com/code/yekenot/mcts-deeptables-nn
- **Last run:** 2024-10-29 14:34:12.530000

---

Based on single LightGBM baseline with minimal FE, LB **0.447**:

https://www.kaggle.com/code/snufkin77/mcts-strength-relevant-baseline

DeepTables: Deep-learning Toolkit for Tabular data

https://github.com/DataCanvasIO/DeepTables

https://deeptables.readthedocs.io/en/latest/model_config.html#parameters

**Version 1**: single DeepTables NN baseline, LB **0.462**.

**Version 6**: single DeepTables NN, LB **0.448**; `ModelConfig(apply_gbm_features=True)`.

**Version 7**: single DeepTables NN, LB **0.438**; `ModelConfig(apply_gbm_features=True)`, `ModelConfig(nets=['dnn_nets'] + ['fm_nets'] + ['cin_nets'])`.

**Version 8**: same as Version 7 + scaling all numerical features to fix issue with divergence in the validation scores. Now overall CV rmse (0.4324) is closer to LB **0.435**. Note that in this version, the `fit_transform` method of scaler was performed on train and test data combined, which is not a good practice, although it does not violate any Kaggle rules.

**Version 9**: same as Version 8, but using scaler with `fit_transform` on the training data alone and then applying `transform` to the test data. CV 0.4319 | LB **0.438**.

**Version 10**: scaling way from Version 9, enable `LearningRateScheduler` with warmup `(LR_START = 1e-4)` on first epoch. CV 0.4343 | LB **0.434**.

**Version 11**: single CatBoost baseline. CV 0.4160 | LB 0.434.

**Version 12**: ensemble NN (CV 0.4343 | LB 0.434) + CatBoost (CV 0.4160 | LB 0.434), `ens_weights = {'nn': 0.50, 'ctb': 0.50}`. CV 0.4138 | LB **0.430**.

**Version 14**: adjust CatBoost params and ensemble weights; ensemble NN (CV 0.4343 | LB 0.434) + CatBoost (CV 0.4154 | LB 0.431), `ens_weights = {'nn': 0.40, 'ctb': 0.60}`. CV 0.4116 | LB **0.428**.

**Version 16**: same as Version 14, but due to non-deterministic behavior during training on GPU, the final score may slightly variate around. Based on this [contribution](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/542932), the DeepTables model saving-loading issue has been overcome. Now `Output` contains saved pretrained models that can be used for inference.

```python
!pip install --no-index -U --find-links=/kaggle/input/tensorflow-2-15/tensorflow tensorflow==2.15.0
!pip install --no-index -U --find-links=/kaggle/input/deeptables-v0-2-5/deeptables-0.2.5 deeptables==0.2.5
!pip install --no-index -U --find-links=/kaggle/input/fix-deeptables/deeptables-0.2.6 deeptables==0.2.6
```

```python
import os
import math
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, polars as pl
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from colorama import Fore, Style

import tensorflow as tf, deeptables as dt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers.legacy import Adam
from deeptables.models import DeepTable, ModelConfig
from deeptables.models import deepnets
from catboost import CatBoostRegressor

import kaggle_evaluation.mcts_inference_server

warnings.filterwarnings('ignore')
print('TensorFlow version:',tf.__version__+',',
      'GPU =',tf.test.is_gpu_available())
print('DeepTables version:',dt.__version__)
```

```python
seed = 42
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_everything(seed=seed)
```

```python
constant_cols = pd.read_csv('/kaggle/input/um-gps-of-mcts-variants-constant-columns/constant_columns.csv').columns.to_list()
target_col = 'utility_agent1'
game_col = 'GameRulesetName'
game_rule_cols = ['EnglishRules', 'LudRules']
output_cols = ['num_wins_agent1', 'num_draws_agent1', 'num_losses_agent1']
dropped_cols = ['Id'] + constant_cols + game_rule_cols + output_cols
agent_cols = ['agent1', 'agent2']

def preprocess_data(df): 
    df = df.drop(filter(lambda x: x in df.columns, dropped_cols))
    if CFG.split_agent_features:
        for col in agent_cols:
            df = df.with_columns(pl.col(col).str.split(by="-").list.to_struct(fields=lambda idx: f"{col}_{idx}")).unnest(col).drop(f"{col}_0")
    df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in df.columns if col[:6] in agent_cols])            
    df = df.with_columns([pl.col(col).cast(pl.Float32) for col in df.columns if col[:6] not in agent_cols and col != game_col])
    df = df.to_pandas()
    print(f'Data shape: {df.shape}\n')
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()
    non_cat_cols = df.select_dtypes(exclude=['category']).columns.tolist()
    num_cols = [num for num in non_cat_cols if num not in [target_col, game_col]]
    return df, cat_cols, num_cols
```

```python
# https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-790/notebook
LR_START = 1e-4
LR_MAX = 1e-3
LR_MIN = 1e-3
LR_RAMPUP_EPOCHS = 1
LR_SUSTAIN_EPOCHS = 0
EPOCHS = 7

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = EPOCHS - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN    
    return lr

rng = [i for i in range(EPOCHS)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, '-o')
plt.xlabel('Epoch'); plt.ylabel('LR')
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}". \
      format(lr_y[0], max(lr_y), lr_y[-1]))
LR_Scheduler = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
```

```python
class CFG:
    train_path = '/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv'
    split_agent_features = True
    scaler = MinMaxScaler()  # Scaler or None
    
    nn = True
    ctb = True
    ens_weights = {'nn': 0.40, 'ctb': 0.60}  # While nn = True and ctb = True
    
    folds = 6
    
    epochs = 7
    batch_size = 128
    LR_Scheduler = [LR_Scheduler]
    optimizer = Adam(learning_rate=1e-3)
    conf = ModelConfig(auto_imputation=False,
                       auto_discrete=False,
                       auto_discard_unique=True,
                       categorical_columns='auto',
                       apply_gbm_features=True,
                       fixed_embedding_dim=True,
                       embeddings_output_dim=4,
                       embedding_dropout=0.2,
                       nets=['dnn_nets'] + ['fm_nets'] + ['cin_nets'],
                       dnn_params={
                           'hidden_units': ((1024, 0.0, True),
                                            (512, 0.0, True),
                                            (256, 0.0, True),
                                            (128, 0.0, True)),
                           'dnn_activation': 'relu',
                       },
                       stacking_op='concat',
                       output_use_bias=False,
                       optimizer=optimizer,
                       task='regression',
                       loss='auto',
                       metrics=["RootMeanSquaredError"],
                       earlystopping_patience=1,
                       )

    ctb_params = dict(iterations=5000,
                      learning_rate=0.04,
                      depth=10,
                      l2_leaf_reg=3,
                      random_strength=0.3,
                      bagging_temperature=0.5,
                      loss_function='RMSE',
                      eval_metric = 'RMSE',
                      metric_period=500,
                      od_type='Iter',
                      od_wait=100,
                      task_type='GPU',
                      allow_writing_files=False,
                      )
```

```python
def train(data, cat_cols, num_cols, scaler):
    cv = GroupKFold(n_splits=CFG.folds)
    groups = data[game_col]
    X = data.drop([target_col, game_col], axis=1)
    y = data[target_col]
    oof = np.zeros(len(data))
    nn_models = []
    ctb_models = []
    
    print('nn = '+str(CFG.nn))
    print('ctb = '+str(CFG.ctb),'\n')
    
    for fi, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
        print("#"*25)
        print(f"### Fold {fi+1}/{CFG.folds} ...")
        print("#"*25)
        
        os.makedirs(f"/kaggle/working/nn_models/fold{fi}", exist_ok=True)
        os.makedirs(f"/tmp/workdir/kaggle/working/nn_models/fold{fi}", exist_ok=True)
        os.makedirs(f"/kaggle/working/ctb_models/fold{fi}", exist_ok=True)

        if CFG.nn == True and CFG.ctb == False:
            print('\n',"nn only model training.",'\n')
            K.clear_session()
            nn_model = DeepTable(config=CFG.conf)
            nn_model.fit(X.iloc[train_idx], y.iloc[train_idx],
                      validation_data=(X.iloc[valid_idx], y.iloc[valid_idx]),
                      callbacks=CFG.LR_Scheduler,
                      batch_size=CFG.batch_size, epochs=CFG.epochs, verbose=2)
            nn_models.append(nn_model)
            
            # Save model
            nn_model.save(f'/kaggle/working/nn_models/fold{fi}')
            os.system(f'cp -r /kaggle/working/nn_models/fold{fi}/* /tmp/workdir/kaggle/working/nn_models/fold{fi}/')
        
            # Avoid some errors
            with K.name_scope(CFG.optimizer.__class__.__name__):
                for j, var in enumerate(CFG.optimizer.weights):
                    name = 'variable{}'.format(j)
                    CFG.optimizer.weights[j] = tf.Variable(var, name=name)
            CFG.conf = CFG.conf._replace(optimizer=CFG.optimizer)

            oof_preds = nn_model.predict(X.iloc[valid_idx], verbose=1, batch_size=512).flatten()
            rmse = np.round(np.sqrt(np.mean((oof_preds - y.iloc[valid_idx])**2)),4)
            print(f'{Fore.GREEN}{Style.BRIGHT}\nFold {fi+1} | rmse: {rmse}\n')
            if fi<CFG.folds: oof[valid_idx] = oof_preds
            else: oof[valid_idx] += oof_preds
                
        elif CFG.nn == False and CFG.ctb == True:
            print('\n',"ctb only model training.",'\n')
            X_ = X.copy()
            if CFG.scaler is not None:
                print(f'\nInverse scaling {len(num_cols)} numerical cols for ctb.')
                X_[num_cols] = scaler.inverse_transform(X_[num_cols])
            ctb_model = CatBoostRegressor(**CFG.ctb_params)
            ctb_model.fit(X_.iloc[train_idx], y.iloc[train_idx],
                          eval_set=[(X_.iloc[valid_idx], y.iloc[valid_idx])],
                          cat_features=cat_cols, use_best_model=True)
            ctb_models.append(ctb_model)
            
            ctb_model.save_model(f'/kaggle/working/ctb_models/fold{fi}/ctb_model.cbm')

            oof_preds = ctb_model.predict(X_.iloc[valid_idx])
            rmse = np.round(np.sqrt(np.mean((oof_preds - y.iloc[valid_idx])**2)),4)
            print(f'\nFold {fi+1} | rmse: {rmse}\n')
            if fi<CFG.folds: oof[valid_idx] = oof_preds
            else: oof[valid_idx] += oof_preds
                
        elif CFG.nn == True and CFG.ctb == True:
            print('\n',"nn & ctb model training.",'\n')
            K.clear_session()
            nn_model = DeepTable(config=CFG.conf)
            nn_model.fit(X.iloc[train_idx], y.iloc[train_idx],
                      validation_data=(X.iloc[valid_idx], y.iloc[valid_idx]),
                      callbacks=CFG.LR_Scheduler,
                      batch_size=CFG.batch_size, epochs=CFG.epochs, verbose=2)
            nn_models.append(nn_model)
            
            # Save model
            nn_model.save(f'/kaggle/working/nn_models/fold{fi}')
            os.system(f'cp -r /kaggle/working/nn_models/fold{fi}/* /tmp/workdir/kaggle/working/nn_models/fold{fi}/')

            # Avoid some errors
            with K.name_scope(CFG.optimizer.__class__.__name__):
                for j, var in enumerate(CFG.optimizer.weights):
                    name = 'variable{}'.format(j)
                    CFG.optimizer.weights[j] = tf.Variable(var, name=name)
            CFG.conf = CFG.conf._replace(optimizer=CFG.optimizer)
            
            X_ = X.copy()
            if CFG.scaler is not None:
                print(f'\nInverse scaling {len(num_cols)} numerical cols for ctb.')
                X_[num_cols] = scaler.inverse_transform(X_[num_cols])
            ctb_model = CatBoostRegressor(**CFG.ctb_params)
            ctb_model.fit(X_.iloc[train_idx], y.iloc[train_idx],
                          eval_set=[(X_.iloc[valid_idx], y.iloc[valid_idx])],
                          cat_features=cat_cols, use_best_model=True)
            ctb_models.append(ctb_model)
            
            ctb_model.save_model(f'/kaggle/working/ctb_models/fold{fi}/ctb_model.cbm')

            oof_preds = CFG.ens_weights['nn'] * nn_model.predict(X.iloc[valid_idx],
                                                                 verbose=1, batch_size=512).flatten() + \
                        CFG.ens_weights['ctb'] * ctb_model.predict(X_.iloc[valid_idx])
            rmse = np.round(np.sqrt(np.mean((oof_preds - y.iloc[valid_idx])**2)),4)
            print(f'{Fore.GREEN}{Style.BRIGHT}\nFold {fi+1} | rmse: {rmse}\n')
            if fi<CFG.folds: oof[valid_idx] = oof_preds
            else: oof[valid_idx] += oof_preds
                
        else:
            raise ValueError("No model selected in CFG.")
    
    rmse = np.round(np.sqrt(np.mean((oof - y)**2)),4)
    print(f'{Fore.BLUE}{Style.BRIGHT}Overall CV rmse: {rmse}\n')
    if CFG.nn==True: plot_model(nn_model.get_model().model)
    return nn_models, ctb_models


def infer(data, nn_models, ctb_models, num_cols, scaler):
    if CFG.nn == True and CFG.ctb == False:
        return np.mean([model.predict(data, verbose=1, batch_size=512).flatten()
                                            for model in nn_models], axis=0)
    elif CFG.nn == False and CFG.ctb == True:
        if CFG.scaler is not None:
            print(f'Inverse scaling {len(num_cols)} numerical cols for ctb.\n')
            data[num_cols] = scaler.inverse_transform(data[num_cols])
        return np.mean([model.predict(data) for model in ctb_models], axis=0)
    
    elif CFG.nn == True and CFG.ctb == True:
        data_ = data.copy()
        if CFG.scaler is not None:
            print(f'Inverse scaling {len(num_cols)} numerical cols for ctb.\n')
            data_[num_cols] = scaler.inverse_transform(data_[num_cols])
        return CFG.ens_weights['nn'] * np.mean([model.predict(data, verbose=1, batch_size=512).flatten()
                                                for model in nn_models], axis=0) + \
               CFG.ens_weights['ctb'] * np.mean([model.predict(data_) for model in ctb_models], axis=0)
    else:
        raise ValueError("No model selected in CFG.")
```

```python
%%time
run_i = 0
scaler = CFG.scaler
def predict(test_data, submission):
    global run_i, scaler, nn_models, ctb_models
    if run_i == 0:
        train_df = pl.read_csv(CFG.train_path)
        train_df, cat_cols, num_cols = preprocess_data(train_df)
        if scaler is not None:
            print(f'Scaling {len(num_cols)} numerical cols.\n')
            train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        nn_models, ctb_models = train(train_df, cat_cols, num_cols, scaler)
    run_i += 1
    test_df, cat_cols, num_cols = preprocess_data(test_data)
    test_df = test_df.drop(columns=game_col)
    if scaler is not None:
        print(f'Scaling {len(num_cols)} numerical cols.\n')
        test_df[num_cols] = scaler.transform(test_df[num_cols])
    return submission.with_columns(pl.Series(target_col, infer(test_df, nn_models, ctb_models,
                                                               num_cols, scaler)))

inference_server = kaggle_evaluation.mcts_inference_server.MCTSInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        ('/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv',
         '/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv'))
```

```python
# Check load model
def load_model(paths):
    models = []
    for fold in sorted(os.listdir(paths)):
        path = os.path.join(paths, fold)
        for file in os.listdir(path):
            if file.endswith('.h5'):
                models.append(DeepTable.load(path, file))
            elif file.endswith('.cbm'):
                print('Load model from:', path+'/'+file)
                models.append(CatBoostRegressor().load_model(path+'/'+file))
    return models

nn_models = load_model("/kaggle/working/nn_models")
print('nn_models:', nn_models,'\n')

ctb_models = load_model("/kaggle/working/ctb_models")
print('ctb_models:', ctb_models)
```
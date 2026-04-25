# NeurIPS_AutoGluon_predict2

- **Author:** MT
- **Votes:** 276
- **Ref:** mtinti/neurips-autogluon-predict2
- **URL:** https://www.kaggle.com/code/mtinti/neurips-autogluon-predict2
- **Last run:** 2025-07-03 19:40:17.240000

---

🚀 Fast Baseline with AutoGluon & External Features

- This notebook provides a simple yet powerful "no-brainer" solution

We're leveraging AutoGluon for automated machine learning, combined with the excellent feature set from Dmitry Uarov's popular notebook:
https://www.kaggle.com/code/dmitryuarov/neurips-baseline-external-data

```python
!cp -r /kaggle/input/autogluon-package/* /kaggle/working/
!pip install -f --quiet --no-index --find-links='/kaggle/input/autogluon-package' 'autogluon.tabular-1.3.1-py3-none-any.whl'
```

```python
!cp -r /kaggle/input/scikit-package/* /kaggle/working/
!pip install -f --quiet --no-index --find-links='/kaggle/input/scikit-package' 'scikit_learn-1.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
```

```python
from autogluon.tabular import TabularDataset, TabularPredictor
```

```python
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl

import gc
import pickle

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

import lightgbm as lgb

from sklearn.model_selection import KFold

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
class CFG:
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    SEED = 42
    FOLDS = 5
    PATH = '/kaggle/input/open-polymer-challenge/'

train = pd.read_csv(CFG.PATH + 'train.csv')
test = pd.read_csv(CFG.PATH + 'test.csv')

def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan

train['SMILES'] = train['SMILES'].apply(lambda s: make_smile_canonical(s))
test['SMILES'] = test['SMILES'].apply(lambda s: make_smile_canonical(s))
# https://www.kaggle.com/datasets/minatoyukinaxlisa/tc-smiles
data_tc = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
data_tc = data_tc.rename(columns={'TC_mean': 'Tc'})

# https://springernature.figshare.com/articles/dataset/dataset_with_glass_transition_temperature/24219958?file=42507037
data_tg2 = pd.read_csv('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv', usecols=['SMILES', 'Tg (C)'])
data_tg2 = data_tg2.rename(columns={'Tg (C)': 'Tg'})

# https://www.sciencedirect.com/science/article/pii/S2590159123000377#ec0005
data_tg3 = pd.read_excel('/kaggle/input/smiles-extra-data/data_tg3.xlsx')
data_tg3 = data_tg3.rename(columns={'Tg [K]': 'Tg'})
data_tg3['Tg'] = data_tg3['Tg'] - 273.15

# https://github.com/Duke-MatSci/ChemProps
data_dnst = pd.read_excel('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
data_dnst = data_dnst.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
data_dnst['SMILES'] = data_dnst['SMILES'].apply(lambda s: make_smile_canonical(s))
data_dnst = data_dnst[(data_dnst['SMILES'].notnull())&(data_dnst['Density'].notnull())&(data_dnst['Density'] != 'nylon')]
data_dnst['Density'] = data_dnst['Density'].astype('float64')
data_dnst['Density'] -= 0.118

def add_extra_data(df_train, df_extra, target):
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    df_extra['SMILES'] = df_extra['SMILES'].apply(lambda s: make_smile_canonical(s))
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Make priority target value from competition's df
    for smile in df_train[df_train[target].notnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            cross_smiles.remove(smile)

    # Imput missing values for competition's SMILES
    for smile in cross_smiles:
        df_train.loc[df_train['SMILES']==smile, target] = df_extra[df_extra['SMILES']==smile][target].values[0]
    
    df_train = pd.concat([df_train, df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]], axis=0).reset_index(drop=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'\nFor target "{target}" added {n_samples_after-n_samples_before} new samples!')
    print(f'New unique SMILES: {len(unique_smiles_extra)}')
    return df_train

train = add_extra_data(train, data_tc, 'Tc')
train = add_extra_data(train, data_tg2, 'Tg')
train = add_extra_data(train, data_tg3, 'Tg')
train = add_extra_data(train, data_dnst, 'Density')

print('\n'*3, '--- SMILES for training ---', )
for t in CFG.TARGETS:
    print(f'"{t}": {len(train[train[t].notnull()])}')
```

```python
useless_cols = [    
    # Nan data
    'BCUT2D_MWHI',
    'BCUT2D_MWLOW',
    'BCUT2D_CHGHI',
    'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI',
    'BCUT2D_LOGPLOW',
    'BCUT2D_MRHI',
    'BCUT2D_MRLOW',

    # Constant data
    'NumRadicalElectrons',
    'SMR_VSA8',
    'SlogP_VSA9',
    'fr_barbitur',
    'fr_benzodiazepine',
    'fr_dihydropyridine',
    'fr_epoxide',
    'fr_isothiocyan',
    'fr_lactam',
    'fr_nitroso',
    'fr_prisulfonamd',
    'fr_thiocyan',

    # High correlated data >0.95
    'MaxEStateIndex',
    'HeavyAtomMolWt',
    'ExactMolWt',
    'NumValenceElectrons',
    'Chi0',
    'Chi0n',
    'Chi0v',
    'Chi1',
    'Chi1n',
    'Chi1v',
    'Chi2n',
    'Kappa1',
    'LabuteASA',
    'HeavyAtomCount',
    'MolMR',
    'Chi3n',
    'BertzCT',
    'Chi2v',
    'Chi4n',
    'HallKierAlpha',
    'Chi3v',
    'Chi4v',
    'MinAbsPartialCharge',
    'MinPartialCharge',
    'MaxAbsPartialCharge',
    'FpDensityMorgan2',
    'FpDensityMorgan3',
    'Phi',
    'Kappa3',
    'fr_nitrile',
    'SlogP_VSA6',
    'NumAromaticCarbocycles',
    'NumAromaticRings',
    'fr_benzene',
    'VSA_EState6',
    'NOCount',
    'fr_C_O',
    'fr_C_O_noCOO',
    'NumHDonors',
    'fr_amide',
    'fr_Nhpyrrole',
    'fr_phenol',
    'fr_phenol_noOrthoHbond',
    'fr_COO2',
    'fr_halogen',
    'fr_diazo',
    'fr_nitro_arom',
    'fr_phos_ester'
]

def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList if desc[0] not in useless_cols]

def compute_graph_features(smiles, graph_feats):
    mol = Chem.MolFromSmiles(smiles)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)

    graph_feats['graph_diameter'].append(nx.diameter(G) if nx.is_connected(G) else 0)
    graph_feats['avg_shortest_path'].append(nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
    graph_feats['num_cycles'].append(len(list(nx.cycle_basis(G))))

def preprocessing(df):
    desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
    descriptors = [compute_all_descriptors(smi) for smi in df['SMILES'].to_list()]

    graph_feats = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': []}
    for smile in df['SMILES']:
         compute_graph_features(smile, graph_feats)
        
    result = pd.concat(
        [
            pd.DataFrame(descriptors, columns=desc_names),
            pd.DataFrame(graph_feats)
        ],
        axis=1
    )

    result = result.replace([-np.inf, np.inf], np.nan)
    return result

train = pd.concat([train, preprocessing(train)], axis=1)
test = pd.concat([test, preprocessing(test)], axis=1)

# Find constant columns for each target
all_features = train.columns[7:].tolist()
features = {}
for target in CFG.TARGETS:
    const_descs = []
    for col in train.columns.drop(CFG.TARGETS):
        if train[train[target].notnull()][col].nunique() == 1:
            const_descs.append(col)
    features[target] = [f for f in all_features if f not in const_descs]

print(train.shape)
train['Ipc']=np.log10(train['Ipc'])  
for n in train.columns[7:]:
    train[n]=train[n].replace(-np.inf,np.nan)
    train[n]=train[n].replace(np.inf,np.nan)    
    train[n].fillna(train[n].mean())
  
print(train.shape)
test['Ipc']=np.log10(test['Ipc'])
for n in test.columns[7:]:
    train[n]=train[n].replace(-np.inf,np.nan)
    train[n]=train[n].replace(np.inf,np.nan)      
    test[n].fillna(train[n].mean())

train.to_pickle('train.pkl')
test.to_pickle('test.pkl')
with open("features.pkl", 'wb') as f:
    pickle.dump(features, f)
```

```python
model_Tg = TabularPredictor.load('/kaggle/input/neurips-train-autogluon-tg2/model/')
pred_Tg = model_Tg.predict(test)
test['Tg']=pred_Tg
pred_Tg.head()
```

```python
model_Rg = TabularPredictor.load('/kaggle/input/neurips-train-autogluon-rg2/model/')
pred_Rg = model_Rg.predict(test)
test['Rg']=pred_Rg
pred_Rg.head()
```

```python
model_FFV = TabularPredictor.load('/kaggle/input/neurips-train-autogluon-ffv2/model/')
pred_FFV = model_FFV.predict(test)
test['FFV']=pred_FFV
pred_FFV.head()
```

```python
#submission.head()
```

```python
model_Density = TabularPredictor.load('/kaggle/input/neurips-train-autogluon-density2/model/')
pred_Density = model_Density.predict(test)
test['Density']=pred_Density
pred_Density.head()
```

```python
model_Tc = TabularPredictor.load('/kaggle/input/neurips-train-autogluon-tc2/model/')
pred_Tc = model_Tc.predict(test)
test['Tc']=pred_Tc
pred_Tc.head()
```

```python
for t in CFG.TARGETS:
    for s in train[train[t].notnull()]['SMILES']:
        if s in test['SMILES'].tolist():
            test.loc[test['SMILES']==s, t] = train[train['SMILES']==s][t].values[0]
```

```python
test[['id'] + CFG.TARGETS].to_csv('submission.csv', index=False)
```

```python
!head 'submission.csv'
```
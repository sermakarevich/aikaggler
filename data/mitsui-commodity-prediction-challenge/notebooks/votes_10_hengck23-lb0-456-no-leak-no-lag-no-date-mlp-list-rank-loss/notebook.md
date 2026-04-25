# lb0.456-no leak,no-lag,no-date: mlp list rank loss

- **Author:** hengck23
- **Votes:** 174
- **Ref:** hengck23/lb0-456-no-leak-no-lag-no-date-mlp-list-rank-loss
- **URL:** https://www.kaggle.com/code/hengck23/lb0-456-no-leak-no-lag-no-date-mlp-list-rank-loss
- **Last run:** 2025-08-12 11:04:28.497000

---

```python
#define model

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self, input_dim=557, output_dim=424):
		super(Net, self).__init__()
		dim=256

		self.mlp = nn.Sequential(
			nn.BatchNorm1d(input_dim),
			nn.Linear(input_dim, dim),
			nn.ReLU(),   
			nn.Linear(dim, dim),
			nn.ReLU(),
			nn.Linear(dim, output_dim)   
		)

	def forward(self, x):
		y = self.mlp(x)
		return y
print('MODEL OK!!!')
```

```python
import numpy as np 
import polars as pl
import os

import pandas as pd
pd.set_option('future.no_silent_downcasting', False)

DEVICE='cpu'
TARGET_COUNT=424
DATA_DIR = '/kaggle/input/mitsui-commodity-prediction-challenge'


net = Net()
print(net.load_state_dict(
    torch.load('/kaggle/input/hengck23-lme-data-demo/best_model.v1.pth',
            map_location=torch.device('cpu'),
            weights_only=False)
))
net = net.to(DEVICE)
net.eval()


def predict(test, lag1,lag2,lag3,lag4):

	test_df = test.to_pandas()
	lag_df = pd.concat([
		lag1.to_pandas().drop('date_id', axis=1).drop('label_date_id', axis=1),
		lag2.to_pandas().drop('date_id', axis=1).drop('label_date_id', axis=1),
		lag3.to_pandas().drop('date_id', axis=1).drop('label_date_id', axis=1),
		lag4.to_pandas().drop('date_id', axis=1).drop('label_date_id', axis=1),
	], axis=1) #not used


	data_df = test_df.drop('is_scored', axis=1) 
	data_df.fillna(-1, inplace=True)

	#----
	date_id = test_df.iloc[0].date_id

	print('called predict() ...', date_id)

	x = data_df.iloc[:1,1:].values
	x = torch.from_numpy(x).float().to(DEVICE)
	with torch.no_grad():
		p = net(x)
	predict = p.cpu().detach().numpy().reshape(-1)
	return pl.DataFrame({f'target_{i}': predict[i] for i in range(TARGET_COUNT)})

# ====================================================
if 1:
	# Serve Model (Locally or on Hidden Test)
	import kaggle_evaluation.mitsui_inference_server


	inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)
	if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
		inference_server.serve()
	else:
		inference_server.run_local_gateway((DATA_DIR,))

print('SUBMIT OK!!!')
```

```python
'''
# implementation of listwise rank loss
 


# load data ===========================================
CUT = 1827
KAGGLE_DIR =\
	'.../data/mitsui-commodity-prediction-challenge'

data_df = pd.read_csv(f'{KAGGLE_DIR}/train.csv')
label_df = pd.read_csv(f'{KAGGLE_DIR}/train_labels.csv')
#train_df(1917, 558)    label_df(1917, 425)
data_df.fillna(-1, inplace=True)


train_data_df = data_df[:CUT].reset_index(drop=True)
train_label_df = label_df[:CUT].reset_index(drop=True)
valid_data_df = data_df[CUT:].reset_index(drop=True)
valid_label_df = label_df[CUT:].reset_index(drop=True)

......

x,y = get_batch()
x = torch.from_numpy(x).float().cuda()
y = torch.from_numpy(y).float().cuda()
truth = y

mask = ~torch.isnan(y)
predict = net(x)

#----
mse_loss = F.mse_loss(predict[mask], truth[mask])

#----
rank_loss=0
for b in range(B):
    p = predict[b][mask[b]]
    y = truth[b][mask[b]]

    pp = F.softmax(p,0)
    py = F.softmax(y,0)
    pp = torch.clamp(pp, 1-4, 1)
    l = -(py * torch.log(pp)).sum()
    rank_loss += l
rank_loss = rank_loss/B

#-----

optimizer.zero_grad()
(mse_loss+rank_loss).backward()
optimizer.step()

'''
```
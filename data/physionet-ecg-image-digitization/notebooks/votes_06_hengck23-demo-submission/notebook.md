# demo submission

- **Author:** hengck23
- **Votes:** 243
- **Ref:** hengck23/demo-submission
- **URL:** https://www.kaggle.com/code/hengck23/demo-submission
- **Last run:** 2025-11-15 01:57:40.890000

---

```python
try:
    import cc3d
except:
    #https://pypi.org/project/connected-components-3d/
    #!pip install connected-components-3d

    !ls /kaggle/input/hengck23-demo-submit-physionet/setup
    !pip install connected-components-3d --no-index --find-links=file:///kaggle/input/hengck23-demo-submit-physionet/setup/

import cc3d
import cv2
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
import shutil

import sys
sys.path.append('/kaggle/input/hengck23-demo-submit-physionet')

print('import ok!!!')
```

```python
MODE   = 'submit'  # submit  local fake
DEVICE = 'cuda'
FLOAT_TYPE = torch.float16 #torch.bfloat16
FAIL_ID = []

KAGGLE_DIR = \
	'/kaggle/input/physionet-ecg-image-digitization'
WEIGHT_DIR = \
	'/kaggle/input/hengck23-demo-submit-physionet/weight'
OUT_DIR = \
    f'/kaggle/working/output-{MODE}'

def make_test_fake_df(): 
    valid_df = pd.read_csv(f'{KAGGLE_DIR}/train.csv')
    valid_df.loc[:,'id']=valid_df['id'].astype(str) 
    fake_test_df=[]
    for i,d in valid_df.iterrows():
        #if i==4: break
        image_id = d['id']
    
        truth_df = pd.read_csv(f'{KAGGLE_DIR}/train/{image_id}/{image_id}.csv')
        non_nan_count = truth_df.count()
        #print(i,image_id,non_nan_count)
        #print(non_nan_count.index)
    
        #lead	fs	number_of_rows 
        this_df = pd.DataFrame({
            'id':image_id ,
            'lead':non_nan_count.index,
            'fs': d['fs'],
            'number_of_rows':non_nan_count.values 
        })
        fake_test_df.append(this_df)
        if i==0: print(this_df)
    fake_test_df = pd.concat(fake_test_df)
    return fake_test_df


# set valid/test data
if MODE == 'local':
	from sample_list import ERROR_ID
	valid_df = pd.read_csv(f'{KAGGLE_DIR}/train.csv')
	valid_df['id']=valid_df['id'].astype(str)

	valid_id = [
		f'{image_id}-{type_id}' for image_id in ERROR_ID
		#f'{image_id}-{type_id}' for image_id in valid_df['id'].values[500:]
		for type_id in ['0001', '0003', '0004', '0005', '0006', '0009', '0010', '0011', '0012']
	]
	valid_id = [
        '11842146-0012','144746082-0009','225208096-0006', '2289894144-0012','1617515072-0006',
        '2289894144-0010','2566168201-0009', '2659677149-0011'
    ]
    
if MODE == 'submit':
	valid_df = pd.read_csv(f'{KAGGLE_DIR}/test.csv')
	valid_df['id']=valid_df['id'].astype(str) 
	valid_id = valid_df['id'].unique().tolist()

if MODE == 'fake':
	valid_df = make_test_fake_df()
	valid_df['id']=valid_df['id'].astype(str) 
	valid_id = valid_df['id'].unique().tolist()

#--------------------------------------

def read_image(sample_id):
    if MODE == 'local':
        image_id, type_id = sample_id.split('-')
        image = cv2.imread(f'{KAGGLE_DIR}/train/{image_id}/{image_id}-{type_id}.png', cv2.IMREAD_COLOR_RGB)
        return image
    if MODE == 'submit':
        image_id = sample_id
        image = cv2.imread(f'{KAGGLE_DIR}/test/{image_id}.png', cv2.IMREAD_COLOR_RGB)
        return image
    if MODE == 'fake':
        image_id = sample_id 
        type_id = ['0001', '0003', '0004', '0005', '0006', '0009', '0010', '0011', '0012'][
            int(image_id)%9
        ] 
        image = cv2.imread(f'{KAGGLE_DIR}/train/{image_id}/{image_id}-{type_id}.png', cv2.IMREAD_COLOR_RGB)
        return image

def read_sampling_length(sample_id):
	if MODE == 'local':
		image_id, type_id = sample_id.split('-')
		d = valid_df[valid_df['id']==image_id].iloc[0]
		length = d.sig_len
		return length
	if MODE == 'submit':
		image_id = sample_id
		d = valid_df[
			(valid_df['id']==image_id) & (valid_df['lead']=='II')
		].iloc[0]
		length = d.number_of_rows
		return length
	if MODE == 'fake':
		image_id = sample_id
		d = valid_df[
			(valid_df['id']==image_id) & (valid_df['lead']=='II')
		].iloc[0]
		length = d.fs*10  #d.number_of_rowsd.number_of_rows
		return length

#valid_id = valid_id[:300]
print('valid_id:', len(valid_id))
print('\t', valid_id[:3], '...')
print('setting ok!!!\n')
```

```python
# stage0
print('*** STARTING STAGE0 ***')

from stage0_model import Net as Stage0Net
from stage0_common import *

os.makedirs(f'{OUT_DIR}/normalised', exist_ok=True)

def run_stage0():
	stage0_net = Stage0Net(pretrained=False)
	stage0_net = load_net(stage0_net, f'{WEIGHT_DIR}/stage0-last.checkpoint.pth')
	stage0_net.to(DEVICE)

	start_timer = timer()
	for n, sample_id in enumerate(valid_id):
		timestamp = time_to_str(timer() - start_timer, 'sec')
		print(f'\r\t {n:4d} {sample_id}', timestamp, end='', flush=True)

		image = read_image(sample_id)
		batch = image_to_batch(image)

		with torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
			with torch.no_grad():
				output = stage0_net(batch)

				try:
					rotated, keypoint = output_to_predict(image, batch, output)
					normalised, keypoint, homo = normalise_by_homography(rotated, keypoint)
					# ---
					cv2.imwrite(f'{OUT_DIR}/normalised/{sample_id}.norm.png', cv2.cvtColor(normalised, cv2.COLOR_RGB2BGR))
					np.save(f'{OUT_DIR}/normalised/{sample_id}.homo.npy', homo)
				except:
					FAIL_ID.append(sample_id)

		torch.cuda.empty_cache()
		if n<10: # optional: show results
			overlay = draw_results_stage0(rotated, keypoint)
			print('')
			print('demo results for stage0--------------')
			print(sample_id)
			plt.imshow(image);plt.show()
			plt.imshow(overlay);plt.show()
			plt.imshow(normalised);plt.show()
			
	print('')

run_stage0()
print('FAIL_ID:', FAIL_ID)
print('run_stage0() ok!!!\n')
```

```python
# stage1
print('*** STARTING STAGE1 ***')

from stage1_model import Net as Stage1Net
from stage1_common import *

os.makedirs(f'{OUT_DIR}/rectified', exist_ok=True)

def run_stage1():
	stage1_net = Stage1Net(pretrained=False)
	stage1_net = load_net(stage1_net, f'{WEIGHT_DIR}/stage1-last.checkpoint.pth')
	stage1_net.to(DEVICE)

	start_timer = timer()
	for n, sample_id in enumerate(valid_id):
		timestamp = time_to_str(timer() - start_timer, 'sec')
		print(f'\r\t {n:4d} {sample_id}', timestamp, end='', flush=True)
		if sample_id in FAIL_ID: continue

		image = cv2.imread(f'{OUT_DIR}/normalised/{sample_id}.norm.png', cv2.IMREAD_COLOR_RGB)
		batch = {
			'image': torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0),
		}
		num_tta = 1

		with torch.amp.autocast('cuda', dtype=FLOAT_TYPE): #torch.bfloat16
			with torch.no_grad():
				output = stage1_net(batch)

				try:
					gridpoint_xy, more = output_to_predict(image, batch, output)
					rectified = rectify_image(image, gridpoint_xy)
					# ---
					cv2.imwrite(f'{OUT_DIR}/rectified/{sample_id}.rect.png', cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR))
					np.save(f'{OUT_DIR}/rectified/{sample_id}.gridpoint_xy.npy',gridpoint_xy)
				except:
					FAIL_ID.append(sample_id)

		torch.cuda.empty_cache()
		if n<10: # optional: show results
			overlay = draw_mapping(image, gridpoint_xy) #
			ghfiltered, gvfiltered = draw_results_stage1(more)
            
			
			print('')
			print('demo results for stage1--------------')
			print(sample_id)
			plt.imshow(overlay);plt.show()
			plt.imshow(gvfiltered);plt.show()
			plt.imshow(ghfiltered);plt.show()
			plt.imshow(rectified);plt.show()
             
	print('')

run_stage1()
print('FAIL_ID:', FAIL_ID)
print('run_stage1() ok!!!\n')
```

```python
# stage2
print('*** STARTING STAGE2 ***')

from stage2_model import Net as Stage2Net, prob_to_series_by_max
from stage2_common import *

os.makedirs(f'{OUT_DIR}/digitalised', exist_ok=True)
#os.makedirs(f'{OUT_DIR}/debug', exist_ok=True)

def run_stage2():
	stage2_net = Stage2Net(pretrained=False)
	stage2_net = load_net(
		stage2_net,
		f'{WEIGHT_DIR}/stage2-00005810.checkpoint.pth'
	)
	stage2_net.to(DEVICE)

	start_timer = timer()
	for n, sample_id in enumerate(valid_id):
		# sample_id =\
		# 	'1445349505-0006' #'1617515072-0006'

		timestamp = time_to_str(timer() - start_timer, 'sec')
		print(f'\r\t {n:4d} {sample_id}', timestamp, end='', flush=True)
		if sample_id in FAIL_ID: continue

		image = cv2.imread(f'{OUT_DIR}/rectified/{sample_id}.rect.png', cv2.IMREAD_COLOR_RGB)
		length = read_sampling_length(sample_id) #5120

		# at rectified coord frame: H, W = 1700, 2200
		x0, x1 = 0, 2176
		y0, y1 = 0, 1696
		zero_mv = [ 703.5, 987.5, 1271.5, 1531.5 ]
		mv_to_pixel = 79.0
		t0,t1 = timespan = 118, 2080

		crop = image[y0:y1, x0:x1]
		batch = {
			'image': torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).unsqueeze(0),
		}
		with torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
			with torch.no_grad():
				output = stage2_net(batch)

		#---
		try:
		#if 1:
			pixel = output['pixel'].float().data.cpu().numpy()[0]
			series_in_pixel = pixel_to_series(pixel[..., t0:t1], zero_mv, length)
			series = (np.array(zero_mv).reshape(4, 1) - series_in_pixel) / mv_to_pixel
			series = filter_series_by_limits(series)

			# ---
			#cv2.imwrite(f'{OUT_DIR}/digitalised/{sample_id}.lead.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
			np.save(f'{OUT_DIR}/digitalised/{sample_id}.series.npy', series)

		except:
			FAIL_ID.append(sample_id)

		if n<10: # optional: show results
			overlay = draw_lead_pixel(crop, pixel)
			plt.imshow(overlay); plt.show()
	 
			if MODE=='local':
				truth_df = read_truth_series(sample_id,KAGGLE_DIR)
				truth_series = truth_df[['series0','series1','series2','series3',]].values.T

			t = np.arange(len(series[0]))
			fig, axes = plt.subplots(4, 1, figsize=(12, 10))
			for j in range(4):
				snr=0
				axes[j].plot(t, series[j], alpha=1.0, color='blue', linewidth=1, label='predict')
				if MODE=='local':
					axes[j].plot(t, truth_series[j], alpha=0.5, color='red', linewidth=1,label='truth')
					snr = -np_snr(series[j], truth_series[j])

				axes[j].set_title(f'snr {snr:8.3f}')
				axes[j].legend()
			plt.show()
	print('')

run_stage2()
print('FAIL_ID:', FAIL_ID)
print('run_stage2() ok!!!\n')
```

```python
#make sbmission csv
#FAIL_ID=[1053922973, ]
def make_submission():
	print('===========================================')
	print('making submission csv ...')

	submit_df=[]
	gb = valid_df.groupby('id')
	for i,(sample_id, df) in enumerate(gb):
        
		#if sample_id in FAIL_ID:
		#	series_by_lead = {}
		#	for j,d in df.iterrows():
		#		series_by_lead[d.lead] = np.zeros(d.number_of_rows)
        
		try:
			series = np.load(f'{OUT_DIR}/digitalised/{sample_id}.series.npy')
			_4_,L = series.shape

            #https://www.kaggle.com/competitions/physionet-ecg-image-digitization/discussion/613179#3306701
            #may be even or odd????
			series_by_lead={}
			for l in range(3):
				lead = [
					['I',   'aVR', 'V1', 'V4'],
					['II',  'aVL', 'V2', 'V5'],
					['III', 'aVF', 'V3', 'V6'],
				][l]

				 

				index = [ 
                    int(round(1*L/4)),
                    int(round(2*L/4)),
                    int(round(3*L/4)),
                ]
				split = np.split(series[l], index)
				#print(length)
				for (k, s) in zip(lead, split):
					series_by_lead[k] = s
					#print(k,len(s))
			series_by_lead['II'] = series[3]
			#print(series_by_lead)
    
		except: 
			series_by_lead = {}
			for j,d in df.iterrows():
				series_by_lead[d.lead] = np.zeros(d.number_of_rows)

		#print('\r\t {sample_id}', end='', flush=True)
		for j,d in df.iterrows():
			#
            
			#assert(len(series_by_lead[d.lead])==d.number_of_rows)
             
            #probably error here ... ???
			series_by_lead[d.lead] = np.concatenate([
                series_by_lead[d.lead], np.zeros_like(series_by_lead[d.lead])
            ])[:d.number_of_rows]

			#print(d.lead, len(series_by_lead[d.lead]),d.number_of_rows)
			assert(len(series_by_lead[d.lead])==d.number_of_rows) 
			print(f'\r\t {i} {sample_id} : {d.lead}', end='', flush=True)

			row_id = [
				f'{sample_id}_{i}_{d.lead}' for i in range(d.number_of_rows)
			]
			this_df = pd.DataFrame({
				'id':row_id,
				'value': series_by_lead[d.lead].astype(np.float32),
			})
			submit_df.append(this_df)

	print('')
	submit_df = pd.concat(submit_df, axis=0, ignore_index=True, sort=False, copy=False)
	print(submit_df)
	submit_df.to_csv('submission.csv',index=False)


if (MODE=='fake')|(MODE=='submit'):
    make_submission()
    print('make_submission() ok!!!\n')
    if MODE=='submit':
        shutil.rmtree(OUT_DIR)
    !ls
    #!rm -rf {OUT_DIR}

'''
fake:
[21618231 rows x 2 columns]
'''
```
# 1 hr fast 2d/3d-unet resnet34d scanner tta 

- **Author:** hengck23
- **Votes:** 295
- **Ref:** hengck23/1-hr-fast-2d-3d-unet-resnet34d-scanner-tta
- **URL:** https://www.kaggle.com/code/hengck23/1-hr-fast-2d-3d-unet-resnet34d-scanner-tta
- **Last run:** 2024-11-22 19:26:34.457000

---

```python
#!pip download connected-components-3d
#!pip download zarr

try:
    import zarr
except: 
    !cp -r '/kaggle/input/hengck-czii-cryo-et-02/wheel_file' '/kaggle/working/'
    !pip install /kaggle/working/wheel_file/asciitree-0.3.3/asciitree-0.3.3
    !pip install --no-index --find-links=/kaggle/working/wheel_file zarr
    !pip install --no-index --find-links=/kaggle/working/wheel_file connected-components-3d

print('PIP INSTALL OK!!!')
```

```python
from datetime import datetime

import pandas as pd
import pytz
print('LOGGING TIME OF START:',  datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), "%Y-%m-%d %H:%M:%S"))

import sys
sys.path.append('/kaggle/input/hengck-czii-cryo-et-02')

from czii_helper import *
from dataset import *
from model import *
import numpy as np
from scipy.optimize import linear_sum_assignment
import glob
import cc3d
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.parallel")

print('IMPORT OK!!!')
```

```python
DATA_KAGGLE_DIR = '/kaggle/input/czii-cryo-et-object-identification'

MODE='submit'

 
if MODE == 'local':
    valid_dir = f'{DATA_KAGGLE_DIR}/train'
    valid_id = ['TS_5_4', 'TS_73_6','TS_99_9'] # 
    #valid_id = ['TS_73_6', ]  # fold2
    #valid_id = ['TS_5_4', 'TS_5_4','TS_5_4',]  # fold0

if MODE == 'submit':
    valid_dir = f'{DATA_KAGGLE_DIR}/test'
    valid_id = glob.glob(f'{valid_dir}/static/ExperimentRuns/*')
    valid_id = [f.split('/')[-1] for f in valid_id]

print('valid_id:', len(valid_id), valid_id)

cfg = dotdict(
    arch='resnet34d',
    checkpoint= \
    '/kaggle/input/hengck-czii-cryo-et-weights-01/resnet34d-scan640-fold0-00000154.pth',
        #'/kaggle/input/hengck-czii-cryo-et-weights-01/resnet18d-aug-rot-scan320-fold0-00005610.pth',
        #'/kaggle/input/hengck-czii-cryo-et-weights-01/resnet34d-scan320-fold0-00003168.pth',
        #'/kaggle/input/hengck-czii-cryo-et-weights-01/resnet34d-simple3.0-00002300.pth',

    threshold={
        'apo-ferritin': 0.05,
        'beta-amylase': 0.05,
        'beta-galactosidase': 0.05,
        'ribosome': 0.05,
        'thyroglobulin': 0.05,
        'virus-like-particle': 0.05,
    },
)

print('MODE:', MODE)
print('SETTING OK!!!')
```

```python
net = Net(pretrained=False, cfg=cfg)
state_dict = torch.load(cfg.checkpoint, map_location=lambda storage, loc: storage, weights_only=True)['state_dict']
print(net.load_state_dict(state_dict, strict=False))
print(net.arch)
print('MODEL OK!!!')
```

```python
def make_weight(h, w, top=10,left=0,bottom=0,right=0):
    weight = np.full((h, w),fill_value=1)

    if top>0:
        wt = np.ones((h, w))
        wt[:top]=np.linspace(0,1,top+1)[1:].reshape(-1,1)
        weight = np.minimum(weight, wt)

    if left>0:
        wt = np.ones((h, w))
        wt[:,:left] = np.linspace(0, 1, left + 1)[1:].reshape(1, -1)
        weight = np.minimum(weight, wt)

    if bottom>0:
        wt = np.ones((h, w))
        wt[-bottom:]=np.linspace(0,1,bottom+1)[1:][::-1].reshape(-1,1)
        weight = np.minimum(weight, wt)

    if right>0:
        wt = np.ones((h, w))
        wt[:,-right:] = np.linspace(0, 1, right + 1)[1:][::-1].reshape(1, -1)
        weight = np.minimum(weight, wt)

    return weight


class Scanner:
    def __init__(self, w,h,d, overlap ):
        csum=np.cumsum(overlap)
        self.xyz = [
            (0, 0, 0),
            (0, 0,   d-csum[0]),
            (0, 0, 2*d-csum[1]),
            (0, 0, 3*d-csum[2]),
            (0, 0, 4*d-csum[3]),
        ]
        self.length = len(self.xyz)
        self.weight = [None for i in range(len(self.xyz))]
        self.weight[0] = make_weight(h=d, w=1, top=0,          left=0, bottom=overlap[0], right=0)
        self.weight[1] = make_weight(h=d, w=1, top=overlap[0], left=0, bottom=overlap[1], right=0)
        self.weight[2] = make_weight(h=d, w=1, top=overlap[1], left=0, bottom=overlap[2], right=0)
        self.weight[3] = make_weight(h=d, w=1, top=overlap[2], left=0, bottom=overlap[3], right=0)
        self.weight[4] = make_weight(h=d, w=1, top=overlap[3], left=0, bottom=0,          right=0)
        self.generator = self.create_generator()

    def create_generator(self):
        for i in range(self.length):
            yield self.weight[i], self.xyz[i]

    def __iter__(self):
        self.generator = self.create_generator()
        return self

    def __next__(self):
        return next(self.generator)


def normalise_by_percentile(data, min=5, max=99):
    min = np.percentile(data,min)
    max = np.percentile(data,max)
    data = (data-min)/(max-min)
    return data
    
def draw_probability(probability, color):
	_6_, D, H, W = probability.shape
	pcolor = np.zeros((_6_, D, H, W, 3), dtype=np.float32)
	for i in range(_6_):
		pcolor[i] += probability[i][..., None] * [[[color[i]]]]

	# ----
	p_max = pcolor.max(0)
	p0 = p_max.max(0)
	p1 = p_max.max(1)
	p2 = p_max.max(2)
	all = np.zeros((H + D, W + D, 3), dtype=np.uint8)
	all[:H, :W] = p0
	all[H:, :W] = p1
	all[:H, W:] = p2.transpose(1, 0, 2)
	all[H] = 255
	all[:, W] = 255
	all = np.clip(all, 0, 255)
	return all



#start here !!!! -------------------------------------------------------
def run_submit(net):  
    
    net.output_type = ['infer']
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net.cuda()
    net.eval()
    
    num_slice = 48
    D, H, W = (184, 630, 630)
    scanner = Scanner(w=1, h=1, d=48, overlap=[14, 14, 14, 14, 14])
    threshold = list(cfg.threshold.values())
    
    with torch.no_grad():
        probability = torch.zeros((7, D, H, W), device='cuda')
        count = torch.zeros((7, D, H, W), device='cuda') 
        scanner.weight=[
            torch.from_numpy(wt.reshape(1, num_slice, 1, 1)).float().cuda() for wt in scanner.weight
        ] 
        threshold = torch.tensor(threshold, device='cuda').reshape(6, 1, 1, 1)

    submit_df = []
    total_time = 0  
    for i,id in enumerate(valid_id):
        start_timer = timer()
        torch.cuda.empty_cache() 
        
        print(i, id, '---------------')
        volume, scale = read_one_data(id, static_dir=f'{valid_dir}/static') 
        volume = normalise_by_percentile(volume)
        D, H, W = volume.shape
        assert ((D, H, W)== (184, 630, 630))
 
        probability.zero_()
        count.zero_()

        
        with torch.amp.autocast('cuda', enabled=True):
            with torch.no_grad():
 
                for weight, (x, y, z) in scanner:
                    print('\r', f'{id}:{(x, y, z)}', end='', flush=True)

                    image = volume[z:z + num_slice]
                    batch = dotdict(
                        image=torch.from_numpy(
                            np.stack([
                                image,
                                np.rot90(image, k=1, axes=(1,2)),
                            ])
                        ),
                    )
                    batch['image']=F.pad(batch['image'],[0,10,0,10])
                    output = net(batch)
                    prob = output['particle'][...,:H,:W]

                    prob0 = prob[0]
                    prob1 = torch.rot90(prob[1], k=-1, dims=(2,3))
                    prob  = (prob0 + prob1)/2

                    probability[:, z:z + num_slice] += weight * prob
                    count[:, z:z + num_slice] += weight
                print('')
                probability = probability / count
                probability0 = probability[1:]
                probability1 = F.interpolate(probability0, scale_factor=0.5, mode='bilinear', align_corners=False)
                #smaller for faster post-processing
        
        binary0 = (probability0 > threshold).data.cpu().numpy()
        binary1 = (probability1 > threshold).data.cpu().numpy()
        location = [np.empty((0,3)) for i in range(6)]

        #1: apo-ferritin, radius=60
        for c in [0]:
            componet = cc3d.connected_components(binary0[c])
            stats = cc3d.statistics(componet)
            zyx = stats['centroids'][1:] * [scale]
            xyz = np.ascontiguousarray(zyx[:, ::-1])
            location[c] = xyz

        # beta-amylase is ignored, rest of particles have radius>=90
        for c in [2,3,4,5]:
            componet = cc3d.connected_components(binary1[c])
            stats = cc3d.statistics(componet)
            zyx = stats['centroids'][1:] * [scale]*[[1,2,2]]
            xyz = np.ascontiguousarray(zyx[:, ::-1])
            location[c] = xyz
        print('location', np.concatenate(location).shape)
        
        for name,xyz in zip(PARTICLE_NAME,location):
            if len(xyz)==0: continue
            submit_df.append(
                pd.DataFrame({'experiment': id, 'particle_type':name,'x':xyz[:,0],'y':xyz[:,1],'z':xyz[:,2]})
            )
        time_taken = timer() - start_timer
        total_time += time_taken
        print(time_to_str(time_taken, 'sec'))

        #debug
        if i==0:
            p = probability0.data.cpu().numpy()
            
            m0 = np.clip(volume,0,1)
            m0 = m0.mean(0) 
            m0 = np.dstack([m0, m0, m0])

            g0 = np.zeros((H, W, 3), dtype=np.float32)
            for c in [0,1,2]:
                color=PARTICLE[c]['color']
                q = p[c].max(0)[...,None]
                g0 += q*[color]
            g0 = np.clip(g0/255 > 0.1,0,1)
            g0 = 1-(1-m0)*(1-g0)

            g1 = np.zeros((H, W, 3), dtype=np.float32)
            for c in [3,4, 5,]:
                color = PARTICLE[c]['color']
                q = p[c].max(0)[..., None]
                g1 += q * [color]
            g1 = np.clip(g1/255 > 0.1,0,1)
            g1 = 1 - (1 - m0) * (1 - g1)

            m0g0g1 =np.hstack([m0,g1,g0])
            plt.imshow(m0g0g1)
            plt.show()
            #plt.waitforbuttonpress()

    
            color = [PARTICLE[c]['color'] for c in range(6)]
            all = draw_probability(p, color)
            plt.imshow(all)
            plt.show()


    torch.cuda.empty_cache() 
    print('\ndone!') 
    num_volume = len(valid_id)
    print(f'Total time for {num_volume} volumes:', time_to_str(total_time, 'min'))
    print(f'Total time for 500 volumes:', time_to_str(total_time/num_volume*500, 'min'))
    print('')
    submit_df = pd.concat(submit_df)
    submit_df.insert(loc=0, column='id', value=np.arange(len(submit_df)))
    return submit_df



if 1:
    submit_df = run_submit(net)
    print('submit_df', submit_df.shape)
    print(submit_df)
    submit_df.to_csv('submission.csv', index=False)

print('MODE:', MODE)
print('SUBMIT OK!!!')
```

```python
if 1:
    if MODE=='local':
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        
        submit_df=pd.read_csv(
           'submission.csv'
            # '/kaggle/input/hengck-czii-cryo-et-weights-01/submission.csv'
        )
        gb, lb_score = compute_lb(submit_df, f'{valid_dir}/overlay')
        print('lb_score:',lb_score)
        print(gb)
        print('')


        #--------------------------------------------
        #visualisation

        fig = plt.figure(figsize=(18, 8))

        id = valid_id[0]
        truth = read_one_truth(id,overlay_dir=f'{valid_dir}/overlay')

        submit_df=pd.read_csv('submission.csv')
        submit_df = submit_df[submit_df['experiment']==id]
        for p in PARTICLE:
            p = dotdict(p)
            xyz_truth = truth[p.name]
            xyz_predict = submit_df[submit_df['particle_type']==p.name][['x','y','z']].values
            hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, p.radius)
            # print(id, p.name)
            # print('\t num truth   :',len(xyz_truth) )
            # print('\t num predict :',len(xyz_predict) )
            # print('\t num hit  :',len(hit[0]) )
            # print('\t num fp   :',len(fp) )
            # print('\t num miss :',len(miss) )
            ax = fig.add_subplot(2, 3, p.label, projection='3d')

            if 0:
                pt = xyz_predict
                ax.scatter(pt[:, 0], pt[:, 1], pt[:, 2], alpha=0.25, color='b', label='predict')
                pt = xyz_truth
                ax.scatter(pt[:, 0], pt[:, 1], pt[:, 2], s=160, alpha=0.25, color='r', label='truth')
                ax.scatter(pt[:, 0], pt[:, 1], pt[:, 2], s=160, facecolors='none', edgecolors='r')
            if 1:
                if hit[0]:
                    pt = xyz_predict[hit[0]]
                    ax.scatter(pt[:, 0], pt[:, 1], pt[:, 2], alpha=0.5, color='r', label='predict')
                    pt = xyz_truth[hit[1]]
                    ax.scatter(pt[:, 0], pt[:, 1], pt[:, 2], s=80, facecolors='none', edgecolors='r', label='truth')
                if fp:
                    pt = xyz_predict[fp]
                    ax.scatter(pt[:, 0], pt[:, 1], pt[:, 2], alpha=0.5, color='k', label='fp')
                if miss:
                    pt = xyz_truth[miss]
                    ax.scatter(pt[:, 0], pt[:, 1], pt[:, 2], s=160, alpha=0.5, facecolors='none', edgecolors='k', label='miss')
            ax.legend()
            ax.set_title(
                f'{id}:{p.name} ({p.difficulty})\npredict={metric[0]}, truth={metric[1]}, hit={metric[2]}, miss={metric[3]}, fp={metric[4]}')

        plt.tight_layout()
        plt.show()
        zz=0
```
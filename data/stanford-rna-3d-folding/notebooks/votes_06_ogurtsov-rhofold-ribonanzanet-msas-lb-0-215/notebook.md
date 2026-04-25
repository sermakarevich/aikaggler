# RhoFold + RibonanzaNet + MSAs [LB 0.215]

- **Author:** Ogurtsov
- **Votes:** 338
- **Ref:** ogurtsov/rhofold-ribonanzanet-msas-lb-0-215
- **URL:** https://www.kaggle.com/code/ogurtsov/rhofold-ribonanzanet-msas-lb-0-215
- **Last run:** 2025-03-15 10:19:09.813000

---

# RhoFold + RibonanzaNet

This notebook combines predictions from [LB 0.179 RNA 3D LR adjusted](https://www.kaggle.com/code/kumarandatascientist/lb-0-179-rna-3d-lr-adjusted) for long (>400 residuals) RNA and [RhoFold](https://github.com/ml4bio/RhoFold) predictions for short (<=400 residuals) RNA. 

It seems like RhoFold makes better predictions but runs OOM for long sequences. 

Possible fix is to make inference on CPU; I have tried it successfully in public test and got notebook scoring forever on LB (Notebook Timeout error).

Please note that only 1 relaxation step is performed in the current version.

The main topics I want to illustrate here is how to run RhoFold on Kaggle and how to combine predictions from different models.

UPD: 

* 0 relaxation steps

* add MSAs from `/kaggle/input/stanford-rna-3d-folding/MSA`

# RibonanzaNet 3D Inference

```python
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
```

```python
config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
    "epochs": 10,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "structural_violation_epoch": 50,
    "balance_weight": False,
}
```

```python
test_data=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
```

```python
from torch.utils.data import Dataset, DataLoader

class RNADataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in (self.data.loc[idx,'sequence'])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)




        return {'sequence':sequence}
```

```python
test_dataset=RNADataset(test_data)
test_dataset[0]
```

```python
import sys

sys.path.append("/kaggle/input/ribonanzanet2d-final")


from Network import *
import yaml



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout=0.2
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load("/kaggle/input/ribonanzanet-weights/RibonanzaNet.pt",map_location='cpu'))
        # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256,64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64,1)) 
        self.dropout=nn.Dropout(0.0)
        self.xyz_predictor=nn.Linear(256,3)

    def forward(self,src):
        
        #with torch.no_grad():
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        xyz=self.xyz_predictor(sequence_features)

        return xyz
```

```python
model=finetuned_RibonanzaNet(load_config_from_yaml("/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml"),pretrained=False).cuda()

model.load_state_dict(torch.load("/kaggle/input/ribonanzanet-3d-finetune/RibonanzaNet-3D.pt"))
```

```python
test_dataset[0]['sequence'].shape
```

```python
model.eval()
preds=[]
for i in range(len(test_dataset)):
    src=test_dataset[i]['sequence'].long()
    src=src.unsqueeze(0).cuda()

    model.train()

    tmp=[]
    for i in range(4):
        with torch.no_grad():
            xyz=model(src).squeeze()
        tmp.append(xyz.cpu().numpy())

    model.eval()
    with torch.no_grad():
        xyz=model(src).squeeze()
    tmp.append(xyz.cpu().numpy())

    tmp=np.stack(tmp,0)
    #exit()
    preds.append(tmp)
```

```python
import plotly.graph_objects as go
import numpy as np

# Example: Generate an Nx3 matrix

xyz = preds[7][0]  # Replace this with your actual Nx3 data
N = len(xyz)

# Extract columns
x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=z,  # Coloring based on z-value
        colorscale='Viridis',  # Choose a colorscale
        opacity=0.8
    )
)])

# Customize layout
fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    title="3D Scatter Plot"
)

# Show figure
fig.show(renderer='iframe')
```

```python
ID=[]
resname=[]
resid=[]
x=[]
y=[]
z=[]

data=[]

for i in range(len(test_data)):
    #print(test_data.loc[i])

    
    for j in range(len(test_data.loc[i,'sequence'])):
        # ID.append(test_data.loc[i,'sequence_id']+f"_{j+1}")
        # resname.append(test_data.loc[i,'sequence'][j])
        # resid.append(j+1) # 1 indexed
        row=[test_data.loc[i,'target_id']+f"_{j+1}",
             test_data.loc[i,'sequence'][j],
             j+1]

        for k in range(5):
            for kk in range(3):
                row.append(preds[i][k][j][kk])
        data.append(row)

columns=['ID','resname','resid']
for i in range(1,6):
    columns+=[f"x_{i}"]
    columns+=[f"y_{i}"]
    columns+=[f"z_{i}"]


submission_ribonanza_net = pd.DataFrame(data,columns=columns)
```

```python
import pandas as pd
import numpy as np
import os
import sys
import argparse
```

```python
!pip install /kaggle/input/openmm/OpenMM-8.2.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

```python
!pip install /kaggle/input/simtk-0-1/simtk-0.1.0-py2.py3-none-any.whl
```

```python
!pip install /kaggle/input/pytest-runner/pytest_runner-6.0.1-py3-none-any.whl
```

```python
!pip install /kaggle/input/biopython/biopython-1.85-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

```python
!pip install /kaggle/input/ml-collections/ml_collections-1.0.0-py3-none-any.whl
```

```python
!cp -R /kaggle/input/rhofold-repo /kaggle/working/
```

```python
!mv rhofold-repo RhoFold
```

```python
cd /kaggle/working/RhoFold
```

```python
!python setup.py install
```

```python
fasta = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
```

```python
fasta
```

```python
!mkdir test_fasta
!mkdir out
```

```python
for target_id in fasta["target_id"].values:
    with open(f"test_fasta/{target_id}.fasta", "w") as text_file:
        text_file.write(">100500\n")
        text_file.write(
            fasta.loc[fasta["target_id"] == target_id, ["sequence"]].to_string(index = False, header = False)
        )
```

```python
for target_id in fasta["target_id"].values:
    os.makedirs(f"./out/{target_id}", exist_ok = True)
    a3m_file = f"/kaggle/input/stanford-rna-3d-folding/MSA/{target_id}.MSA.fasta"
    fasta_file = f"/kaggle/working/RhoFold/test_fasta/{target_id}.fasta"
    if len(open(fasta_file, 'r').read()) > 400:
        continue
    device = "cpu" if len(open(fasta_file, 'r').read()) > 400 else "cuda:0"
    run_folding = f"python inference.py --relax_steps 0 --input_fas {fasta_file} --input_a3m {a3m_file} --output_dir ./out/{target_id}/ --device {device} --ckpt ./pretrained/RhoFold_pretrained.pt"
    os.system(run_folding)
```

```python
def extract_c1_coordinates(pdb_file):
    # Create a PDB parser object
    parser = PDBParser()
    
    # Load the structure from the PDB file
    structure = parser.get_structure('RNA_structure', pdb_file)
    
    # Initialize a list to store the coordinates of C1' atoms
    c1_coordinates = []
    
    # Iterate over all models in the structure (usually only one model)
    for model in structure:
        # Iterate over all chains in the model
        for chain in model:
            # Iterate over all residues in the chain
            for residue in chain:
                # Check if the residue is an RNA nucleotide
                if residue.get_resname() in ['A', 'U', 'G', 'C']:
                    # Try to get the C1' atom
                    try:
                        c1_atom = residue['C1\'']
                        # Append the coordinates of the C1' atom to the list
                        c1_coordinates.append((residue.get_resname(), c1_atom.get_coord()))
                    except KeyError:
                        # If C1' atom is not found, skip this residue
                        print(f"C1' atom not found in residue {residue.get_resname()}{residue.get_id()[1]}")
    
    return c1_coordinates

# (c) DeepSeek
```

```python
from pathlib import Path
from Bio.PDB import PDBParser
```

```python
subm = []
for target_id in fasta["target_id"].values:
    fasta_file = f"/kaggle/working/RhoFold/test_fasta/{target_id}.fasta"    
    if len(open(fasta_file, 'r').read()) > 400:
        continue
    pdb_file = f"/kaggle/working/RhoFold/out/{target_id}/unrelaxed_model.pdb"
    if not Path(pdb_file).exists():
        continue
    coords = extract_c1_coordinates(pdb_file)
    res_name = np.array([x[0] for x in coords])
    res_num = np.array(range(len(res_name))) + 1
    coords = np.array([x[1] for x in coords])

    res = pd.DataFrame({
        "ID" : [target_id + "_" + x for x in res_num.astype("str")],
        "resname": res_name,
        "resid": res_num,
        "x_1": coords[:, 0],
        "y_1": coords[:, 1],
        "z_1": coords[:, 2],
        "x_2": coords[:, 0],
        "y_2": coords[:, 1],
        "z_2": coords[:, 2],
        "x_3": coords[:, 0],
        "y_3": coords[:, 1],
        "z_3": coords[:, 2],
        "x_4": coords[:, 0],
        "y_4": coords[:, 1],
        "z_4": coords[:, 2],
        "x_5": coords[:, 0],
        "y_5": coords[:, 1],
        "z_5": coords[:, 2],
    })

    subm.append(res)

submit_rhofold = pd.concat(subm, axis = 0)
submit_rhofold
```

```python
merged_df = submission_ribonanza_net.merge(
    submit_rhofold, 
    on=['ID', 'resname', 'resid'], 
    how='left', 
    suffixes=('', '_new')
)
merged_df
```

```python
for col in submission_ribonanza_net.columns:
    if col + '_new' in merged_df.columns:
        if not col in ["x_5", "y_5", "z_5"]:
            submission_ribonanza_net[col] = merged_df[col + '_new'].fillna(submission_ribonanza_net[col])
```

```python
submission_ribonanza_net
```

```python
submission_ribonanza_net.to_csv("/kaggle/working/submission.csv", index = False)
```
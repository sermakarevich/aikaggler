# RNAPro inference with TBM

- **Author:** g john rao
- **Votes:** 232
- **Ref:** jaejohn/rnapro-inference-with-tbm
- **URL:** https://www.kaggle.com/code/jaejohn/rnapro-inference-with-tbm
- **Last run:** 2026-01-20 14:46:17.167000

---

```python
# V2: imports recomputed ccd-cache
```

Reference

- https://www.kaggle.com/code/theoviel/stanford-rna-3d-folding-pt2-rnapro-inference

```python
import os
IS_SCORING_RUN = os.environ.get('KAGGLE_IS_COMPETITION_RERUN')
print(IS_SCORING_RUN)
```

```python
!cp -r /kaggle/input/rnapro-src/RNAPro .
!cp /kaggle/input/rnapro-src/rnapro-private-best-500m.ckpt .
```

```python
cd RNAPro
```

```python
pwd
```

```python
pip install -e . --no-deps
```

```python
cd ..
```

```python
pwd
```

TBM
---

```python
# Cell 1: Imports and Setup
import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance_matrix
import random
import time
import warnings

from tqdm.auto import tqdm
warnings.filterwarnings('ignore')
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq

# Enable tqdm for pandas operations
tqdm.pandas()
    
# Initialize global aligner with RNA-appropriate parameters
ALIGNER = PairwiseAligner()
ALIGNER.mode = 'global'
ALIGNER.match_score = 2.1
ALIGNER.mismatch_score = -1
ALIGNER.open_gap_score = -10
ALIGNER.extend_gap_score = -0.5

seed = 21
np.random.seed(seed)
random.seed(seed)

# Cell 2: Load Data
BASE_PATH = '/kaggle/input/stanford-rna-3d-folding-2'

# Load with progress indication
test_seqs = pd.read_csv(f'{BASE_PATH}/test_sequences.csv')
train_seqs = pd.read_csv(f'{BASE_PATH}/train_sequences.csv')
validation_seqs = pd.read_csv(f'{BASE_PATH}/validation_sequences.csv')
train_labels = pd.read_csv(f'{BASE_PATH}/train_labels.csv', low_memory=False)
validation_labels = pd.read_csv(f'{BASE_PATH}/validation_labels.csv')
sample_submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

print(f"✓ Loaded {len(train_seqs)} training sequences")
print(f"✓ Loaded {len(validation_seqs)} validation sequences") 
print(f"✓ Loaded {len(test_seqs)} test sequences")

# Cell 3: Process Training Labels to Coordinate Dictionary
def process_labels(labels_df, use_first_model_only=True):
    """
    Process labels dataframe to create a dictionary mapping target_id to coordinates.
    Vectorized implementation for improved performance.
    
    Args:
        labels_df: DataFrame with ID, resid, x_1, y_1, z_1, etc.
        use_first_model_only: If True, only extract first model coordinates
        
    Returns:
        Dictionary mapping target_id to numpy array of coordinates
    """
    print("Extracting target IDs...")
    # Vectorized target_id extraction
    labels_df = labels_df.copy()
    labels_df['target_id'] = labels_df['ID'].str.rsplit('_', n=1).str[0]
    
    # Sort once for all groups
    labels_df = labels_df.sort_values(['target_id', 'resid'])
    
    # Vectorized coordinate extraction
    print("Extracting coordinates...")
    coord_cols = ['x_1', 'y_1', 'z_1']
    
    # Replace placeholder values with NaN in one operation
    coords_array = labels_df[coord_cols].values.copy()
    coords_array[coords_array < -1e6] = np.nan
    labels_df[coord_cols] = coords_array
    
    # Group and convert to dictionary with progress bar
    print("Grouping by target_id...")
    coords_dict = {}
    
    grouped = labels_df.groupby('target_id', sort=False)
    for target_id, group in tqdm(grouped, desc="Processing structures", total=len(grouped)):
        # Directly extract coordinates as numpy array (already sorted)
        coords_dict[target_id] = group[coord_cols].values
    
    return coords_dict


train_coords_dict = process_labels(train_labels)
valid_coords_dict = process_labels(validation_labels)

# Cell 4: Sequence Alignment and Template Finding
def get_alignment_score(query_seq, template_seq):
    alignments = ALIGNER.align(query_seq, template_seq)
    best_alignment = next(iter(alignments), None)
    
    if best_alignment is None:
        return 0.0
    
    # Normalize score by theoretical max (perfect match of shorter sequence)
    max_possible = 2.1 * min(len(query_seq), len(template_seq))
    normalized_score = best_alignment.score / max_possible
    
    return min(normalized_score, 1.0)  # Cap at 1.0


def get_aligned_sequences(query_seq, template_seq):
    alignments = ALIGNER.align(query_seq, template_seq)
    best_alignment = next(iter(alignments), None)
    
    if best_alignment is None:
        return None, None
    
    # Use the robust method that builds from alignment coordinates
    return build_aligned_sequences(query_seq, template_seq, best_alignment)


def build_aligned_sequences(query_seq, template_seq, alignment):
    """
    Build aligned sequences with gaps from alignment coordinates.
    Uses alignment.aligned property which returns tuples of (start, end) ranges.
    """
    # Get aligned blocks: alignment.aligned returns ((query_ranges), (template_ranges))
    query_ranges, template_ranges = alignment.aligned
    
    # If no aligned blocks, return None
    if len(query_ranges) == 0:
        return None, None
    
    aligned_query = []
    aligned_template = []
    
    query_pos = 0
    template_pos = 0
    
    for (q_start, q_end), (t_start, t_end) in zip(query_ranges, template_ranges):
        # Add gaps for unaligned query residues (query has residues, template doesn't)
        while query_pos < q_start:
            aligned_query.append(query_seq[query_pos])
            aligned_template.append('-')
            query_pos += 1
        
        # Add gaps for unaligned template residues (template has residues, query doesn't)
        while template_pos < t_start:
            aligned_query.append('-')
            aligned_template.append(template_seq[template_pos])
            template_pos += 1
        
        # Add aligned region (both have residues)
        block_len = q_end - q_start  # Should equal t_end - t_start
        for i in range(block_len):
            aligned_query.append(query_seq[q_start + i])
            aligned_template.append(template_seq[t_start + i])
        
        query_pos = q_end
        template_pos = t_end
    
    # Add any remaining unaligned query residues at the end
    while query_pos < len(query_seq):
        aligned_query.append(query_seq[query_pos])
        aligned_template.append('-')
        query_pos += 1
    
    # Add any remaining unaligned template residues at the end
    while template_pos < len(template_seq):
        aligned_query.append('-')
        aligned_template.append(template_seq[template_pos])
        template_pos += 1
    
    return ''.join(aligned_query), ''.join(aligned_template)


def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, 
                          temporal_cutoff=None, top_n=5):
    """
    Find sequences in the training data similar to the query sequence.
    Uses modern PairwiseAligner for sequence comparison.
    
    Args:
        query_seq: The RNA sequence to find templates for
        train_seqs_df: DataFrame containing training sequences
        train_coords_dict: Dictionary mapping target_ids to their 3D coordinates
        temporal_cutoff: Only consider training sequences published before this date
        top_n: Number of top templates to return
        
    Returns:
        List of (target_id, sequence, similarity_score, coordinates) tuples
    """
    similar_seqs = []
    
    # Filter training sequences by temporal cutoff if provided
    if temporal_cutoff:
        filtered_train_seqs = train_seqs_df[train_seqs_df['temporal_cutoff'] < temporal_cutoff]
    else:
        filtered_train_seqs = train_seqs_df
    
    for _, row in filtered_train_seqs.iterrows():
        target_id = row['target_id']
        train_seq = row['sequence']
        
        # Skip if coordinates not available
        if target_id not in train_coords_dict:
            continue
        
        # Skip if sequence length difference is too large (>50%)
        len_diff = abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq))
        if len_diff > 0.5:
            continue
        
        # Calculate similarity score using new aligner
        similarity_score = get_alignment_score(query_seq, train_seq)
        
        similar_seqs.append((target_id, train_seq, similarity_score, train_coords_dict[target_id]))
    
    # Sort by similarity score (higher is better)
    similar_seqs.sort(key=lambda x: x[2], reverse=True)
    return similar_seqs[:top_n]

# Cell 5: Template Adaptation Functions
def adapt_template_to_query(query_seq, template_seq, template_coords):
    # Get aligned sequences
    aligned_query, aligned_template = get_aligned_sequences(query_seq, template_seq)
    
    if aligned_query is None or aligned_template is None:
        return adapt_template_simple(query_seq, template_seq, template_coords)
    
    # Initialize coordinates for query sequence
    query_coords = np.zeros((len(query_seq), 3))
    query_coords.fill(np.nan)
    
    # Map template coordinates to query based on alignment
    query_idx = 0
    template_idx = 0
    
    for i in range(len(aligned_query)):
        query_char = aligned_query[i]
        template_char = aligned_template[i]
        
        if query_char != '-' and template_char != '-':
            # Both aligned - copy template coordinate to query
            if query_idx < len(query_seq) and template_idx < len(template_coords):
                # Handle NaN coordinates in template
                if not np.any(np.isnan(template_coords[template_idx])):
                    query_coords[query_idx] = template_coords[template_idx]
            template_idx += 1
            query_idx += 1
        elif query_char != '-' and template_char == '-':
            # Gap in template - query residue has no template coord
            query_idx += 1
        elif query_char == '-' and template_char != '-':
            # Gap in query - skip template residue
            template_idx += 1
    
    # Fill in gaps by interpolation
    query_coords = fill_coordinate_gaps(query_coords)
    
    return query_coords


def adapt_template_simple(query_seq, template_seq, template_coords):
    """Simple template adaptation without Biopython alignment."""
    query_coords = np.zeros((len(query_seq), 3))
    
    # Simple mapping based on position
    scale = len(template_coords) / len(query_seq)
    for i in range(len(query_seq)):
        template_idx = int(i * scale)
        template_idx = min(template_idx, len(template_coords) - 1)
        if not np.any(np.isnan(template_coords[template_idx])):
            query_coords[i] = template_coords[template_idx]
        else:
            query_coords[i] = [np.nan, np.nan, np.nan]
    
    # Fill gaps
    query_coords = fill_coordinate_gaps(query_coords)
    return query_coords


def fill_coordinate_gaps(coords):
    """Fill NaN gaps in coordinates by interpolation."""
    n = len(coords)
    typical_step = 4.0
    
    # First pass: interpolate between valid points
    for i in range(n):
        if np.isnan(coords[i, 0]):
            prev_valid = next((j for j in range(i-1, -1, -1) if not np.isnan(coords[j, 0])), -1)
            next_valid = next((j for j in range(i+1, n) if not np.isnan(coords[j, 0])), -1)
            
            if prev_valid >= 0 and next_valid >= 0:
                weight = (i - prev_valid) / (next_valid - prev_valid)
                coords[i] = (1 - weight) * coords[prev_valid] + weight * coords[next_valid]
    
    # Second pass: handle remaining NaNs at edges
    for i in range(n):
        if np.isnan(coords[i, 0]):
            if i == 0:
                first_valid = next((j for j in range(1, n) if not np.isnan(coords[j, 0])), -1)
                if first_valid >= 0:
                    for j in range(first_valid - 1, -1, -1):
                        direction = np.random.normal(0, 1, 3)
                        direction = direction / (np.linalg.norm(direction) + 1e-10) * typical_step
                        coords[j] = coords[j + 1] - direction
                else:
                    coords = generate_basic_structure_coords(n)
                    break
            else:
                prev_valid = next((j for j in range(i-1, -1, -1) if not np.isnan(coords[j, 0])), -1)
                if prev_valid >= 0:
                    direction = np.random.normal(0, 1, 3)
                    direction = direction / (np.linalg.norm(direction) + 1e-10) * typical_step
                    coords[i] = coords[prev_valid] + direction
    
    # Final cleanup
    coords = np.nan_to_num(coords)
    return coords


def generate_basic_structure(sequence):
    """Generate a simple helical structure."""
    return generate_basic_structure_coords(len(sequence))


def generate_basic_structure_coords(n_residues):
    """Generate basic helical coordinates."""
    coords = np.zeros((n_residues, 3))
    for i in range(n_residues):
        angle = i * 0.6
        coords[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]
    return coords

# Cell 6: RNA Geometric Constraints
def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    """
    Apply RNA geometric constraints with adaptive strength based on confidence.
    """
    refined_coords = coordinates.copy()
    n_residues = len(sequence)
    
    constraint_strength = 0.8 * (1.0 - min(confidence, 0.8))
    
    # Sequential distance constraints
    seq_min_dist, seq_max_dist = 5.5, 6.5
    
    for i in range(n_residues - 1):
        current_pos = refined_coords[i]
        next_pos = refined_coords[i + 1]
        current_dist = np.linalg.norm(next_pos - current_pos)
        
        if current_dist < seq_min_dist or current_dist > seq_max_dist:
            target_dist = (seq_min_dist + seq_max_dist) / 2
            direction = next_pos - current_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            adjustment = (target_dist - current_dist) * constraint_strength
            refined_coords[i + 1] = current_pos + direction * (current_dist + adjustment)
    
    # Steric clash prevention
    min_allowed_distance = 3.8
    dist_matrix = distance_matrix(refined_coords, refined_coords)
    severe_clashes = np.where((dist_matrix < min_allowed_distance) & (dist_matrix > 0))
    
    for idx in range(len(severe_clashes[0])):
        i, j = severe_clashes[0][idx], severe_clashes[1][idx]
        if abs(i - j) <= 1 or i >= j:
            continue
        
        pos_i, pos_j = refined_coords[i], refined_coords[j]
        current_dist = dist_matrix[i, j]
        direction = pos_j - pos_i
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        adjustment = (min_allowed_distance - current_dist) * constraint_strength
        refined_coords[i] = pos_i - direction * (adjustment / 2)
        refined_coords[j] = pos_j + direction * (adjustment / 2)
    
    return refined_coords

# Cell 7: De Novo Structure Generation
def generate_rna_structure(sequence, seed=None):
    """Generate a more realistic RNA structure prediction."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    n_residues = len(sequence)
    coordinates = np.zeros((n_residues, 3))
    
    # Initialize first residues
    for i in range(min(3, n_residues)):
        angle = i * 0.6
        coordinates[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]
    
    current_direction = np.array([0.0, 0.0, 1.0])
    complementary = {'G': 'C', 'C': 'G', 'A': 'U', 'U': 'A'}
    
    for i in range(3, n_residues):
        current_base = sequence[i]
        has_pair = False
        pair_idx = -1
        
        window_size = min(i, 15)
        for j in range(i - window_size, i):
            if j >= 0 and sequence[j] == complementary.get(current_base, 'X'):
                has_pair = True
                pair_idx = j
                break
        
        if has_pair and i - pair_idx <= 10 and random.random() < 0.7:
            pair_pos = coordinates[pair_idx]
            random_offset = np.random.normal(0, 1, 3) * 2.0
            base_pair_distance = 10.0 + random.uniform(-1.0, 1.0)
            
            center = np.mean(coordinates[:i], axis=0)
            direction = center - pair_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            coordinates[i] = pair_pos + direction * base_pair_distance + random_offset
            current_direction = np.random.normal(0, 0.3, 3)
            current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)
        else:
            if random.random() < 0.3:
                angle = random.uniform(0.2, 0.6)
                axis = np.random.normal(0, 1, 3)
                axis = axis / (np.linalg.norm(axis) + 1e-10)
                rotation = R.from_rotvec(angle * axis)
                current_direction = rotation.apply(current_direction)
            else:
                current_direction += np.random.normal(0, 0.15, 3)
                current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)
            
            step_size = random.uniform(3.5, 4.5)
            coordinates[i] = coordinates[i - 1] + step_size * current_direction
    
    return coordinates

# Cell 8: Main Prediction Function
def predict_rna_structures(sequence, target_id, train_seqs_df, train_coords_dict, 
                          n_predictions=5, temporal_cutoff=None):
    """Generate multiple structure predictions for an RNA sequence."""
    predictions = []
    
    # Find similar sequences
    similar_seqs = find_similar_sequences(
        sequence, train_seqs_df, train_coords_dict,
        temporal_cutoff=temporal_cutoff, top_n=n_predictions
    )
    
    # Use templates if found
    if similar_seqs:
        for template_id, template_seq, similarity, template_coords in similar_seqs:
            adapted_coords = adapt_template_to_query(sequence, template_seq, template_coords)
            
            if adapted_coords is not None:
                refined_coords = adaptive_rna_constraints(adapted_coords, sequence, confidence=similarity)
                random_scale = max(0.05, 0.8 - similarity)
                randomized_coords = refined_coords + np.random.normal(0, random_scale, refined_coords.shape)
                predictions.append(randomized_coords)
                
                if len(predictions) >= n_predictions:
                    break
    
    # Fill remaining with de novo structures
    while len(predictions) < n_predictions:
        seed_value = hash(target_id) % 10000 + len(predictions) * 1000
        de_novo_coords = generate_rna_structure(sequence, seed=seed_value)
        refined_de_novo = adaptive_rna_constraints(de_novo_coords, sequence, confidence=0.2)
        predictions.append(refined_de_novo)
    
    return predictions[:n_predictions]

# Cell 9: Generate Predictions
def generate_predictions_for_dataset(seqs_df, train_seqs_df, train_coords_dict, dataset_name="dataset"):
    """Generate predictions for a given sequence dataset."""
    all_predictions = []
    start_time = time.time()
    total_targets = len(seqs_df)
    
    print(f"\n=== Generating Predictions for {dataset_name} ({total_targets} sequences) ===")
    
    for idx, row in seqs_df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        temporal_cutoff = row.get('temporal_cutoff', None)
        
        if idx % 5 == 0:
            elapsed = time.time() - start_time
            if idx > 0:
                est_remaining = elapsed / (idx + 1) * (total_targets - idx - 1)
                print(f"Processing {idx+1}/{total_targets}: {target_id} ({len(sequence)} nt), "
                      f"elapsed: {elapsed:.1f}s, remaining: {est_remaining:.1f}s")
            else:
                print(f"Processing {idx+1}/{total_targets}: {target_id} ({len(sequence)} nt)")
        
        predictions = predict_rna_structures(
            sequence, target_id, train_seqs_df, train_coords_dict,
            n_predictions=5, temporal_cutoff=temporal_cutoff
        )
        
        for j in range(len(sequence)):
            pred_row = {
                'ID': f"{target_id}_{j+1}",
                'resname': sequence[j],
                'resid': j + 1
            }
            for i in range(5):
                pred_row[f'x_{i+1}'] = predictions[i][j][0]
                pred_row[f'y_{i+1}'] = predictions[i][j][1]
                pred_row[f'z_{i+1}'] = predictions[i][j][2]
            all_predictions.append(pred_row)
    
    # Create DataFrame
    df = pd.DataFrame(all_predictions)
    column_order = ['ID', 'resname', 'resid']
    for i in range(1, 6):
        for coord in ['x', 'y', 'z']:
            column_order.append(f'{coord}_{i}')
    df = df[column_order]
    
    print(f"\nGenerated predictions for {total_targets} sequences")
    print(f"Total runtime: {time.time() - start_time:.1f} seconds")
    print(f"Output shape: {df.shape}")
    
    return df

# Cell 10: Generate Predictions for TEST Set (Submission)
# Generate test predictions for submission
test_pred_df = generate_predictions_for_dataset(
    test_seqs, train_seqs, train_coords_dict,
    dataset_name="Test"
)
test_pred_df.head()

# Cell 11: Save Submission
test_pred_df.to_csv('submission_tbm.csv', index=False)
print("\n=== Submission file saved ===")
print(f"submission.csv shape: {test_pred_df.shape}")
print(f"\nFirst few rows:")
test_pred_df.head(10)
```

```python
sub_tbm = pd.read_csv("/kaggle/working/submission_tbm.csv")
sub_tbm
```

```python
pwd
```

```python
!ls
```

continue RNAPro
---

```python
cd RNAPro
```

```python
!python preprocess/convert_templates_to_pt_files.py --input_csv /kaggle/working/submission_tbm.csv --output_name templates.pt
```

```python
DIST = "/kaggle/working/RNAPro/release_data/ccd_cache/"
!mkdir -p $DIST
```

```python
# Recomputed the cdd cache using python preprocess/gen_ccd_cache.py
# Exported as dataset here: https://www.kaggle.com/datasets/jaejohn/rnapro-ccd-cache

# You will need the following packages to recompute it on our own

# pip install gemmi
# pip install pdbeccdutils
```

```python
# updated file paths
!cp /kaggle/input/rnapro-ccd-cache/ccd_cache/components.cif $DIST
!cp /kaggle/input/rnapro-ccd-cache/ccd_cache/components.cif.rdkit_mol.pkl $DIST
```

Inference  
---

```python
# %%python
import pandas as pd
df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv")
if not IS_SCORING_RUN:
    df = df.head(5)
df.to_csv('/kaggle/working/sample_sequences.csv', index=False)
```

```python
%%writefile runner/inference.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import logging
import traceback
import warnings
import argparse
from contextlib import nullcontext
from os.path import join as opjoin
from typing import Any, Mapping

import json
import torch
import pandas as pd
import numpy as np
from biotite.structure.io import pdbx

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from runner.dumper import DataDumper

from rnapro.config import parse_sys_args
from rnapro.config.config import ConfigManager, ArgumentNotSet
from rnapro.data.infer_data_pipeline import get_inference_dataloader
from rnapro.model.RNAPro import RNAPro
from rnapro.utils.distributed import DIST_WRAPPER
from rnapro.utils.seed import seed_everything
from rnapro.utils.torch_utils import to_device

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)

# Silence all info logging
logging.basicConfig(level=logging.WARNING)
# Silence dataloader logging specifically
logging.getLogger("rnapro.data").setLevel(logging.WARNING)
logging.getLogger("rnapro").setLevel(logging.WARNING)


def parse_configs(
    configs: dict, arg_str: str = None, fill_required_with_null: bool = False
):
    """
    Parses and merges configuration settings from a dictionary and command-line arguments.

    Args:
        configs (dict): A dictionary containing initial configuration settings.
        arg_str (str, optional): A string representing command-line arguments. Defaults to None.
        fill_required_with_null (bool, optional):
            A boolean flag indicating whether required values should be filled with `None` if not provided. Defaults to False.

    Returns:
        ConfigDict: The merged configuration dictionary.
    """
    manager = ConfigManager(configs, fill_required_with_null=fill_required_with_null)
    parser = argparse.ArgumentParser()

    # This is new
    parser.add_argument(
        "--max_len",
        type=int,
        default=10000,
        required=False,
        help="Maximum length of the sequence. Longer sequences will be skipped during inference"
    )

    # Register arguments
    for key, (
        dtype,
        default_value,
        allow_none,
        required,
    ) in manager.config_infos.items():
        # All config use str type, strings will be converted to real dtype later
        parser.add_argument(
            "--" + key, type=str, default=ArgumentNotSet(), required=required
        )
    # Merge user commandline pargs with default ones
    merged_configs = manager.merge_configs(
        vars(parser.parse_args(arg_str.split())) if arg_str else {}
    )

    max_len = parser.parse_args(arg_str.split()).max_len
    merged_configs.max_len = max_len

    return merged_configs


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class InferenceRunner(object):
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    def init_env(self) -> None:
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set `CUTLASS_PATH` env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            if env is not None:
                logging.info(
                    "The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time."
                )
        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", None)
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "The kernels will be compiled when fast_layernorm is called for the first time."
            )

        logging.info("Finished init ENV.")

    def init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        self.model = RNAPro(self.configs).to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters())
        self.print(f"Total number of parameters: {num_params:,}")

    def load_checkpoint(self) -> None:
        checkpoint_path = self.configs.load_checkpoint_path

        if not os.path.exists(checkpoint_path):
            raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(checkpoint_path, self.device)

        sample_key = [k for k in checkpoint["model"].keys()][0]
        # self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module."):]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=True,
        )
        self.model.eval()
        # self.print("Finish loading checkpoint.")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ):
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    def print_dict(self, d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: ", v.shape)
            else:
                pass
                # print(f"{k}: {v}")

    # Adapted from runner.train.Trainer.evaluate
    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        # print("eval_precision: ", eval_precision)
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        #         print('input_feature_dict: ', self.print_dict(data["input_feature_dict"]))
        #         exit(0)

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )

        return prediction

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            # logger.info(msg)
            print(msg)

    def update_model_configs(self, new_configs: Any) -> None:
        self.model.configs = new_configs


def update_inference_configs(configs: Any, N_token: int):
    # Setting the default inference configs for different N_token and N_atom
    # when N_token is larger than 3000, the default config might OOM even on a
    # A100 80G GPUS,
    if N_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif N_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True
    return configs


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    # Data
    # logger.info(f"Loading data from {configs.input_json_path}")
    try:
        dataloader = get_inference_dataloader(configs=configs)
    except Exception as e:
        error_message = f"{e}:\n{traceback.format_exc()}"
        logger.info(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a") as f:
            f.write(error_message)
        return

    num_data = len(dataloader.dataset)
    for seed in configs.seeds:
        seed_everything(seed=seed, deterministic=configs.deterministic)
        for batch in dataloader:
            try:
                data, atom_array, data_error_message = batch[0]
                sample_name = data["sample_name"]

                if len(data_error_message) > 0:
                    logger.info(data_error_message)
                    with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a") as f:
                        f.write(data_error_message)
                    continue

                logger.info(
                    (
                        f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
                        f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                        f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                    )
                )
                new_configs = update_inference_configs(configs, data["N_token"].item())
                runner.update_model_configs(new_configs)
                prediction = runner.predict(data)
                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type=data["entity_poly_type"],
                )

                logger.info(
                    f"[Rank {DIST_WRAPPER.rank}] {data['sample_name']} succeeded - "
                    f"Results saved to {configs.dump_dir}"
                )
                torch.cuda.empty_cache()
            except Exception as e:
                error_message = f"[Rank {DIST_WRAPPER.rank}]{data['sample_name']} {e}:\n{traceback.format_exc()}"
                logger.info(error_message)
                # Save error info
                with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a") as f:
                    f.write(error_message)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()


# data helper
def make_dummy_solution(valid_df):
    solution = dotdict()
    for i, row in valid_df.iterrows():
        target_id = row.target_id
        sequence = row.sequence
        solution[target_id] = dotdict(
            target_id=target_id,
            sequence=sequence,
            coord=[],
        )
    return solution


def solution_to_submit_df(solution):
    submit_df = []
    for k, s in solution.items():
        df = coord_to_df(s.sequence, s.coord, s.target_id)
        submit_df.append(df)

    submit_df = pd.concat(submit_df)
    return submit_df


def coord_to_df(sequence, coord, target_id):
    L = len(sequence)
    df = pd.DataFrame()
    df["ID"] = [f"{target_id}_{i + 1}" for i in range(L)]
    df["resname"] = [s for s in sequence]
    df["resid"] = [i + 1 for i in range(L)]

    num_coord = len(coord)
    for j in range(num_coord):
        df[f"x_{j+1}"] = coord[j][:, 0]
        df[f"y_{j+1}"] = coord[j][:, 1]
        df[f"z_{j+1}"] = coord[j][:, 2]
    return df


def main(configs: Any) -> None:
    # Runner
    runner = InferenceRunner(configs)
    infer_predict(runner, configs)


def create_input_json(sequence, target_id):
    # print("input_no_msa")
    input_json = [
        {
            "sequences": [
                {
                    "rnaSequence": {
                        "sequence": sequence,
                        "count": 1,
                    }
                }
            ],
            "name": target_id,
        }
    ]
    return input_json


def extract_c1_coordinates(cif_file_path):
    try:
        # Read the CIF file using the correct biotite method
        with open(cif_file_path, "r") as f:
            cif_data = pdbx.CIFFile.read(f)

        # Get structure from CIF data
        atom_array = pdbx.get_structure(cif_data, model=1)

        # Clean atom names and find C1' atoms
        atom_names_clean = np.char.strip(atom_array.atom_name.astype(str))
        mask_c1 = atom_names_clean == "C1'"
        c1_atoms = atom_array[mask_c1]

        if len(c1_atoms) == 0:
            print(f"Warning: No C1' atoms found in {cif_file_path}")
            return None

        # Sort by residue ID and return coordinates
        sort_indices = np.argsort(c1_atoms.res_id)
        c1_atoms_sorted = c1_atoms[sort_indices]
        c1_coords = c1_atoms_sorted.coord

        return c1_coords
    except Exception as e:
        print(f"Error extracting C1' coordinates from {cif_file_path}: {e}")
        return None


def process_sequence(sequence, target_id, temp_dir):
    # Create input JSON
    input_json = create_input_json(sequence, target_id)

    # Save JSON to temporary file
    os.makedirs(temp_dir, exist_ok=True)
    input_json_path = os.path.join(temp_dir, f"{target_id}_input.json")
    with open(input_json_path, "w") as f:
        json.dump(input_json, f, indent=4)


def run_ptx(target_id, sequence, configs, solution, template_idx, runner):
    # Create directories
    temp_dir = f"./{configs.dump_dir}/input"  # Same as in kaggle_inference.py
    output_dir = f"./{configs.dump_dir}/output"  # Same as in kaggle_inference.py
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    process_sequence(sequence=sequence, target_id=target_id, temp_dir=temp_dir)
    configs.input_json_path = os.path.join(temp_dir, f"{target_id}_input.json")
    configs.template_idx = int(template_idx)

    infer_predict(runner, configs)

    cif_file_path = (
        f"{configs.dump_dir}/{target_id}/seed_42/predictions/{target_id}_sample_0.cif"
    )
    cif_new_path = f"{configs.dump_dir}/{target_id}/seed_42/predictions/{target_id}_sample_{template_idx}_new.cif"
    shutil.copy(cif_file_path, cif_new_path)
    coord = extract_c1_coordinates(cif_file_path)
    if coord is None:
        coord = np.zeros((len(sequence), 3), dtype=np.float32)
    elif coord.shape[0] < (len(sequence)):
        pad_len = len(sequence) - coord.shape[0]
        pad = np.zeros((pad_len, 3), dtype=np.float32)
        coord = np.concatenate([coord, pad], axis=0)
    solution[target_id].coord.append(coord)


def run() -> None:
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.WARNING,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    # Silence dataloader and rnapro module logging
    logging.getLogger("rnapro.data").setLevel(logging.WARNING)
    logging.getLogger("rnapro").setLevel(logging.WARNING)
    configs_base["use_deepspeed_evo_attention"] = (
        os.environ.get("USE_DEEPSPEED_EVO_ATTENTION", False) == "true"
    )
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )

    valid_df = pd.read_csv(configs.sequences_csv)
    print(f"\n -> Loaded {len(valid_df)} sequence(s)")

    # Build model and load checkpoint once before looping over sequences

    print('\n -> Building model and loading checkpoint')
    runner = InferenceRunner(configs)
    print('\n -> Done, starting inference...')

    solution = make_dummy_solution(valid_df)
    for idx, row in valid_df.iterrows():
        print(f"\n -> Sequence {row.target_id}: {row.sequence}")

        if len(row.sequence) > configs.max_len:
            print(f'Sequence is too long ({len(row.sequence)} > {configs.max_len}), skipping')
            for template_idx in range(5):
                coord = np.zeros((len(row.sequence), 3), dtype=np.float32)
                solution[row.target_id].coord.append(coord)
            continue

        try:
            target_id = row.target_id
            sequence = row.sequence
            for template_idx in range(5):
                print()
                run_ptx(
                    target_id=target_id,
                    sequence=sequence,
                    configs=configs,
                    solution=solution,
                    template_idx=template_idx,
                    runner=runner,
                )
        except Exception as e:
            print(f"Error processing {row.target_id}: {e}")
            continue

    print('\n\n -> Inference done ! Saving to submission.csv')
    submit_df = solution_to_submit_df(solution)
    submit_df = submit_df.fillna(0.0)
    submit_df.to_csv("./submission.csv", index=False)


if __name__ == "__main__":
    run()
```

```python
%%writefile rnapro_inference_kaggle.sh

export LAYERNORM_TYPE=torch # fast_layernorm, torch


# Inference parameters (RNAPro)
SEED=42
N_SAMPLE=1
N_STEP=200
N_CYCLE=10

# Paths
DUMP_DIR="../output"
# Set a valid checkpoint file path below
CHECKPOINT_PATH="../rnapro-private-best-500m.ckpt"

# Template/MSA settings
TEMPLATE_DATA="./release_data/kaggle/templates.pt"
# Note: template_idx supports 5 choices and maps to top-k:
# 0->top1, 1->top2, 2->top3, 3->top4, 4->top5
TEMPLATE_IDX=0
RNA_MSA_DIR="/kaggle/input/stanford-rna-3d-folding-2/MSA"

# SEQUENCES_CSV="/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv"
SEQUENCES_CSV="/kaggle/working/sample_sequences.csv"

# RibonanzaNet2 path (keep as-is per request)
RIBONANZA_PATH="/kaggle/input/ribonanzanet2/pytorch/alpha/1/"

# Model selection: keep to an existing key to align defaults (N_step=200, N_cycle=10)
MODEL_NAME="rnapro_base"
mkdir -p "${DUMP_DIR}"

python3 runner/inference.py \
    --model_name "${MODEL_NAME}" \
    --seeds ${SEED} \
    --dump_dir "${DUMP_DIR}" \
    --load_checkpoint_path "${CHECKPOINT_PATH}" \
    --use_msa true \
    --use_template "ca_precomputed" \
    --model.use_template "ca_precomputed" \
    --model.use_RibonanzaNet2 true \
    --model.template_embedder.n_blocks 2 \
    --model.ribonanza_net_path "${RIBONANZA_PATH}" \
    --template_data "${TEMPLATE_DATA}" \
    --template_idx ${TEMPLATE_IDX} \
    --rna_msa_dir "${RNA_MSA_DIR}" \
    --model.N_cycle ${N_CYCLE} \
    --sample_diffusion.N_sample ${N_SAMPLE} \
    --sample_diffusion.N_step ${N_STEP} \
    --load_strict true \
    --num_workers 0 \
    --triangle_attention "torch" \
    --triangle_multiplicative "torch" \
    --sequences_csv "${SEQUENCES_CSV}" \
    --max_len 1000


# --triangle_attention supports 'triattention', 'cuequivariance', 'deepspeed', 'torch'
# --triangle_multiplicative supports 'cuequivariance', 'torch'
# --max_len 1000: Sequences longer than max_len will be skipped to avoid oom
```

```python
!bash ./rnapro_inference_kaggle.sh
```

```python
!mv submission.csv ..
```

```python
cd ..
```

```python
!head submission.csv
```

```python
import pandas as pd
sub = pd.read_csv("/kaggle/working/submission.csv")
sub
```

```python
import pandas as pd
import numpy as np

df_tbm = pd.read_csv("submission_tbm.csv")
df_rnapro = pd.read_csv("submission.csv")
df_seqs = pd.read_csv("/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv")
long_targets = df_seqs[df_seqs['sequence'].str.len() > 1000]['target_id'].values

print(f"Targets to replace with TBM (len > 1000): {long_targets}")

mask_long = df_rnapro['ID'].apply(lambda x: any(str(x).startswith(t + "_") for t in long_targets))

if mask_long.sum() > 0:
    print(f"Replacing {mask_long.sum()} residues with TBM predictions...")
    
    # Set index to ID for easy alignment
    df_rnapro_idx = df_rnapro.set_index('ID')
    df_tbm_idx = df_tbm.set_index('ID')
    
    # Update rows in RNAPro df with TBM df for the specific IDs
    # This works if indices match
    ids_to_update = df_rnapro_idx[mask_long.values].index
    
    # Check if these IDs exist in TBM file
    valid_ids = [i for i in ids_to_update if i in df_tbm_idx.index]
    
    df_rnapro_idx.loc[valid_ids] = df_tbm_idx.loc[valid_ids]
    
    # Reset index to get ID column back
    df_final = df_rnapro_idx.reset_index()
    
    # Save merged
    df_final.to_csv("submission.csv", index=False)
    print("Merged submission saved to submission.csv")
else:
    print("No long targets found to replace. Keeping RNAPro submission as is.")
```

```python
sub = pd.read_csv("/kaggle/working/submission.csv")
sub
```

Evaluation  
---

```python
%%writefile /kaggle/working/metric.py
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import math
import pandas as pd
from pathlib import Path
import shutil
import sys
import csv

# ---------------------
# Helper: parse USalign output
# ---------------------
def parse_tmscore_output(output: str) -> float:
    matches = re.findall(r'TM-score=\s+([\d.]+)', output)
    if len(matches) < 2:
        raise ValueError('No TM score found in USalign output')
    return float(matches[1])

# ---------------------
# PDB writers
# ---------------------

def sanitize(xyz):
    MIN_COORD=-999.999
    MAX_COORD=9999.999
    return min(max(xyz,MIN_COORD),MAX_COORD)

def write_target_line(atom_name, atom_serial, residue_name, chain_id, residue_num,
                      x_coord, y_coord, z_coord, occupancy=1.0, b_factor=0.0, atom_type='P') -> str:
    return f'ATOM  {atom_serial:>5d}  {atom_name:4s}{residue_name:>3s} {chain_id:1s}{residue_num:>4d}    {sanitize(x_coord):>8.3f}{sanitize(y_coord):>8.3f}{sanitize(z_coord):>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}           {atom_type}\n'
    
def write2pdb(df: pd.DataFrame, xyz_id: int, target_path: str) -> int:
    """
    Write single-chain PDB (chain 'A') using row['resid'] as residue_num.
    Raises exceptions on invalid data.
    """
    resolved_cnt = 0
    with open(target_path, 'w') as fh:
        for _, row in df.iterrows():
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if x > -1e6 and y > -1e6 and z > -1e6:
                resolved_cnt += 1
                resid_num = int(row['resid'])
                fh.write(write_target_line("C1'", resid_num, row['resname'], 'A', resid_num, x, y, z, atom_type='C'))
    return resolved_cnt

def write2pdb_singlechain_native(df_native: pd.DataFrame, xyz_id: int, target_path: str) -> int:
    """
    Write native single-chain using row['resid'] as residue numbers.
    Assumes all required columns exist and are valid.
    """
    df_sorted = df_native.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int').reset_index(drop=True)

    resolved_cnt = 0
    with open(target_path, 'w') as fh:
        for _, row in df_sorted.iterrows():
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if x > -1e6 and y > -1e6 and z > -1e6:
                resolved_cnt += 1
                resid_num = int(row['resid'])
                fh.write(write_target_line("C1'", resid_num, row['resname'], 'A', resid_num, x, y, z, atom_type='C'))
    return resolved_cnt

def write2pdb_multichain_from_solution(df_solution: pd.DataFrame, xyz_id: int, target_path: str) -> int:
    """
    Write multi-chain PDB for native solution using columns 'chain' and 'copy' to assign chain letters.
    Expects 'resid' convertible to int and chain/copy present. No fallbacks.
    """
    df_sorted = df_solution.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int')

    chain_map = {}
    next_ord = ord('A')
    written = 0
    with open(target_path, 'w') as fh:
        for _, row in df_sorted.iterrows():
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if not (x > -1e6 and y > -1e6 and z > -1e6):
                continue
            chain_val = row['chain']
            copy_key = int(row['copy'])
            g = (str(chain_val), copy_key)
            if g not in chain_map:
                if next_ord <= ord('Z'):
                    ch = chr(next_ord)
                else:
                    ov = next_ord - ord('Z') - 1
                    if ov < 26:
                        ch = chr(ord('a') + ov)
                    else:
                        ch = chr(ord('0') + (ov - 26) % 10)
                chain_map[g] = ch
                next_ord += 1
            chain_id = chain_map[g]
            written += 1
            resid_num = int(row['resid'])
            fh.write(write_target_line("C1'", resid_num, row['resname'], chain_id, resid_num, x, y, z, atom_type='C'))
    return written

def write2pdb_multichain_from_groups(df_pred: pd.DataFrame, xyz_id: int, target_path: str, groups_list) -> (int, list):
    """
    Write predicted multichain PDB based on a positional groups_list (tuple per residue: (chain, copy)).
    Requires groups_list length == number of residues in df_pred (after sorting).
    Returns (written_count, chain_letters_per_res).
    """
    df_sorted = df_pred.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int').reset_index(drop=True)

    if groups_list is None or len(groups_list) != len(df_sorted):
        raise ValueError("groups_list must be provided and match number of residues in predicted df")

    chain_map = {}
    next_ord = ord('A')
    chain_letters = []
    written = 0
    with open(target_path, 'w') as fh:
        for idx, row in df_sorted.iterrows():
            g = groups_list[idx]
            if isinstance(g, tuple):
                gkey = (str(g[0]), int(g[1]))
            else:
                gkey = (str(g), None)
            if gkey not in chain_map:
                if next_ord <= ord('Z'):
                    ch = chr(next_ord)
                else:
                    ov = next_ord - ord('Z') - 1
                    if ov < 26:
                        ch = chr(ord('a') + ov)
                    else:
                        ch = chr(ord('0') + (ov - 26) % 10)
                chain_map[gkey] = ch
                next_ord += 1
            chain_id = chain_map[gkey]
            chain_letters.append(chain_id)
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if x > -1e6 and y > -1e6 and z > -1e6:
                written += 1
                resid_num = int(row['resid'])
                fh.write(write_target_line("C1'", resid_num, row['resname'], chain_id, resid_num, x, y, z, atom_type='C'))
    return written, chain_letters

def write2pdb_singlechain_permuted_pred(df_pred: pd.DataFrame, xyz_id: int, permuted_indices: list, target_path: str) -> int:
    """
    Create single-chain PDB by concatenating predicted residues in permuted_indices order.
    Output residue numbers are sequential starting at 1 and increase for every permuted position.
    Raises exception if indices out of range.
    """
    df_sorted = df_pred.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int').reset_index(drop=True)

    written = 0
    next_res = 1
    with open(target_path, 'w') as fh:
        for idx in permuted_indices:
            if idx < 0 or idx >= len(df_sorted):
                # strict behavior: raise error for invalid index
                raise IndexError(f"permuted index {idx} out of range for predicted residues")
            row = df_sorted.iloc[idx]
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            out_resnum = next_res
            if x > -1e6 and y > -1e6 and z > -1e6:
                written += 1
                fh.write(write_target_line("C1'", out_resnum, row['resname'], 'A', out_resnum, x, y, z, atom_type='C'))
            next_res += 1
    return written

# ---------------------
# USalign wrappers
# ---------------------
def run_usalign_raw(predicted_pdb: str, native_pdb: str, usalign_bin='USalign', align_sequence=False, tmscore=None) -> str:
    cmd = f'{usalign_bin} {predicted_pdb} {native_pdb} -atom " C1\'"'
    if tmscore is not None:
        cmd += f' -TMscore {tmscore}'
        if int(tmscore) == 0:
            cmd += ' -mm 1 -ter 0'
    elif not align_sequence:
        cmd += ' -TMscore 1'
    return os.popen(cmd).read()

def parse_usalign_chain_orders(output: str):
    """
    Parse USalign output for both Structure_1 and Structure_2 chain lists.
    Returns (chain_list_structure1, chain_list_structure2).
    Raises if parsing fails to find either line.
    """
    chain1 = None
    chain2 = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith('Name of Structure_1:'):
            parts = line.split(':')
            clist = []
            for part in parts[2:]:
                token = part.strip()
                if token == '':
                    continue
                token0 = token.split()[0]
                last = token0.split(',')[-1]
                ch = re.sub(r'[^A-Za-z0-9]', '', last)
                if ch:
                    clist.append(ch)
            chain1 = clist
        elif line.startswith('Name of Structure_2:'):
            parts = line.split(':')
            clist = []
            for part in parts[2:]:
                token = part.strip()
                if token == '':
                    continue
                token0 = token.split()[0]
                last = token0.split(',')[-1]
                ch = re.sub(r'[^A-Za-z0-9]', '', last)
                if ch:
                    clist.append(ch)
            chain2 = clist
    if chain1 is None or chain2 is None:
        raise ValueError("Failed to parse chain orders from USalign output")
    return chain1, chain2

# ---------------------
# Main scoring function (no try/except, no fallbacks)
# ---------------------
def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, usalign_bin_hint: str = None) -> float:
    """
    Enhanced scoring with chain-permutation handling for multicopy targets.
    This version contains no try/except blocks and will raise on any error.
    """
    # determine usalign binary
    if usalign_bin_hint:
        usalign_bin = usalign_bin_hint
    else:
        if os.path.exists('/kaggle/input/usalign/USalign') and not os.path.exists('/kaggle/working/USalign'):
            shutil.copy2('/kaggle/input/usalign/USalign', '/kaggle/working/USalign')
            os.chmod('/kaggle/working/USalign', 0o755)
        usalign_bin = '/kaggle/working/USalign' if os.path.exists('/kaggle/working/USalign') else 'USalign'

    sol = solution.copy()
    sub = submission.copy()
    sol['target_id'] = sol['ID'].apply(lambda x: '_'.join(str(x).split('_')[:-1]))
    sub['target_id'] = sub['ID'].apply(lambda x: '_'.join(str(x).split('_')[:-1]))

    results = []

    for target_id, group_native in sol.groupby('target_id'):
        group_predicted = sub[sub['target_id'] == target_id]
        has_chain_copy = ('chain' in group_native.columns) and ('copy' in group_native.columns)
        is_multicopy = has_chain_copy and (group_native['copy'].astype(float).max() > 1)

        # precompute native models that have coords
        native_with_coords = []
        for native_cnt in range(1, 41):
            native_pdb = f'native_{target_id}_{native_cnt}.pdb'
            resolved_native = write2pdb(group_native, native_cnt, native_pdb)
            if resolved_native > 0:
                native_with_coords.append(native_cnt)
            else:
                if os.path.exists(native_pdb):
                    os.remove(native_pdb)

        if not native_with_coords:
            raise ValueError(f"No native models with coordinates for target {target_id}")

        best_per_pred = []
        for pred_cnt in range(1, 6):
            if not is_multicopy:
                predicted_pdb = f'predicted_{target_id}_{pred_cnt}.pdb'
                resolved_pred = write2pdb(group_predicted, pred_cnt, predicted_pdb)
                if resolved_pred <= 2:
                    #print(f"Predicted model {pred_cnt} for target {target_id} has insufficient coordinates")
                    best_per_pred.append( 0.0 )
                    continue
                
                scores = []
                for native_cnt in native_with_coords:
                    native_pdb = f'native_{target_id}_{native_cnt}.pdb'
                    out = run_usalign_raw(predicted_pdb, native_pdb, usalign_bin=usalign_bin, align_sequence=False, tmscore=1)
                    s = parse_tmscore_output(out)
                    scores.append(s)
                best_per_pred.append(max(scores))

            else:
                # multicopy
                # strict: require chain and copy columns convertible
                gn_sorted = group_native.copy()
                gn_sorted['__resid_int'] = gn_sorted['resid'].astype(int)
                gn_sorted = gn_sorted.sort_values('__resid_int').reset_index(drop=True)
                groups_list = []
                for _, r in gn_sorted.iterrows():
                    chain_val = r['chain']
                    copy_i = int(r['copy'])
                    groups_list.append((chain_val, copy_i))

                # predicted multichain - groups_list must match predicted residue count or error
                dfp_sorted = group_predicted.copy()
                dfp_sorted['__resid_int'] = dfp_sorted['resid'].astype(int)
                dfp_sorted = dfp_sorted.sort_values('__resid_int').reset_index(drop=True)
                if len(groups_list) != len(dfp_sorted):
                    raise ValueError(f"groups_list length ({len(groups_list)}) does not match predicted residue count ({len(dfp_sorted)}) for target {target_id}")

                predicted_multi_pdb = f'pred_multi_{target_id}_{pred_cnt}.pdb'
                resolved_pred_multi, pred_chain_letters = write2pdb_multichain_from_groups(group_predicted, pred_cnt, predicted_multi_pdb, groups_list)
                if resolved_pred_multi == 0:
                    #print(f"Predicted multi model {pred_cnt} for target {target_id} has no coordinates")
                    best_per_pred.append( 0.0 )
                    continue

                scores = []
                for native_cnt in native_with_coords:
                    native_multi_pdb = f'native_multi_{target_id}_{native_cnt}.pdb'
                    resolved_native_multi = write2pdb_multichain_from_solution(group_native, native_cnt, native_multi_pdb)
                    if resolved_native_multi == 0:
                        continue

                    raw_out = run_usalign_raw(predicted_multi_pdb, native_multi_pdb, usalign_bin=usalign_bin, align_sequence=True, tmscore=0)
                    chain1, chain2 = parse_usalign_chain_orders(raw_out)  # will raise if parsing fails

                    # build native->pred mapping chain2[i] -> chain1[i]
                    native_to_pred = {n_ch: p_ch for n_ch, p_ch in zip(chain2, chain1)}

                    # canonical native order = chain2 unique in order seen
                    #native_chain_order = []
                    #for ch in chain2:
                    #    if ch not in native_chain_order:
                    #        native_chain_order.append(ch)
                    native_chain_order = list(native_to_pred.keys())
                    native_chain_order.sort() # this is critical...

                    # predicted chain order by following native chain A,B,...
                    pred_chain_order = [native_to_pred[n_ch] for n_ch in native_chain_order if native_to_pred.get(n_ch) is not None]

                    # construct pred_positions_by_chain
                    pred_positions_by_chain = {}
                    for idx, ch in enumerate(pred_chain_letters):
                        if ch is None:
                            continue
                        pred_positions_by_chain.setdefault(ch, []).append(idx)

                    # require that each chain in pred_chain_order exists in pred_positions_by_chain
                    pred_chain_order = [p for p in pred_chain_order if p in pred_positions_by_chain]

                    # form permuted indices by concatenation
                    permuted_indices = []
                    for ch in pred_chain_order:
                        permuted_indices.extend(pred_positions_by_chain[ch])
                    # append any remaining
                    for idx in range(len(pred_chain_letters)):
                        if idx not in permuted_indices:
                            permuted_indices.append(idx)

                    # write permuted single-chain predicted and native single-chain
                    pred_single_perm = f'pred_permuted_{target_id}_{pred_cnt}_{native_cnt}.pdb'
                    written_pred_single = write2pdb_singlechain_permuted_pred(group_predicted, pred_cnt, permuted_indices, pred_single_perm)
                    native_single = f'native_single_{target_id}_{native_cnt}.pdb'
                    written_native = write2pdb_singlechain_native(group_native, native_cnt, native_single)

                    if written_pred_single <= 2 or written_native <= 2:
                        raise ValueError(f"Insufficient residues after permutation for target {target_id}, pred {pred_cnt}, native {native_cnt}")

                    out = run_usalign_raw(pred_single_perm, native_single, usalign_bin=usalign_bin, align_sequence=False, tmscore=1)
                    score_final = parse_tmscore_output(out)
                    scores.append(score_final)

                best_per_pred.append(max(scores))

        results.append(max(best_per_pred))

    if not results:
        pass
        #raise ValueError("No targets scored")
    return float(sum(results) / len(results)) if len(results)>0 else 0.0
```

```python
import runpy
module_globals = runpy.run_path("/kaggle/working/metric.py")
score = module_globals['score']
```

```python
if not IS_SCORING_RUN:
    import pandas as pd
    sub = pd.read_csv('/kaggle/working/submission.csv')
    sol = pd.read_csv('/kaggle/input/stanford-rna-3d-folding-2/validation_labels.csv')

    sub['target_id'] = sub['ID'].apply(lambda x: '_'.join(str(x).split('_')[:-1]))
    sol['target_id'] = sol['ID'].apply(lambda x: '_'.join(str(x).split('_')[:-1]))
    
    # Get unique targets from submission
    sub_targets = sub['target_id'].unique()
    
    results = []
    for target_id in sub_targets:
        group_native = sol[sol['target_id'] == target_id]
        group_predicted = sub[sub['target_id'] == target_id]
        result = score(group_native, group_predicted, 'ID')
        print(f"{target_id}: {result:.4f}")
        results.append(result)
    
    print(f"\nMean score: {sum(results)/len(results):.4f} (n={len(results)})")
```
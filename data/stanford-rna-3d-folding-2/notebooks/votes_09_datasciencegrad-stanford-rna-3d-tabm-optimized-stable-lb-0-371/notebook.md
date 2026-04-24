# Stanford RNA 3D | TaBM Optimized | Stable LB-0.371

- **Author:** Sagar Nagpure
- **Votes:** 209
- **Ref:** datasciencegrad/stanford-rna-3d-tabm-optimized-stable-lb-0-371
- **URL:** https://www.kaggle.com/code/datasciencegrad/stanford-rna-3d-tabm-optimized-stable-lb-0-371
- **Last run:** 2026-02-09 03:00:23.853000

---

# ✅ Stanford RNA 3D Folding Part 2 : Templat Based Ensemble Prediction

## Approach
- Template matching via sequence alignment
- Top-5 similar structure selection
- Coordinate transfer with gap interpolation
- RNA backbone constraint refinement (5.5-6.5Å)
- Ensemble generation with confidence-weighted noise

## Output
- 5 diverse predictions per test sequence
- Submission format: ID, resname, resid, x_1-5, y_1-5, z_1-5

##### Based On : [Notebook](https://www.kaggle.com/code/kami1976/stanford-rna-3d-folding-part-2a18)

```python
!pip install --no-index /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
```

```python
import pandas as pd
import numpy as np
import random
import time
import warnings
import os, sys

warnings.filterwarnings('ignore')

DATA_PATH = '/kaggle/input/competitions/stanford-rna-3d-folding-2/extra'
train_seqs = pd.read_csv(DATA_PATH + 'train_sequences.csv')
test_seqs = pd.read_csv(DATA_PATH + 'test_sequences.csv')
train_labels = pd.read_csv(DATA_PATH + 'train_labels.csv')

sys.path.append(os.path.join(DATA_PATH, "extra"))

# --- Robust import for Kaggle's extra/parse_fasta_py.py (it may miss typing imports) ---
try:
    import typing as _typing
    import builtins as _builtins

    # Make these names available during module import-time annotation evaluation
    _builtins.Dict  = getattr(_typing, "Dict")
    _builtins.Tuple = getattr(_typing, "Tuple")
    _builtins.List  = getattr(_typing, "List")

    from parse_fasta_py import parse_fasta as _parse_fasta_raw

    # Normalize output to: {chain_id: sequence_string}
    def parse_fasta(fasta_content: str):
        d = _parse_fasta_raw(fasta_content)
        out = {}
        for k, v in d.items():
            # some variants return (sequence, headers/lines) or similar
            out[k] = v[0] if isinstance(v, tuple) else v
        return out

except Exception:
    # Fallback FASTA parser: {chain_id: sequence_string}
    def parse_fasta(fasta_content: str):
        out = {}
        cur = None
        seq_parts = []
        for line in str(fasta_content).splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    out[cur] = "".join(seq_parts)
                header = line[1:]
                # First token is usually chain id in this dataset
                cur = header.split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.replace(" ", ""))
        if cur is not None:
            out[cur] = "".join(seq_parts)
        return out

def parse_stoichiometry(stoich: str):
    if pd.isna(stoich) or str(stoich).strip() == "":
        return []
    out = []
    for part in str(stoich).split(';'):
        ch, cnt = part.split(':')
        out.append((ch.strip(), int(cnt)))
    return out

def get_chain_segments(row):
    """
    Returns list of (start,end) segments in row['sequence'] corresponding to chain copies in stoichiometry order.
    Falls back to single segment if parsing fails.
    """
    seq = row['sequence']
    stoich = row.get('stoichiometry', '')
    all_seq = row.get('all_sequences', '')

    if pd.isna(stoich) or pd.isna(all_seq) or str(stoich).strip()=="" or str(all_seq).strip()=="":
        return [(0, len(seq))]

    try:
        chain_dict = parse_fasta(all_seq)  # dict: chain_id -> sequence
        order = parse_stoichiometry(stoich)
        segs = []
        pos = 0
        for ch, cnt in order:
            base = chain_dict.get(ch)
            if base is None:
                return [(0, len(seq))]
            for _ in range(cnt):
                L = len(base)
                segs.append((pos, pos + L))
                pos += L
        if pos != len(seq):
            return [(0, len(seq))]
        return segs
    except Exception:
        return [(0, len(seq))]

def build_segments_map(df):
    seg_map = {}
    stoich_map = {}
    for _, r in df.iterrows():
        tid = r['target_id']
        seg_map[tid] = get_chain_segments(r)
        stoich_map[tid] = str(r.get('stoichiometry', '') if not pd.isna(r.get('stoichiometry', '')) else '')
    return seg_map, stoich_map

train_segs_map, train_stoich_map = build_segments_map(train_seqs)
test_segs_map,  test_stoich_map  = build_segments_map(test_seqs)

def process_labels(labels_df):
    coords_dict = {}
    # Faster + safer prefix extraction
    prefixes = labels_df['ID'].str.rsplit('_', n=1).str[0]
    for id_prefix, group in labels_df.groupby(prefixes):
        coords_dict[id_prefix] = group.sort_values('resid')[['x_1', 'y_1', 'z_1']].values
    return coords_dict

train_coords_dict = process_labels(train_labels)

from Bio.Align import PairwiseAligner

aligner = PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 2
aligner.mismatch_score = -1.5

# Stronger gap penalties discourage "sliding" (critical: residue numbering must match)
aligner.open_gap_score   = -8
aligner.extend_gap_score = -0.4

# Also penalize terminal gaps (prevents end-gap semi-global behavior)
aligner.query_left_open_gap_score  = -8
aligner.query_left_extend_gap_score = -0.4
aligner.query_right_open_gap_score = -8
aligner.query_right_extend_gap_score = -0.4
aligner.target_left_open_gap_score = -8
aligner.target_left_extend_gap_score = -0.4
aligner.target_right_open_gap_score = -8
aligner.target_right_extend_gap_score = -0.4

def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, top_n=5):
    similar_seqs = []
    
    # Pre-filter: Iterate only valid targets
    # Note: aligner.score is much faster than generating full alignments
    for _, row in train_seqs_df.iterrows():
        target_id, train_seq = row['target_id'], row['sequence']
        if target_id not in train_coords_dict: continue
        
        # Length filter (keep your original logic)
        if abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq)) > 0.3: continue
        
        # FAST SCORE: Calculates score without traceback overhead
        raw_score = aligner.score(query_seq, train_seq)
        
        normalized_score = raw_score / (2 * min(len(query_seq), len(train_seq)))
        similar_seqs.append((target_id, train_seq, normalized_score, train_coords_dict[target_id]))
    
    similar_seqs.sort(key=lambda x: x[2], reverse=True)
    return similar_seqs[:top_n]

def adapt_template_to_query(query_seq, template_seq, template_coords):
    # Generate the alignment object
    # aligner.align returns an iterator; we take the first optimal alignment
    alignment = next(iter(aligner.align(query_seq, template_seq)))
    
    new_coords = np.full((len(query_seq), 3), np.nan)
    
    # VECTORIZED MAPPING:
    # alignment.aligned returns lists of (start, end) tuples for matched segments.
    # This avoids the slow python loop "for char_q, char_t in zip..."
    for (q_start, q_end), (t_start, t_end) in zip(*alignment.aligned):
        # Map the coordinate chunk directly
        t_chunk = template_coords[t_start:t_end]
        
        # Safety check to ensure shapes match (handles edge cases)
        if len(t_chunk) == (q_end - q_start):
            new_coords[q_start:q_end] = t_chunk

    # --- Interpolation Logic (Unchanged) ---
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            prev_v = next((j for j in range(i-1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            next_v = next((j for j in range(i+1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if prev_v >= 0 and next_v >= 0:
                w = (i - prev_v) / (next_v - prev_v)
                new_coords[i] = (1-w)*new_coords[prev_v] + w*new_coords[next_v]
            elif prev_v >= 0: new_coords[i] = new_coords[prev_v] + [3, 0, 0]
            elif next_v >= 0: new_coords[i] = new_coords[next_v] + [3, 0, 0]
            else: new_coords[i] = [i*3, 0, 0]
            
    return np.nan_to_num(new_coords)

def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    refined_coords = coordinates.copy()
    n = len(sequence)
    strength = 0.68 * (1.0 - min(confidence, 0.96))

    for _ in range(2):
        for i in range(n - 1):
            p1, p2 = refined_coords[i], refined_coords[i+1]
            dist = np.linalg.norm(p2 - p1)
            if dist > 0:
                adj = (5.95 - dist) * strength * 0.45
                refined_coords[i+1] += (p2 - p1) / dist * adj

            if i < n - 2:
                p3 = refined_coords[i+2]
                dist2 = np.linalg.norm(p3 - p1)
                if dist2 > 0:
                    adj2 = (10.2 - dist2) * strength * 0.25
                    refined_coords[i+2] += (p3 - p1) / dist2 * adj2

    return refined_coords

def predict_rna_structures(sequence, target_id, train_seqs_df, train_coords_dict, n_predictions=5):
    predictions = []
    similar_seqs = find_similar_sequences(sequence, train_seqs_df, train_coords_dict, top_n=n_predictions)
    
    for i in range(n_predictions):
        if i < len(similar_seqs):
            t_id, t_seq, sim, t_coords = similar_seqs[i]
            adapted = adapt_template_to_query(sequence, t_seq, t_coords)
            refined = adaptive_rna_constraints(adapted, sequence, confidence=sim)
            
            # ШУМ: Слот 0 - чистый, остальные - микро-шум
            noise = 0.0 if i == 0 else max(0.006, (0.38 - sim) * 0.07)
            if noise > 0: refined += np.random.normal(0, noise, refined.shape)
            predictions.append(refined)
        else:
            n = len(sequence)
            coords = np.zeros((n, 3))
            for j in range(1, n): coords[j] = coords[j-1] + [4.0, 0, 0]
            predictions.append(coords)
    return predictions

all_predictions = []
start_time = time.time()
for idx, row in test_seqs.iterrows():
    if idx % 10 == 0: print(f"Processing {idx} | {time.time()-start_time:.1f}s")
    tid, seq = row['target_id'], row['sequence']
    preds = predict_rna_structures(seq, tid, train_seqs, train_coords_dict)
    for j in range(len(seq)):
        res = {'ID': f"{tid}_{j+1}", 'resname': seq[j], 'resid': j+1}
        for i in range(5):
            res[f'x_{i+1}'], res[f'y_{i+1}'], res[f'z_{i+1}'] = preds[i][j]
        all_predictions.append(res)

sub = pd.DataFrame(all_predictions)
cols = ['ID', 'resname', 'resid'] + [f'{c}_{i}' for i in range(1,6) for c in ['x','y','z']]
sub[cols].to_csv('submission.csv', index=False)
print("submission.csv! saved")
```
# [0.409] Stanford RNA Folding 2: Protenix+ Template

- **Author:** parthenos
- **Votes:** 203
- **Ref:** nihilisticneuralnet/0-409-stanford-rna-folding-2-protenix-template
- **URL:** https://www.kaggle.com/code/nihilisticneuralnet/0-409-stanford-rna-folding-2-protenix-template
- **Last run:** 2026-02-16 17:22:16.737000

---

```python
import os
import sys

IS_KAGGLE = True
DATA_PATH = '/kaggle/input/stanford-rna-3d-folding-2/'
OUTPUT_PATH = '/kaggle/working/output'
USALIGN_BIN = '/kaggle/working/USalign'
PROTENIX_DIR = '/kaggle/working/Protenix'
! cp /kaggle/input/protenix-packages/packages/USalign /kaggle/working/
! chmod +x /kaggle/working/USalign
sys.path.insert(0, '/kaggle/input/rna-3d-utils/')

print(f"Data path: {DATA_PATH}")
print(f"Protenix dir: {PROTENIX_DIR}")
```

```python
if IS_KAGGLE:
    !cp -r /kaggle/input/protenix-packages/packages /kaggle/working
    %cd /kaggle/working/packages
    !pip install --no-deps --exists-action=i *.whl
    %cd /kaggle/working

    !mv /kaggle/working/packages/ihm-2.3/ihm-2.3 /kaggle/working
    !mv /kaggle/working/packages/modelcif-0.7/modelcif-0.7 /kaggle/working

    !pip install /kaggle/working/ihm-2.3
    !pip install /kaggle/working/modelcif-0.7

    !rm -rf /kaggle/working/ihm-2.3
    !rm -rf /kaggle/working/modelcif-0.7

    !pip install /kaggle/input/biopython/biopython-1.85-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    !pip install /kaggle/input/ml-collections/ml_collections-1.0.0-py3-none-any.whl

    !rm -rf /kaggle/working/packages

    !cp -r /kaggle/input/protenix-mg-packages/protenix_mg_packages /kaggle/working
    %cd /kaggle/working/protenix_mg_packages
    !pip install --no-deps --exists-action=i *.whl
    %cd /kaggle/working
    !rm -rf /kaggle/working/protenix_mg_packages

    !cp -R /kaggle/input/protenix-rmsa-repo/protenix_kaggle /kaggle/working/
    !mv protenix_kaggle Protenix
```

```python
import json
import numpy as np
import pandas as pd
import time
import random
import warnings
import contextlib
from pathlib import Path
from Bio.Align import PairwiseAligner

warnings.filterwarnings('ignore')

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(42)
```

```python
train_seqs = pd.read_csv(DATA_PATH + 'train_sequences.csv')
validation_seqs = pd.read_csv(DATA_PATH + 'validation_sequences.csv')
test_seqs = pd.read_csv(DATA_PATH + 'test_sequences.csv')
train_labels = pd.read_csv(DATA_PATH + 'train_labels.csv')
validation_labels = pd.read_csv(DATA_PATH + 'validation_labels.csv')

SHOW_VALIDATION = False
MAKE_SUBMISSION = True
USE_PROTENIX = True

MIN_SIMILARITY = 0.45
MIN_PERCENT_IDENTITY = 55
TOP_TEMPLATES = 10
ENSEMBLE_WEIGHTS = [0.40, 0.25, 0.15, 0.12, 0.08]
```

```python
def make_aligner():
    al = PairwiseAligner()
    al.mode = 'global'
    al.match_score = 2.5
    al.mismatch_score = -2.0
    al.open_gap_score = -10
    al.extend_gap_score = -0.5
    al.query_left_open_gap_score = -10
    al.query_left_extend_gap_score = -0.5
    al.query_right_open_gap_score = -10
    al.query_right_extend_gap_score = -0.5
    al.target_left_open_gap_score = -10
    al.target_left_extend_gap_score = -0.5
    al.target_right_open_gap_score = -10
    al.target_right_extend_gap_score = -0.5
    return al

_aligner = make_aligner()

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
            cur = line[1:].split()[0]
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
    seq = row['sequence']
    stoich = row.get('stoichiometry', '')
    all_seq = row.get('all_sequences', '')
    
    if pd.isna(stoich) or pd.isna(all_seq) or str(stoich).strip() == "" or str(all_seq).strip() == "":
        return [(0, len(seq))]
    
    try:
        chain_dict = parse_fasta(all_seq)
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

def process_labels(labels_df):
    coords_dict = {}
    prefixes = labels_df['ID'].str.rsplit('_', n=1).str[0]
    for id_prefix, group in labels_df.groupby(prefixes):
        coords_dict[id_prefix] = group.sort_values('resid')[['x_1', 'y_1', 'z_1']].values
    return coords_dict

def compute_sequence_features(seq):
    length = len(seq)
    gc_content = (seq.count('G') + seq.count('C')) / length if length > 0 else 0
    au_content = (seq.count('A') + seq.count('U')) / length if length > 0 else 0
    return {
        'length': length,
        'gc_content': gc_content,
        'au_content': au_content,
        'complexity': len(set(seq)) / 4.0
    }

def enhanced_template_selection(query_seq, train_seqs_df, train_coords_dict, temporal_cutoff=None, top_n=10):
    similar_seqs = []
    query_features = compute_sequence_features(query_seq)
    
    if temporal_cutoff is not None:
        filtered = train_seqs_df[train_seqs_df['temporal_cutoff'] < temporal_cutoff]
    else:
        filtered = train_seqs_df
    
    for _, row in filtered.iterrows():
        target_id, train_seq = row['target_id'], row['sequence']
        if target_id not in train_coords_dict:
            continue
        
        len_ratio = abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq))
        if len_ratio > 0.4:
            continue
        
        template_features = compute_sequence_features(train_seq)
        feature_similarity = 1.0 - abs(query_features['gc_content'] - template_features['gc_content'])
        
        alignment = next(iter(_aligner.align(query_seq, train_seq)))
        raw_score = alignment.score
        normalized_score = raw_score / (2 * min(len(query_seq), len(train_seq)))
        
        identical = 0
        for (qs, qe), (ts, te) in zip(*alignment.aligned):
            for q_pos, t_pos in zip(range(qs, qe), range(ts, te)):
                if query_seq[q_pos] == train_seq[t_pos]:
                    identical += 1
        percent_identity = 100 * identical / len(query_seq)
        
        combined_score = 0.7 * normalized_score + 0.2 * (percent_identity / 100) + 0.1 * feature_similarity
        
        aligned_query, aligned_template = _build_aligned_strings(query_seq, train_seq, alignment)
        
        similar_seqs.append((
            target_id, train_seq, combined_score, normalized_score,
            train_coords_dict[target_id], percent_identity,
            aligned_query, aligned_template
        ))
    
    similar_seqs.sort(key=lambda x: x[2], reverse=True)
    return similar_seqs[:top_n]

def _build_aligned_strings(query_seq, template_seq, alignment):
    q_segments, t_segments = alignment.aligned
    aligned_q = []
    aligned_t = []
    qi = 0
    ti = 0
    
    for (qs, qe), (ts, te) in zip(q_segments, t_segments):
        while qi < qs:
            aligned_q.append(query_seq[qi])
            aligned_t.append('-')
            qi += 1
        while ti < ts:
            aligned_q.append('-')
            aligned_t.append(template_seq[ti])
            ti += 1
        for q_pos, t_pos in zip(range(qs, qe), range(ts, te)):
            aligned_q.append(query_seq[q_pos])
            aligned_t.append(template_seq[t_pos])
        qi = qe
        ti = te
    
    while qi < len(query_seq):
        aligned_q.append(query_seq[qi])
        aligned_t.append('-')
        qi += 1
    while ti < len(template_seq):
        aligned_q.append('-')
        aligned_t.append(template_seq[ti])
        ti += 1
    
    return ''.join(aligned_q), ''.join(aligned_t)

def adapt_template_to_query(query_seq, template_seq, template_coords):
    alignment = next(iter(_aligner.align(query_seq, template_seq)))
    new_coords = np.full((len(query_seq), 3), np.nan)
    
    for (q_start, q_end), (t_start, t_end) in zip(*alignment.aligned):
        t_chunk = template_coords[t_start:t_end]
        if len(t_chunk) == (q_end - q_start):
            new_coords[q_start:q_end] = t_chunk
    
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            prev_v = next((j for j in range(i - 1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            next_v = next((j for j in range(i + 1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if prev_v >= 0 and next_v >= 0:
                w = (i - prev_v) / (next_v - prev_v)
                new_coords[i] = (1 - w) * new_coords[prev_v] + w * new_coords[next_v]
            elif prev_v >= 0:
                new_coords[i] = new_coords[prev_v] + [3.8, 0, 0]
            elif next_v >= 0:
                new_coords[i] = new_coords[next_v] + [3.8, 0, 0]
            else:
                new_coords[i] = [i * 3.8, 0, 0]
    
    return np.nan_to_num(new_coords)

def enhanced_rna_constraints(coordinates, target_id, segments_map, confidence=1.0, passes=3):
    coords = coordinates.copy()
    segments = segments_map.get(target_id, [(0, len(coords))])
    
    base_strength = 0.85 * (1.0 - min(confidence, 0.98))
    base_strength = max(base_strength, 0.01)
    
    for pass_idx in range(passes):
        strength = base_strength * (1.0 - pass_idx * 0.15)
        
        for (s, e) in segments:
            X = coords[s:e]
            L = e - s
            if L < 3:
                coords[s:e] = X
                continue
            
            d = X[1:] - X[:-1]
            dist = np.linalg.norm(d, axis=1) + 1e-6
            target_bond = 5.9
            scale = (target_bond - dist) / dist
            adj = (d * scale[:, None]) * (0.28 * strength)
            X[:-1] -= adj
            X[1:] += adj
            
            if L >= 3:
                d2 = X[2:] - X[:-2]
                dist2 = np.linalg.norm(d2, axis=1) + 1e-6
                target2 = 10.4
                scale2 = (target2 - dist2) / dist2
                adj2 = (d2 * scale2[:, None]) * (0.15 * strength)
                X[:-2] -= adj2
                X[2:] += adj2
            
            if L >= 4:
                lap = 0.5 * (X[:-2] + X[2:]) - X[1:-1]
                X[1:-1] += (0.08 * strength) * lap
            
            if L >= 6:
                backbone_smooth = 0.33 * (X[:-2] + X[1:-1] + X[2:])
                X[1:-1] = (1 - 0.12 * strength) * X[1:-1] + (0.12 * strength) * backbone_smooth
            
            if L >= 30:
                k = min(L, 180) if L > 250 else L
                if k < L:
                    idx = np.linspace(0, L - 1, k).astype(int)
                else:
                    idx = np.arange(L)
                
                P = X[idx]
                diff = P[:, None, :] - P[None, :, :]
                distm = np.linalg.norm(diff, axis=2) + 1e-6
                sep = np.abs(idx[:, None] - idx[None, :])
                
                mask = (sep > 2) & (distm < 3.5)
                if np.any(mask):
                    force = (3.5 - distm) / distm
                    vec = (diff * force[:, :, None] * mask[:, :, None]).sum(axis=1)
                    X[idx] += (0.018 * strength) * vec
            
            coords[s:e] = X
    
    return coords

def _rotmat(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(ang), np.sin(ang)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ], dtype=float)

def apply_hinge(coords, seg, rng, max_angle_deg=20):
    s, e = seg
    L = e - s
    if L < 35:
        return coords
    pivot = s + int(rng.integers(12, L - 12))
    axis = rng.normal(size=3)
    ang = np.deg2rad(float(rng.uniform(-max_angle_deg, max_angle_deg)))
    R = _rotmat(axis, ang)
    X = coords.copy()
    p0 = X[pivot].copy()
    X[pivot + 1:e] = (X[pivot + 1:e] - p0) @ R.T + p0
    return X

def weighted_ensemble_prediction(templates, query_seq, segments_map, target_id):
    if not templates:
        return None
    
    predictions = []
    weights = ENSEMBLE_WEIGHTS[:len(templates)]
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    for (tmpl_id, tmpl_seq, combined_score, sim, tmpl_coords, pct_id, _, _), weight in zip(templates, weights):
        adapted = adapt_template_to_query(query_seq, tmpl_seq, tmpl_coords)
        refined = enhanced_rna_constraints(adapted, target_id, segments_map, confidence=sim, passes=3)
        predictions.append((refined, weight))
    
    if len(predictions) == 1:
        return predictions[0][0]
    
    ensemble_coords = np.zeros_like(predictions[0][0])
    for coords, weight in predictions:
        ensemble_coords += weight * coords
    
    final = enhanced_rna_constraints(ensemble_coords, target_id, segments_map, confidence=0.85, passes=2)
    
    return final

def generate_rna_structure(sequence, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = len(sequence)
    coords = np.zeros((n, 3))
    for i in range(n):
        angle = i * 0.58
        coords[i] = [11.0 * np.cos(angle), 11.0 * np.sin(angle), i * 2.6]
    return coords

train_coords_dict = process_labels(train_labels)
combined_seqs = pd.concat([train_seqs, validation_seqs], ignore_index=True)
combined_labels = pd.concat([train_labels, validation_labels], ignore_index=True)
combined_coords_dict = process_labels(combined_labels)

validation_segments_map, _ = build_segments_map(validation_seqs)
test_segments_map, _ = build_segments_map(test_seqs)

from biotite.structure.io.pdbx import CIFFile, get_structure
import contextlib

def extract_c1_atoms(cif_path):
    cif_file = CIFFile.read(cif_path)
    model = get_structure(cif_file, model=1)
    chain = model[model.chain_id == "A"]
    mask = chain.atom_name == "C1'"
    c1_atoms = chain[mask]
    df = pd.DataFrame.from_dict(c1_atoms._annot)
    df["x"] = c1_atoms.coord[:, 0]
    df["y"] = c1_atoms.coord[:, 1]
    df["z"] = c1_atoms.coord[:, 2]
    return df[["res_name", "res_id", "x", "y", "z"]]

def prepare_protenix_json(target_id, sequence, output_path, input_path, max_length=400):
    if len(sequence) <= max_length:
        input_json = [{
            "sequences": [{
                "rnaSequence": {
                    "sequence": sequence,
                    "count": 1,
                    "msa": {
                        "precomputed_msa_dir": f"{input_path}/MSA/{target_id}.MSA.fasta",
                        "pairing_db": "rnacentral"
                    }
                }
            }],
            "name": target_id,
        }]
    else:
        input_json = [{
            "sequences": [{
                "rnaSequence": {
                    "sequence": sequence[:max_length],
                    "count": 1,
                }
            }],
            "name": target_id,
        }]
    
    json_path = Path(output_path) / "input_json" / f"{target_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(input_json, f, indent=4)

def run_protenix_inference(target_id, sequence, output_path, input_path,
                           seed=101, n_cycle=12, n_sample=5, n_step=250, max_length=400):
    if IS_KAGGLE:
        checkpoint_path = "/kaggle/input/protenix-finetuned-rna3db-all-1599/1599_ema_0.999.pt"
    else:
        checkpoint_path = f"{DATA_PATH}/protenix_chpt/1599_ema_0.999.pt"
    
    output_path = Path(output_path)
    input_json_path = output_path / "input_json" / f"{target_id}.json"
    dump_dir = output_path / target_id
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    use_msa = "True" if len(sequence) <= max_length else "False"
    
    sys.argv = [
        "runner/inference.py",
        f"--seeds={seed}",
        f"--dump_dir={dump_dir}",
        f"--input_json_path={input_json_path}",
        f"--model.N_cycle={n_cycle}",
        f"--sample_diffusion.N_sample={n_sample}",
        f"--sample_diffusion.N_step={n_step}",
        "--augment.use_rnalm True",
        f"--use_msa {use_msa}",
        f"--load_checkpoint_path={checkpoint_path}",
        "",
    ]
    
    from runner.inference import run
    run()

def get_protenix_predictions(target_id, sequence, output_path, seed=101, n_sample=5):
    output_path = Path(output_path)
    predictions = []
    for i in range(n_sample):
        cif_path = (output_path / target_id / target_id / f"seed_{seed}" /
                    "predictions" / f"{target_id}_seed_{seed}_sample_{i}.cif")
        if cif_path.exists():
            pred_df = extract_c1_atoms(cif_path)
            coords = np.zeros((len(sequence), 3))
            n_atoms = min(len(pred_df), len(sequence))
            coords[:n_atoms] = pred_df[["x", "y", "z"]].values[:n_atoms]
            predictions.append(coords)
    return predictions

@contextlib.contextmanager
def protenix_context():
    original_dir = os.getcwd()
    os.chdir(PROTENIX_DIR)
    try:
        yield
    finally:
        os.chdir(original_dir)

template_info_dict = {}
prediction_metadata_dict = {}

def record_template_info(target_id, template_id, similarity, percent_identity):
    if target_id not in template_info_dict:
        template_info_dict[target_id] = {
            'template_ids': [], 'similarities': [], 'percent_identities': []
        }
    template_info_dict[target_id]['template_ids'].append(template_id)
    template_info_dict[target_id]['similarities'].append(similarity)
    template_info_dict[target_id]['percent_identities'].append(percent_identity)

def record_prediction_metadata(target_id, pred_num, source, template_id=None,
                               similarity=None, percent_identity=None):
    if target_id not in prediction_metadata_dict:
        prediction_metadata_dict[target_id] = {}
    prediction_metadata_dict[target_id][pred_num] = {
        'source': source,
        'template_id': template_id if source == 'template' else None,
        'similarity': similarity if source == 'template' else None,
        'percent_identity': percent_identity if source == 'template' else None,
    }

def predict_with_enhanced_templates(sequence, target_id, train_seqs_df, train_coords_dict,
                                   segments_map, n_predictions=5, temporal_cutoff=None):
    predictions = []
    pred_num = 1
    
    print(f"\nTarget: {target_id} ({len(sequence)} nt)")
    
    similar_seqs = enhanced_template_selection(
        sequence, train_seqs_df, train_coords_dict,
        temporal_cutoff=temporal_cutoff, top_n=TOP_TEMPLATES
    )
    
    if similar_seqs:
        top_templates = []
        for i, (tmpl_id, tmpl_seq, combined_score, similarity, tmpl_coords,
                pct_id, aligned_q, aligned_t) in enumerate(similar_seqs):
            
            if (similarity < MIN_SIMILARITY or pct_id < MIN_PERCENT_IDENTITY) and len(tmpl_seq) < 500:
                print(f"  Template {i+1}: {tmpl_id} SKIPPED (sim={similarity:.3f}, id={pct_id:.1f}%)")
                break
            
            if USE_PROTENIX and len(aligned_q) < 100 and i == 4:
                print(f"  Leaving 1 slot for Protenix")
                break
            
            record_template_info(target_id, tmpl_id, similarity, pct_id)
            print(f"  Template {i+1}: {tmpl_id} (sim={similarity:.3f}, id={pct_id:.1f}%)")
            
            top_templates.append((tmpl_id, tmpl_seq, combined_score, similarity, 
                                 tmpl_coords, pct_id, aligned_q, aligned_t))
            
            if len(top_templates) >= min(5, n_predictions):
                break
        
        if top_templates:
            ensemble_pred = weighted_ensemble_prediction(top_templates, sequence, 
                                                        segments_map, target_id)
            if ensemble_pred is not None:
                record_prediction_metadata(target_id, pred_num, 'ensemble_template',
                                         top_templates[0][0], top_templates[0][3], 
                                         top_templates[0][5])
                predictions.append(ensemble_pred)
                pred_num += 1
    
    n_from_templates = len(predictions)
    n_needed = n_predictions - n_from_templates
    
    if n_needed > 0:
        print(f"  -> {n_from_templates} ensemble pred, {n_needed} slots for Protenix")
    else:
        print(f"  -> {n_predictions} predictions from ensemble")
    
    return predictions, n_needed, pred_num

def generate_predictions_batch(sequences_df, train_seqs_df, train_coords_dict,
                               dataset_name, use_temporal_cutoff=True,
                               protenix_output_path=None):
    start_time = time.time()
    total_targets = len(sequences_df)
    
    print(f"\n{'='*70}")
    print(f"Predicting {total_targets} {dataset_name} sequences")
    print(f"{'='*70}")
    
    segments_map, _ = build_segments_map(sequences_df)
    
    print("\nPHASE 1: Enhanced template-based ensemble predictions")
    
    template_predictions = {}
    protenix_queue = {}
    
    for _, row in sequences_df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        temporal_cutoff = row.get('temporal_cutoff', None) if use_temporal_cutoff else None
        
        preds, n_needed, next_pred = predict_with_enhanced_templates(
            sequence, target_id, train_seqs_df, train_coords_dict,
            segments_map, n_predictions=5, temporal_cutoff=temporal_cutoff
        )
        
        template_predictions[target_id] = preds
        if n_needed > 0:
            protenix_queue[target_id] = (n_needed, next_pred, sequence)
    
    template_time = time.time() - start_time
    print(f"\nPhase 1 done: {template_time:.1f}s | {len(protenix_queue)} targets need Protenix")
    
    protenix_predictions = {}
    
    if protenix_queue and USE_PROTENIX:
        print(f"\nPHASE 2: Protenix for {len(protenix_queue)} targets")
        
        if protenix_output_path is None:
            protenix_output_path = Path(OUTPUT_PATH) / f"{dataset_name}_protenix"
        protenix_output_path = Path(protenix_output_path)
        protenix_output_path.mkdir(parents=True, exist_ok=True)
        input_path = Path(DATA_PATH)
        
        with protenix_context():
            for i, (target_id, (n_needed, next_pred, sequence)) in enumerate(protenix_queue.items()):
                print(f"\n  [{i+1}/{len(protenix_queue)}] {target_id} ({len(sequence)} nt, need {n_needed})")
                
                try:
                    prepare_protenix_json(target_id, sequence, protenix_output_path, input_path)
                    
                    t0 = time.time()
                    run_protenix_inference(
                        target_id, sequence, protenix_output_path, input_path,
                        seed=101, n_cycle=12, n_sample=n_needed, n_step=250
                    )
                    print(f"    Done in {(time.time()-t0)/60:.1f} min")
                    
                    preds = get_protenix_predictions(
                        target_id, sequence, protenix_output_path,
                        seed=101, n_sample=n_needed
                    )
                    protenix_predictions[target_id] = preds
                    print(f"    Got {len(preds)} Protenix predictions")
                    
                except Exception as e:
                    print(f"    Protenix FAILED: {e}")
                    protenix_predictions[target_id] = None
        
        ptx_time = time.time() - start_time - template_time
        print(f"\nPhase 2 done: {ptx_time/60:.1f} min")
    
    elif protenix_queue and not USE_PROTENIX:
        print(f"\nPHASE 2: Protenix disabled, will use de novo for {len(protenix_queue)} targets")
    
    print(f"\nPHASE 3: Combining predictions")
    
    all_rows = []
    
    for _, row in sequences_df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        
        predictions = list(template_predictions[target_id])
        pred_num = len(predictions) + 1
        
        if target_id in protenix_queue:
            ptx_preds = protenix_predictions.get(target_id)
            if ptx_preds:
                for coords in ptx_preds:
                    record_prediction_metadata(target_id, pred_num, 'protenix')
                    predictions.append(coords)
                    pred_num += 1
                    if len(predictions) >= 5:
                        break
        
        n_denovo = 0
        while len(predictions) < 5:
            record_prediction_metadata(target_id, pred_num, 'de_novo')
            seed_val = hash(target_id) % 10000 + len(predictions) * 1000
            de_novo = generate_rna_structure(sequence, seed=seed_val)
            refined = enhanced_rna_constraints(de_novo, target_id, segments_map, confidence=0.15, passes=3)
            predictions.append(refined)
            pred_num += 1
            n_denovo += 1
        
        if n_denovo > 0:
            print(f"  {target_id}: filled {n_denovo} slots with de novo")
        
        for j in range(len(sequence)):
            pred_row = {
                'ID': f"{target_id}_{j+1}",
                'resname': sequence[j],
                'resid': j + 1,
            }
            for i in range(5):
                pred_row[f'x_{i+1}'] = predictions[i][j][0]
                pred_row[f'y_{i+1}'] = predictions[i][j][1]
                pred_row[f'z_{i+1}'] = predictions[i][j][2]
            all_rows.append(pred_row)
    
    submission_df = pd.DataFrame(all_rows)
    column_order = ['ID', 'resname', 'resid']
    for i in range(1, 6):
        for coord in ['x', 'y', 'z']:
            column_order.append(f'{coord}_{i}')
    submission_df = submission_df[column_order]
    
    total_time = time.time() - start_time
    n_template_only = sum(1 for tid in sequences_df['target_id'] if tid not in protenix_queue)
    
    print(f"\n{'='*70}")
    print(f"{dataset_name.upper()} PREDICTIONS COMPLETE")
    print(f"  Ensemble template targets: {n_template_only}")
    print(f"  Targets with Protenix: {len(protenix_queue)}")
    print(f"  Total residues: {len(submission_df)}")
    print(f"  Runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}\n")
    
    return submission_df
```

```python
template_info_dict.clear()
prediction_metadata_dict.clear()

test_predictions = generate_predictions_batch(
    test_seqs,
    combined_seqs,
    combined_coords_dict,
    dataset_name="test",
    use_temporal_cutoff=False,
)

test_predictions.to_csv('submission.csv', index=False)
print("Saved: submission.csv")
```
# Stanford RNA 3D Folding Part 2|Protenix+TBM

- **Author:** Kh0a
- **Votes:** 232
- **Ref:** llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm
- **URL:** https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm
- **Last run:** 2026-02-20 09:41:33.140000

---

Reference

https://www.kaggle.com/code/qiweiyin/protenix-v1-inference-2026

https://www.kaggle.com/code/nihilisticneuralnet/0-409-stanford-rna-folding-2-protenix-template

https://www.kaggle.com/code/alexxanderlarko/protenix-v1

| version | USE_MSA | USE_RNA_MSA | USE_TEMPLATE | TBM | LB    |
|--------:|:-------:|:-----------:|:------------:|:---:|:------|
| [3](https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm?scriptVersionId=298498006) | ❌ | ❌ | ❌ | ✅ | 0.408 |
| [4](https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm?scriptVersionId=298498822) | ❌ | ❌ | ❌ | ❌ | 0.249 |
| [5](https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm?scriptVersionId=298660741) | ✅ | ✅ | ❌ | ✅ | 0.396 |
| [6](https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm?scriptVersionId=298662848) | ✅ | ❌ | ❌ | ✅ | 0.402 |
| [7](https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm?scriptVersionId=298662875) | ❌ | ✅ | ❌ | ✅ | 0.413 |
| [8](https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm?scriptVersionId=298669207) | ✅ | ✅ | ✅ | ✅ | 0.406 |
| [9](https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm?scriptVersionId=298699592) | ❌ | ✅ | ✅ | ✅ | 0.405 |

```python
import os
import sys
import pandas as pd

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
# On Kaggle competition rerun, KAGGLE_IS_COMPETITION_RERUN is set to a truthy value.
# When running locally we do NOT exit — instead we cap the test set to a small
# number of samples so the notebook finishes quickly.

IS_KAGGLE = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))

# How many test samples to use when running locally
LOCAL_N_SAMPLES = 2

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode — all test targets will be processed.")
else:
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets "
          f"will be processed to save time.")
```

```python
import gc
import json
import os
import time

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm
```

```python
def get_c1_mask(data: dict, atom_array) -> torch.Tensor:
    # 1. Try atom_array attributes first
    if atom_array is not None:
        try:
            if hasattr(atom_array, "centre_atom_mask"):
                m = atom_array.centre_atom_mask == 1
                if hasattr(atom_array, "is_rna"):
                    m = m & atom_array.is_rna
                return torch.from_numpy(m).bool()
            
            if hasattr(atom_array, "atom_name"):
                base = atom_array.atom_name == "C1'"
                if hasattr(atom_array, "is_rna"):
                    base = base & atom_array.is_rna
                return torch.from_numpy(base).bool()
        except Exception:
            pass

    # 2. Fallback to feature dict
    f = data["input_feature_dict"]
    
    if "centre_atom_mask" in f:
        return (f["centre_atom_mask"] == 1).bool()
    if "center_atom_mask" in f:
        return (f["center_atom_mask"] == 1).bool()
        
    # Heuristic fallback: check which index gives us roughly N_token atoms
    n_tokens = data.get("N_token", torch.tensor(0)).item()
    mask11 = (f["atom_to_tokatom_idx"] == 11).bool()
    mask12 = (f["atom_to_tokatom_idx"] == 12).bool()
    
    c11 = mask11.sum().item()
    c12 = mask12.sum().item()
    
    # Return the one closer to N_tokens (likely one per residue)
    if abs(c11 - n_tokens) < abs(c12 - n_tokens):
        return mask11
    else:
        return mask12
```

```python
# ─────────────── Paths & Constants ───────────────────────────────────────────
DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS     = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV        = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS       = f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"

DEFAULT_CODE_DIR = (
    "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted"
    "/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
)
DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR

MODEL_NAME    = "protenix_base_20250630_v1.0.0"
N_SAMPLE      = 5
SEED          = 42
MAX_SEQ_LEN   = int(os.environ.get("MAX_SEQ_LEN",   "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP",  "64"))

# TBM quality thresholds — sequences below these get routed to Protenix
MIN_SIMILARITY       = float(os.environ.get("MIN_SIMILARITY",       "0.0"))
MIN_PERCENT_IDENTITY = float(os.environ.get("MIN_PERCENT_IDENTITY", "50.0"))

# Set False to skip Protenix and use de-novo fallback instead
USE_PROTENIX = True


def parse_bool(value: str, default: bool = False) -> str:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return "true"
    if v in {"0", "false", "f", "no", "n", "off"}:
        return "false"
    return "true" if default else "false"


USE_MSA      = parse_bool(os.environ.get("USE_MSA",      "false"))
USE_TEMPLATE = parse_bool(os.environ.get("USE_TEMPLATE", "false"))
USE_RNA_MSA  = parse_bool(os.environ.get("USE_RNA_MSA",  "true"))

MODEL_N_SAMPLE = int(os.environ.get("MODEL_N_SAMPLE", str(N_SAMPLE)))


# ─────────────── General Utilities ───────────────────────────────────────────
def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True)


def resolve_paths():
    test_csv   = os.environ.get("TEST_CSV",           DEFAULT_TEST_CSV)
    output_csv = os.environ.get("SUBMISSION_CSV",     DEFAULT_OUTPUT)
    code_dir   = os.environ.get("PROTENIX_CODE_DIR",  DEFAULT_CODE_DIR)
    
    # FIX: Check if we created a writable root in previous cells
    if os.path.isdir("/kaggle/working/protenix_data"):
        root_dir = "/kaggle/working/protenix_data"
    elif os.path.isdir("/kaggle/working/protenix_root"):
        root_dir = "/kaggle/working/protenix_root"
    else:
        root_dir = os.environ.get("PROTENIX_ROOT_DIR",  DEFAULT_ROOT_DIR)
        
    return test_csv, output_csv, code_dir, root_dir


def ensure_required_files(root_dir: str) -> None:
    for p, name in [
        (Path(root_dir) / "checkpoint" / f"{MODEL_NAME}.pt",          "checkpoint"),
        (Path(root_dir) / "common" / "components.cif",                "CCD file"),
        (Path(root_dir) / "common" / "components.cif.rdkit_mol.pkl",  "CCD cache"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")


# ─────────────── Protenix Input / Config Helpers ─────────────────────────────
def build_input_json(df: pd.DataFrame, json_path: str) -> None:
    data = [
        {
            "name": row["target_id"],
            "covalent_bonds": [],
            "sequences": [{"rnaSequence": {"sequence": row["sequence"], "count": 1}}],
        }
        for _, row in df.iterrows()
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def build_configs(input_json_path: str, dump_dir: str, model_name: str):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs

    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(t, p):
        for k, v in p.items():
            if isinstance(v, dict) and k in t and isinstance(t[k], dict):
                deep_update(t[k], v)
            else:
                t[k] = v

    deep_update(base, model_configs[model_name])
    arg_str = " ".join([
        f"--model_name {model_name}",
        f"--input_json_path {input_json_path}",
        f"--dump_dir {dump_dir}",
        f"--use_msa {USE_MSA}",
        f"--use_template {USE_TEMPLATE}",
        f"--use_rna_msa {USE_RNA_MSA}",
        f"--sample_diffusion.N_sample {MODEL_N_SAMPLE}",
        f"--seeds {SEED}",
    ])
    return parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)


def get_c1_mask(data: dict, atom_array) -> torch.Tensor:
    # 1. Try atom_array attributes first
    if atom_array is not None:
        try:
            if hasattr(atom_array, "centre_atom_mask"):
                m = atom_array.centre_atom_mask == 1
                if hasattr(atom_array, "is_rna"):
                    m = m & atom_array.is_rna
                return torch.from_numpy(m).bool()
            
            if hasattr(atom_array, "atom_name"):
                base = atom_array.atom_name == "C1'"
                if hasattr(atom_array, "is_rna"):
                    base = base & atom_array.is_rna
                return torch.from_numpy(base).bool()
        except Exception:
            pass

    # 2. Fallback to feature dict
    f = data["input_feature_dict"]
    
    # CASE A: center_atom_mask exists
    if "center_atom_mask" in f:
        return (f["center_atom_mask"] == 1).bool()
    if "centre_atom_mask" in f:
        return (f["centre_atom_mask"] == 1).bool()
        
    # CASE B: Use atom_name
    if "atom_name" in f:
        # Check against "C1'" (byte encoded or string?)
        # For now assume typical behavior is center_atom_mask is present.
        pass

    # CASE C: atom_to_tokatom_idx fallback
    # The index for C1' is typically 11 or 12 depending on featurizer.
    # Let's try to match exactly C1' if possible.
    # But usually 'centre_atom_mask' should be there.
    
    # If we fall through, assume standard mask
    return (f["atom_to_tokatom_idx"] == 11).bool()


def get_feature_c1_mask(data: dict) -> torch.Tensor:
    f = data["input_feature_dict"]
    if "centre_atom_mask" in f:
        return f["centre_atom_mask"].long() == 1
    return f["atom_to_tokatom_idx"].long() == 12


def coords_to_rows(target_id: str, seq: str, coords: np.ndarray) -> list:
    """coords shape: (N_SAMPLE, seq_len, 3)"""
    rows = []
    for i in range(len(seq)):
        row = {"ID": f"{target_id}_{i + 1}", "resname": seq[i], "resid": i + 1}
        for s in range(N_SAMPLE):
            if s < coords.shape[0] and i < coords.shape[1]:
                x, y, z = coords[s, i]
            else:
                x, y, z = 0.0, 0.0, 0.0
            row[f"x_{s + 1}"] = float(x)
            row[f"y_{s + 1}"] = float(y)
            row[f"z_{s + 1}"] = float(z)
        rows.append(row)
    return rows


def pad_samples(coords: np.ndarray, n: int) -> np.ndarray:
    if coords.shape[0] >= n:
        return coords[:n]
    if coords.shape[0] == 0:
        return np.zeros((n, coords.shape[1], 3), dtype=coords.dtype)
    extra = np.repeat(coords[:1], n - coords.shape[0], axis=0)
    return np.concatenate([coords, extra], axis=0)


# ─────────────── TBM Core Functions ──────────────────────────────────────────
def _make_aligner() -> PairwiseAligner:
    al = PairwiseAligner()
    al.mode                           = "global"
    al.match_score                    = 2
    al.mismatch_score                 = -1.5
    al.open_gap_score                 = -8
    al.extend_gap_score               = -0.4
    al.query_left_open_gap_score      = -8
    al.query_left_extend_gap_score    = -0.4
    al.query_right_open_gap_score     = -8
    al.query_right_extend_gap_score   = -0.4
    al.target_left_open_gap_score     = -8
    al.target_left_extend_gap_score   = -0.4
    al.target_right_open_gap_score    = -8
    al.target_right_extend_gap_score  = -0.4
    return al


_aligner = _make_aligner()


def parse_stoichiometry(stoich: str) -> list:
    if pd.isna(stoich) or str(stoich).strip() == "":
        return []
    return [(ch.strip(), int(cnt)) for part in str(stoich).split(";")
            for ch, cnt in [part.split(":")]]


def parse_fasta(fasta_content: str) -> dict:
    out, cur, parts = {}, None, []
    for line in str(fasta_content).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur is not None:
                out[cur] = "".join(parts)
            cur = line[1:].split()[0]
            parts = []
        else:
            parts.append(line.replace(" ", ""))
    if cur is not None:
        out[cur] = "".join(parts)
    return out


def get_chain_segments(row) -> list:
    seq    = row["sequence"]
    stoich = row.get("stoichiometry", "")
    all_sq = row.get("all_sequences", "")
    if (pd.isna(stoich) or pd.isna(all_sq)
            or str(stoich).strip() == "" or str(all_sq).strip() == ""):
        return [(0, len(seq))]
    try:
        chain_dict = parse_fasta(all_sq)
        order = parse_stoichiometry(stoich)
        segs, pos = [], 0
        for ch, cnt in order:
            base = chain_dict.get(ch)
            if base is None:
                return [(0, len(seq))]
            for _ in range(cnt):
                segs.append((pos, pos + len(base)))
                pos += len(base)
        return segs if pos == len(seq) else [(0, len(seq))]
    except Exception:
        return [(0, len(seq))]


def build_segments_map(df: pd.DataFrame) -> tuple:
    seg_map, stoich_map = {}, {}
    for _, r in df.iterrows():
        tid               = r["target_id"]
        seg_map[tid]      = get_chain_segments(r)
        raw_s             = r.get("stoichiometry", "")
        stoich_map[tid]   = "" if pd.isna(raw_s) else str(raw_s)
    return seg_map, stoich_map


def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes):
        coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords


def _build_aligned_strings(query_seq, template_seq, alignment):
    q_segs, t_segs = alignment.aligned
    aq, at, qi, ti = [], [], 0, 0
    for (qs, qe), (ts, te) in zip(q_segs, t_segs):
        while qi < qs: aq.append(query_seq[qi]);    at.append("-");              qi += 1
        while ti < ts: aq.append("-");              at.append(template_seq[ti]); ti += 1
        for qp, tp in zip(range(qs, qe), range(ts, te)):
            aq.append(query_seq[qp]); at.append(template_seq[tp])
        qi, ti = qe, te
    while qi < len(query_seq):    aq.append(query_seq[qi]);    at.append("-");              qi += 1
    while ti < len(template_seq): aq.append("-");              at.append(template_seq[ti]); ti += 1
    return "".join(aq), "".join(at)


def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict:
            continue
        if abs(len(tseq) - len(query_seq)) / max(len(tseq), len(query_seq)) > 0.3:
            continue
        aln       = next(iter(_aligner.align(query_seq, tseq)))
        norm_s    = aln.score / (2 * min(len(query_seq), len(tseq)))
        identical = sum(
            1 for (qs, qe), (ts, te) in zip(*aln.aligned)
            for qp, tp in zip(range(qs, qe), range(ts, te))
            if query_seq[qp] == tseq[tp]
        )
        pct_id = 100 * identical / len(query_seq)
        aq, at = _build_aligned_strings(query_seq, tseq, aln)
        results.append((tid, tseq, norm_s, train_coords_dict[tid], pct_id, aq, at))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]


def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    aln        = next(iter(_aligner.align(query_seq, template_seq)))
    new_coords = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        chunk = template_coords[ts:te]
        if len(chunk) == (qe - qs):
            new_coords[qs:qe] = chunk
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i - 1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i + 1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv >= 0 and nv >= 0:
                w = (i - pv) / (nv - pv)
                new_coords[i] = (1 - w) * new_coords[pv] + w * new_coords[nv]
            elif pv >= 0:
                new_coords[i] = new_coords[pv] + [3, 0, 0]
            elif nv >= 0:
                new_coords[i] = new_coords[nv] + [3, 0, 0]
            else:
                new_coords[i] = [i * 3, 0, 0]
    return np.nan_to_num(new_coords)


def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3:
                continue
            # bond i–i+1  ~5.95 Å
            d    = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj  = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            # soft i–i+2  ~10.2 Å
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            # Laplacian smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            # self-avoidance
            if L >= 25:
                idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 2) & (dm < 3.2)
                if np.any(mask):
                    vec = (diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec
            X[s:e] = C
    return X


def _rotmat(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a) + 1e-12
    x, y, z = a; c, s = np.cos(ang), np.sin(ang); CC = 1 - c
    return np.array([[c+x*x*CC, x*y*CC-z*s, x*z*CC+y*s],
                     [y*x*CC+z*s, c+y*y*CC, y*z*CC-x*s],
                     [z*x*CC-y*s, z*y*CC+x*s, c+z*z*CC]])


def apply_hinge(coords, seg, rng, deg=22):
    s, e = seg; L = e - s
    if L < 30: return coords
    pivot = s + int(rng.integers(10, L - 10))
    R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
    X = coords.copy(); p0 = X[pivot].copy()
    X[pivot+1:e] = (X[pivot+1:e] - p0) @ R.T + p0
    return X


def jitter_chains(coords, segs, rng, deg=12, trans=1.5):
    X = coords.copy(); gc_ = X.mean(0, keepdims=True)
    for s, e in segs:
        R     = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
        shift = rng.normal(size=3); shift = shift / (np.linalg.norm(shift) + 1e-12) * float(rng.uniform(0, trans))
        c     = X[s:e].mean(0, keepdims=True)
        X[s:e] = (X[s:e] - c) @ R.T + c + shift
    X -= X.mean(0, keepdims=True) - gc_
    return X


def smooth_wiggle(coords, segs, rng, amp=0.8):
    X = coords.copy()
    for s, e in segs:
        L = e - s
        if L < 20: continue
        ctrl = np.linspace(0, L - 1, 6); disp = rng.normal(0, amp, (6, 3)); t = np.arange(L)
        X[s:e] += np.vstack([np.interp(t, ctrl, disp[:, k]) for k in range(3)]).T
    return X


def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    """Idealized A-form RNA helix — last-resort de-novo fallback."""
    if seed is not None:
        np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return coords


# ─────────────── TBM Phase ───────────────────────────────────────────────────
def tbm_phase(test_df, train_seqs_df, train_coords_dict, segments_map):
    """
    Phase 1 — Template-Based Modeling.

    Returns
    -------
    template_predictions : {target_id: [np.ndarray(seq_len, 3), ...]}
        0 to N_SAMPLE predictions per target, from real templates.
    protenix_queue : {target_id: (n_needed, full_sequence)}
        Targets that still need more predictions.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1: Template-Based Modeling")
    print(f"  MIN_SIMILARITY = {MIN_SIMILARITY}  |  MIN_PCT_IDENTITY = {MIN_PERCENT_IDENTITY}")
    print(f"{'='*60}")
    t0 = time.time()

    template_predictions: dict = {}
    protenix_queue:       dict = {}

    for _, row in test_df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]
        segs = segments_map.get(tid, [(0, len(seq))])

        similar = find_similar_sequences_detailed(seq, train_seqs_df, train_coords_dict, top_n=30)
        preds   = []
        used    = set()

        for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id, _, _) in enumerate(similar):
            if len(preds) >= N_SAMPLE:
                break
            if sim < MIN_SIMILARITY or pct_id < MIN_PERCENT_IDENTITY:
                break           # list is sorted by sim, so no point continuing
            if tmpl_id in used:
                continue

            rng     = np.random.default_rng((abs(hash(tid)) + i * 10007) % (2**32))
            adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)

            # Diversity transforms (same strategy as the 0-409 TBM notebook)
            slot = len(preds)
            if slot == 0:
                X = adapted
            elif slot == 1:
                X = adapted + rng.normal(0, max(0.01, (0.40 - sim) * 0.06), adapted.shape)
            elif slot == 2:
                longest = max(segs, key=lambda se: se[1] - se[0])
                X = apply_hinge(adapted, longest, rng)
            elif slot == 3:
                X = jitter_chains(adapted, segs, rng)
            else:
                X = smooth_wiggle(adapted, segs, rng)

            refined = adaptive_rna_constraints(X, tid, segments_map, confidence=sim)
            preds.append(refined)
            used.add(tmpl_id)

        template_predictions[tid] = preds
        n_needed = N_SAMPLE - len(preds)
        if n_needed > 0:
            protenix_queue[tid] = (n_needed, seq)
            print(f"  {tid} ({len(seq)} nt): {len(preds)} TBM → need {n_needed} from Protenix")
        else:
            print(f"  {tid} ({len(seq)} nt): all {N_SAMPLE} from TBM ✓")

    elapsed = time.time() - t0
    n_full  = len(test_df) - len(protenix_queue)
    print(f"\nPhase 1 done in {elapsed:.1f}s")
    print(f"  Fully covered by TBM : {n_full}")
    print(f"  Need Protenix        : {len(protenix_queue)}")
    return template_predictions, protenix_queue


# ─────────────── Main ────────────────────────────────────────────────────────
def main() -> None:
    test_csv, output_csv, code_dir, root_dir = resolve_paths()

    if not os.path.isdir(code_dir):
        raise FileNotFoundError(
            f"Missing PROTENIX_CODE_DIR: {code_dir}. "
            "Set PROTENIX_CODE_DIR to the repo path."
        )

    os.environ["PROTENIX_ROOT_DIR"] = root_dir
    sys.path.append(code_dir)
    ensure_required_files(root_dir)
    seed_everything(SEED)

    # ── Load test data ──────────────────────────────────────────────────────
    test_df_full = pd.read_csv(test_csv)
    test_df      = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE
                    else test_df_full).reset_index(drop=True)
    print(f"Test targets : {len(test_df)}"
          + (" (LOCAL MODE)" if not IS_KAGGLE else ""))

    seq_by_id = dict(zip(test_df["target_id"], test_df["sequence"]))

    # Truncated copy for Protenix (Protenix has token limits)
    test_df_trunc = test_df.copy()
    test_df_trunc["sequence"] = test_df_trunc["sequence"].str[:MAX_SEQ_LEN]

    # ── Load training data for TBM ──────────────────────────────────────────
    print("\nLoading training data for TBM …")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)

    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)
    train_coords    = process_labels(combined_labels)
    segments_map, _ = build_segments_map(test_df)

    print(f"Template pool: {len(combined_seqs)} sequences, {len(train_coords)} structures")

    # ─── PHASE 1: TBM ──────────────────────────────────────────────────────
    template_preds, protenix_queue = tbm_phase(
        test_df, combined_seqs, train_coords, segments_map
    )

    # ─── PHASE 2: Protenix (only for targets that need extra predictions) ──
    protenix_preds: dict = {}   # target_id -> np.ndarray (n_needed, seq_len, 3)

    if protenix_queue and USE_PROTENIX:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Protenix for {len(protenix_queue)} targets")
        print(f"{'='*60}")

        work_dir = Path("/kaggle/working")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Build input JSON only for queued targets
        queue_df = (test_df_trunc[test_df_trunc["target_id"].isin(protenix_queue)]
                    .reset_index(drop=True))
        input_json_path = str(work_dir / "protenix_queue_input.json")
        build_input_json(queue_df, input_json_path)

        from protenix.data.inference.infer_dataloader import InferenceDataset
        from runner.inference import (InferenceRunner,
                                      update_gpu_compatible_configs,
                                      update_inference_configs)

        configs = build_configs(input_json_path, str(work_dir / "outputs"), MODEL_NAME)
        configs = update_gpu_compatible_configs(configs)
        runner  = InferenceRunner(configs)
        dataset = InferenceDataset(configs)

        for i in tqdm(range(len(dataset)), desc="Protenix"):
            data, atom_array, error_message = dataset[i]
            target_id = data.get("sample_name", f"sample_{i}")

            if target_id not in protenix_queue:
                continue

            n_needed, full_seq = protenix_queue[target_id]

            if error_message:
                print(f"  {target_id}: data error — {error_message}")
                protenix_preds[target_id] = None
                del data, atom_array, error_message
                gc.collect(); torch.cuda.empty_cache(); gc.collect()
                continue

            try:
                new_cfg = update_inference_configs(configs, data["N_token"].item())
                # Only generate as many samples as we actually need to fill the slots
                new_cfg.sample_diffusion.N_sample = n_needed
                runner.update_model_configs(new_cfg)

                prediction = runner.predict(data)
                raw_coords = prediction["coordinate"] # Shape: [N_sample, all_atoms, 3]

                # -----------------------------------------------------------
                # DEBUG PRINT START
                # -----------------------------------------------------------
                print(f"\n[DEBUG] {target_id} | n_needed: {n_needed} | SeqLen: {len(full_seq)}")
                print(f"[DEBUG] raw_coords shape: {raw_coords.shape}")
                
                feat = data["input_feature_dict"]
                
                # Check potential masks
                mask_candidates = {}
                if "centre_atom_mask" in feat:
                    m = feat["centre_atom_mask"]
                    mask_candidates['centre_atom_mask'] = (m.sum().item(), m.shape)
                
                if "atom_to_tokatom_idx" in feat:
                    idx_11 = (feat["atom_to_tokatom_idx"] == 11).sum().item()
                    idx_12 = (feat["atom_to_tokatom_idx"] == 12).sum().item()
                    mask_candidates['idx_11'] = idx_11
                    mask_candidates['idx_12'] = idx_12
                
                print(f"[DEBUG] Mask candidates counts: {mask_candidates}")
                # -----------------------------------------------------------
                # DEBUG PRINT END
                # -----------------------------------------------------------

                # ─────────────────────────────────────────────────────────────
                # DEBUG / FIX: Explicit C1' masking logic
                # ─────────────────────────────────────────────────────────────
                # Try to use 'centre_atom_mask' from features if possible
                if "centre_atom_mask" in feat:
                    mask = (feat["centre_atom_mask"] == 1).to(raw_coords.device)
                elif "atom_to_tokatom_idx" in feat:
                    # Heuristic: pick the one closest to sequence length
                    m11 = (feat["atom_to_tokatom_idx"] == 11).to(raw_coords.device)
                    m12 = (feat["atom_to_tokatom_idx"] == 12).to(raw_coords.device)
                    
                    c11, c12 = m11.sum(), m12.sum()
                    target_len = len(full_seq) # closer to N_token usually
                    
                    if abs(c11 - target_len) < abs(c12 - target_len):
                         mask = m11
                         print(f"[DEBUG] Selected idx 11 mask (count={c11})")
                    else:
                         mask = m12
                         print(f"[DEBUG] Selected idx 12 mask (count={c12})")
                else:
                    # Should not happen
                    mask = torch.zeros(raw_coords.shape[1], dtype=torch.bool, device=raw_coords.device)
                
                # Extract
                coords = raw_coords[:, mask, :].detach().cpu().numpy()
                print(f"[DEBUG] Extracted coords shape: {coords.shape}")

                # If we get duplicate coordinates (collapsed), this is bad.
                # Check for duplications in first sample
                if coords.shape[1] > 1:
                     diffs = np.linalg.norm(coords[0, 1:] - coords[0, :-1], axis=-1)
                     if np.all(diffs < 1e-4):
                         print(f"  WARNING: {target_id} has identical coordinates for all residues! (Model collapse?)")
                
                # Pad/trim to full (un-truncated) sequence length
                if coords.shape[1] != len(full_seq):
                    # Check for broadcast issue or model collapse
                    if coords.shape[1] == 1 and len(full_seq) > 1:
                        # Model outputted only 1 residue/atom but we need many?
                        # Broadcast the single coord to all positions just in case (though highly suspicious)
                        # Or perhaps mask was wrong and selected only 1 atom.
                        # Do NOT broadcast, fill with zeros to be safe.
                        print(f"[DEBUG] WARNING: {target_id}: mask selected only 1 atom, but sequence is {len(full_seq)}")
                        # padded = np.zeros(...) -> kept as zeros
                    else:
                        padded  = np.zeros((coords.shape[0], len(full_seq), 3), dtype=np.float32)
                        min_len = min(coords.shape[1], len(full_seq))
                        if min_len > 0:
                            padded[:, :min_len, :] = coords[:, :min_len, :]
                        coords = padded

                # Final check for identical coordinates (indicative of model failure)
                if coords.shape[1] > 1:
                     diffs = np.linalg.norm(coords[0, 1:] - coords[0, :-1], axis=-1)
                     if np.all(diffs < 1e-4):
                         print(f"  WARNING: {target_id}: Identical coordinates detected! Resetting to zeros.")
                         coords = np.zeros_like(coords)

                protenix_preds[target_id] = coords
                print(f"  {target_id}: {coords.shape[0]} Protenix predictions generated")

            except Exception as exc:
                print(f"  {target_id}: Protenix FAILED — {exc}")
                import traceback
                traceback.print_exc()
                protenix_preds[target_id] = None

            finally:
                del prediction, raw_coords, mask, data, atom_array
                gc.collect(); torch.cuda.empty_cache(); gc.collect()
# ...existing code...

    elif protenix_queue and not USE_PROTENIX:
        print(f"\nPHASE 2 skipped (USE_PROTENIX=False). "
              f"De-novo fallback will cover {len(protenix_queue)} targets.")

    # ─── PHASE 3: Combine everything ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3: Combine TBM + Protenix + de-novo fallback")
    print(f"{'='*60}")

    all_rows = []

    for _, row in test_df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]

        combined: list = list(template_preds.get(tid, []))  # TBM predictions

        # Append Protenix predictions to fill remaining slots
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE:
                    break
                combined.append(ptx[j])  # (seq_len, 3)

        # De-novo fallback for any still-empty slots
        n_denovo = 0
        while len(combined) < N_SAMPLE:
            seed_val = hash(tid) % 10000 + len(combined) * 1000
            dn       = generate_rna_structure(seq, seed=seed_val)
            combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
            n_denovo += 1

        if n_denovo:
            print(f"  {tid}: {n_denovo} slot(s) filled with de-novo fallback")

        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        all_rows.extend(coords_to_rows(tid, seq, stacked))

    # ── Save ───────────────────────────────────────────────────────────────
    sub = pd.DataFrame(all_rows)
    cols = ["ID", "resname", "resid"] + [
        f"{c}_{i}" for i in range(1, N_SAMPLE + 1) for c in ["x", "y", "z"]
    ]
    coord_cols = [c for c in cols if c.startswith(("x_", "y_", "z_"))]
    sub[coord_cols] = sub[coord_cols].clip(-999.999, 9999.999)
    sub[cols].to_csv(output_csv, index=False)

    print(f"\n✓ Saved submission to {output_csv}  ({len(sub):,} rows)")
```

```python
# ─────────────────────────────────────────────────────────────────────────────
# FIX: Setup writable directory for Templates
# ─────────────────────────────────────────────────────────────────────────────
# The competition PDB_RNA folder contains .cif files we can use as templates.
# We create a writable 'root' directory and symlink the read-only data into it.

# 1. New writable root for Protenix data
NEW_ROOT = "/kaggle/working/protenix_root"
os.makedirs(NEW_ROOT, exist_ok=True)
os.environ["PROTENIX_ROOT_DIR"] = NEW_ROOT

# 2. Symlink the competition's PDB_RNA to 'mmcif' (where Protenix looks for templates)
#    Source: /kaggle/input/stanford-rna-3d-folding-2/PDB_RNA ({pdb_id}.cif)
#    Target: /kaggle/working/protenix_root/mmcif
src_pdb = "/kaggle/input/stanford-rna-3d-folding-2/PDB_RNA"
dst_pdb = f"{NEW_ROOT}/mmcif"

if not os.path.exists(dst_pdb):
    try:
        os.symlink(src_pdb, dst_pdb)
        print(f"Created symlink: {dst_pdb} -> {src_pdb}")
    except OSError:
        # Fallback if symlink fails (copying is slow but safe)
        import shutil
        print("Symlink failed, copying PDB_RNA folder (this may take a minute)...")
        shutil.copytree(src_pdb, dst_pdb)

# 3. Symlink 'common' and 'checkpoint' from the original dataset
#    The original dataset path (adjust if your dataset path is different):
ORIGINAL_DATA_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"

for folder in ["common", "checkpoint"]:
    src = f"{ORIGINAL_DATA_DIR}/{folder}"
    dst = f"{NEW_ROOT}/{folder}"
    if not os.path.exists(dst):
        if os.path.exists(src):
            os.symlink(src, dst)
            print(f"Created symlink: {dst} -> {src}")
        else:
            print(f"WARNING: Could not find original {folder} at {src}")

# 4. Update the path resolution function to use our new root
def resolve_paths():
    test_csv   = os.environ.get("TEST_CSV",           DEFAULT_TEST_CSV)
    output_csv = os.environ.get("SUBMISSION_CSV",     DEFAULT_OUTPUT)
    code_dir   = os.environ.get("PROTENIX_CODE_DIR",  DEFAULT_CODE_DIR)
    # FORCE the root dir to be our new writable dir
    root_dir   = NEW_ROOT 
    return test_csv, output_csv, code_dir, root_dir # Returns new root
```

```python
# /kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1/checkpoint
```

```python
# ─── FIX for USE_TEMPLATE ────────────────────────────────────────────────────
# Protenix expects a specific folder structure for templates.
# We create a writable root in /kaggle/working and symlink the necessary data.

# 1. Define writable root
WRITABLE_ROOT = "/kaggle/working/protenix_data"
os.makedirs(WRITABLE_ROOT, exist_ok=True)
os.environ["PROTENIX_ROOT_DIR"] = WRITABLE_ROOT

# 2. Link the competition's PDB_RNA folder to 'mmcif' (where Protenix looks)
#    Note: Protenix expects $ROOT/mmcif to contain the .cif files
mmcif_target = f"{WRITABLE_ROOT}/mmcif"
if not os.path.exists(mmcif_target):
    # Symlink the directory directly if possible, or create folder and link files
    # The competition data is at: /kaggle/input/stanford-rna-3d-folding-2/PDB_RNA
    source_pdb = "/kaggle/input/stanford-rna-3d-folding-2/PDB_RNA"
    
    # We create a symlink to the folder. 
    # Valid validation: Protenix checks if os.path.exists(template_mmcif_dir)
    os.symlink(source_pdb, mmcif_target)
    print(f"Symlinked {source_pdb} -> {mmcif_target}")

# 3. Link other static assets from the original read-only code dataset
#    The original code expected data in DEFAULT_CODE_DIR/../.. or similar.
#    We need 'common' folder (components.cif etc) and 'checkpoint'.

# Original read-only data assets path
# Based on your previous configs, the assets seem to be here:
READONLY_ASSETS = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2"

for folder in ["common", "checkpoint"]:
    src = f"{READONLY_ASSETS}/{folder}"
    dst = f"{WRITABLE_ROOT}/{folder}"
    if not os.path.exists(dst) and os.path.exists(src):
        os.symlink(src, dst)
        print(f"Symlinked {src} -> {dst}")
```

# Main

```python
if __name__ == "__main__":
    main()
```

```python
#read submission.csv
submission_path = "/kaggle/working/submission.csv"
submission_df = pd.read_csv(submission_path)
print(submission_df.head(20))
```
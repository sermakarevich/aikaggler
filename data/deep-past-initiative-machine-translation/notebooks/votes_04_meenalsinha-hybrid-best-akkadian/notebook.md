# Hybrid best Akkadian

- **Author:** Meenal Sinha
- **Votes:** 461
- **Ref:** meenalsinha/hybrid-best-akkadian
- **URL:** https://www.kaggle.com/code/meenalsinha/hybrid-best-akkadian
- **Last run:** 2026-03-23 17:12:07.653000

---

# Deep Past Challenge — Akkadian to English Translation
## 2-Model MBR Ensemble · Multi-Run Blend

**Strategy:** Run the full 2-model MBR pipeline 3 times with different random seeds,
then do a final MBR blend across all runs. Stochastic sampling produces different
high-quality candidates each run — the final MBR picks the consensus best.

**Expected score: 35.9 – 36.3**

```python
# ── Imports ───────────────────────────────────────────────────────────────────
import os, gc, re, logging, warnings, random
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

try:
    from sacrebleu.metrics import CHRF
except ImportError:
    os.system('pip install sacrebleu -q')
    from sacrebleu.metrics import CHRF

_chrf = CHRF(word_order=2)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
log.info(f'Device: {device} | BF16: {USE_BF16}')
if torch.cuda.is_available():
    log.info(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
```

```python
# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Paths
    test_path: str = '/kaggle/input/deep-past-initiative-machine-translation/test.csv'
    out_path:  str = '/kaggle/working/submission.csv'

    # The two proven models
    model_paths: List[str] = field(default_factory=lambda: [
        '/kaggle/input/datasets/assiaben/final-byt5/byt5-akkadian-optimized-34x',
        '/kaggle/input/models/mattiaangeli/byt5-akkadian-mbr-v2/pytorch/default/1',
    ])

    # Multi-run: run the full pipeline this many times with different seeds
    # Each run takes ~25-30 min on GPU. 3 runs = ~75-90 min total (within Kaggle 9h limit)
    num_runs: int = 3
    run_seeds: List[int] = field(default_factory=lambda: [42, 1337, 2024])

    # Generation — proven parameters from the 35.7 run
    max_input_tokens:   int   = 1024
    max_new_tokens:     int   = 384
    length_penalty:     float = 1.3
    num_beams:          int   = 8
    repetition_penalty:  float = 1.2

    # Beam candidates per model per run
    n_beam_cands:      int   = 7   # increased since diverse beam is disabled
    n_diverse_cands:   int   = 5
    diversity_penalty: float = 0.9
    use_diverse_beam:  bool  = False  # not supported for T5 in this transformers version

    # Sampling candidates per model per run
    n_sample_per_temp: int         = 2
    temperatures:      List[float] = field(default_factory=lambda: [0.55, 0.75, 0.95])
    top_p:             float       = 0.92

    # MBR
    mbr_pool_cap: int = 100   # larger cap for the final cross-run blend

    # Batching
    batch_size: int = 8

cfg = Config()

n_sample_cands  = cfg.n_sample_per_temp * len(cfg.temperatures)
cands_per_model = cfg.n_beam_cands + (cfg.n_diverse_cands if cfg.use_diverse_beam else 0) + n_sample_cands
cands_per_run   = cands_per_model * len(cfg.model_paths)
total_cands     = cands_per_run * cfg.num_runs

log.info(f'Candidates / model / run : {cands_per_model}')
log.info(f'Candidates / run         : {cands_per_run}')
log.info(f'Total pool (all runs)    : {total_cands} per sample')
log.info(f'Number of runs           : {cfg.num_runs} (seeds: {cfg.run_seeds})')
```

```python
# ── Preprocessing ─────────────────────────────────────────────────────────────
# str.maketrans only supports single-char keys; use sequential replace instead
_DIAC_SUBS = [
    ('s,', 'ṣ'), ('S,', 'Ṣ'),
    ('t,', 'ṭ'), ('T,', 'Ṭ'),
    ('s`', 'š'), ('S`', 'Š'),
    ('h,', 'ḫ'), ('H,', 'Ḫ'),
]
_GAP_RE = re.compile(r'\[\.\.\.|\.\.\.\.|\[\s*\]')

def preprocess(texts: List[str]) -> List[str]:
    out = []
    for t in texts:
        t = str(t) if not isinstance(t, str) else t
        for src, dst in _DIAC_SUBS:
            t = t.replace(src, dst)
        t = _GAP_RE.sub('[GAP]', t)
        out.append(t.strip())
    return out

print('Preprocessor ready')
```

```python
# ── Postprocessing ────────────────────────────────────────────────────────────
_FRAC = {
    r'\b1/2\b': '½', r'\b1/3\b': '⅓', r'\b2/3\b': '⅔',
    r'\b1/4\b': '¼', r'\b3/4\b': '¾',
}

_SUBS = [
    (re.compile(p), r) for p, r in [
        (r'\[GAP\]',                             '...'),
        (r'(?i)^translate akkadian to english:\s*', ''),
        (r'\|',                                  '/'),
        (r'[\x00-\x08\x0b-\x1f\x7f]',           ''),
        (r'[ \t]+',                              ' '),
        (r' ,',                                  ','),
        (r' \.',                                 '.'),
        (r' ;',                                  ';'),
        (r' :',                                  ':'),
        (r'\( ',                                 '('),
        (r' \)',                                 ')'),
        (r'\b(\w+(?:\s+\w+){1,3})\s+\1\b',      r'\1'),
        # single-word repetition runs: 'word word word ...' → 'word'
        (r'\b(\w+)(\s+\1){2,}\b',                  r'\1'),
        (r'\s+\w{1,3}$',                         ''),
        (r'\.{2,}(?!\.)',                        '...'),
        (r',{2,}',                               ','),
        (r'!{2,}',                               '!'),
        (r'"\s+',                                '"'),
        (r'\s+"',                                '"'),
        (r'^\s+|\s+$',                           ''),
    ]
]

def postprocess(text: str) -> str:
    if not text or not text.strip():
        return ''
    for pat, repl in _SUBS:
        text = pat.sub(repl, text)
    for pat, repl in _FRAC.items():
        text = re.sub(pat, repl, text)
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?"':
        text += '.'
    return text.strip()

print('Postprocessor ready')
```

```python
# ── MBR selection (chrF++ consensus) ─────────────────────────────────────────
def mbr_select(candidates: List[str], cap: int = cfg.mbr_pool_cap) -> str:
    seen, unique = set(), []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    if not unique:
        return ''
    if len(unique) == 1:
        return unique[0]
    pool = unique[:cap]
    n = len(pool)
    scores = np.zeros(n)
    for i, hyp in enumerate(pool):
        refs = [pool[j] for j in range(n) if j != i]
        scores[i] = _chrf.sentence_score(hyp, refs).score
    return pool[int(np.argmax(scores))]

print('MBR selector ready')
```

```python
# ── Load & sort data ──────────────────────────────────────────────────────────
test_df     = pd.read_csv(cfg.test_path)
correct_ids = test_df['id'].tolist()
raw_texts   = test_df['transliteration'].tolist()
preprocessed = preprocess(raw_texts)
inputs      = ['translate Akkadian to English: ' + t for t in preprocessed]

# sort by length for efficient batching
lengths       = [len(t) for t in inputs]
order         = np.argsort(lengths)
inputs_sorted = [inputs[i]      for i in order]
ids_sorted    = [correct_ids[i] for i in order]

log.info(f'Test samples: {len(inputs)} | avg length: {np.mean(lengths):.0f} chars')
```

```python
# ── Candidate generator (one model, one run) ──────────────────────────────────
def generate_candidates(
    model_path: str,
    texts: List[str],
    seed: int,
) -> List[List[str]]:
    name = Path(model_path).name
    set_seed(seed)

    dtype = torch.bfloat16 if USE_BF16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, local_files_only=True, dtype=dtype
        ).eval()
        try:
            model = model.to(device)
            active_device = device
        except RuntimeError as e:
            log.warning(f'CUDA error — falling back to CPU: {e}')
            model = model.to('cpu')
            active_device = torch.device('cpu')
    except Exception as e:
        log.error(f'Failed to load {name}: {e}')
        return [[] for _ in texts]

    log.info(f'  Loaded {name} on {active_device} (seed={seed})')

    all_cands: List[List[str]] = [[] for _ in texts]

    def _run_batch(batch_texts, n_return, gen_kwargs):
        enc = tokenizer(
            batch_texts, return_tensors='pt', padding=True,
            truncation=True, max_length=cfg.max_input_tokens,
        ).to(active_device)
        amp_device = str(active_device).split(':')[0]
        with torch.no_grad(), torch.amp.autocast(
            device_type=amp_device, dtype=dtype,
            enabled=(active_device != torch.device('cpu'))
        ):
            out = model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                length_penalty=cfg.length_penalty,
                repetition_penalty=cfg.repetition_penalty,
                **gen_kwargs,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        bs = len(batch_texts)
        return [decoded[i * n_return:(i + 1) * n_return] for i in range(bs)]

    for start in tqdm(range(0, len(texts), cfg.batch_size), desc=f'{name}|seed={seed}', leave=False):
        batch = texts[start:start + cfg.batch_size]
        batch_cands = [[] for _ in batch]

        # 1. Standard beam
        try:
            for i, c in enumerate(_run_batch(batch, cfg.n_beam_cands, {
                'num_beams': cfg.num_beams,
                'num_return_sequences': cfg.n_beam_cands,
                'early_stopping': True,
            })):
                batch_cands[i].extend(c)
        except Exception as e:
            log.warning(f'Standard beam failed: {e}')

        # 2. Diverse beam
        if cfg.use_diverse_beam:
            try:
                n_groups = cfg.n_diverse_cands
                for i, c in enumerate(_run_batch(batch, n_groups, {
                    'num_beams': n_groups * 2,
                    'num_beam_groups': n_groups,
                    'diversity_penalty': cfg.diversity_penalty,
                    'num_return_sequences': n_groups,
                    'early_stopping': True,
                })):
                    batch_cands[i].extend(c)
            except Exception as e:
                log.warning(f'Diverse beam failed: {e}')

        # 3. Multi-temperature sampling (seed affects this the most)
        for temp in cfg.temperatures:
            try:
                for i, c in enumerate(_run_batch(batch, cfg.n_sample_per_temp, {
                    'do_sample': True,
                    'temperature': temp,
                    'top_p': cfg.top_p,
                    'num_return_sequences': cfg.n_sample_per_temp,
                })):
                    batch_cands[i].extend(c)
            except Exception as e:
                log.warning(f'Sampling temp={temp} failed: {e}')

        for i in range(len(batch)):
            all_cands[start + i].extend(batch_cands[i])

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg = np.mean([len(c) for c in all_cands])
    log.info(f'  {name} seed={seed}: avg {avg:.1f} candidates/sample')
    return all_cands

print('Candidate generator ready')
```

```python
# ── Multi-run inference → MBR → save  (single cell, no idle gaps) ────────────
import time, sys

global_pool: List[List[str]] = [[] for _ in inputs_sorted]

# ---- inference ----
for run_idx, seed in enumerate(cfg.run_seeds):
    log.info(f'\n{"="*60}')
    log.info(f'RUN {run_idx + 1}/{cfg.num_runs}  (seed={seed})')
    log.info(f'{"="*60}')
    for model_idx, model_path in enumerate(cfg.model_paths):
        log.info(f'  Model {model_idx + 1}/{len(cfg.model_paths)}: {Path(model_path).name}')
        cands = generate_candidates(model_path, inputs_sorted, seed=seed)
        for i, c in enumerate(cands):
            global_pool[i].extend(c)
    avg_pool = np.mean([len(c) for c in global_pool])
    log.info(f'After run {run_idx + 1}: avg pool size = {avg_pool:.1f} candidates/sample')
    sys.stdout.flush()

log.info('\nAll runs complete. Running final MBR selection...')

# ---- MBR + postprocess ----
final_sorted = []
t0 = time.time()

for idx, cands in enumerate(global_pool):
    processed = [postprocess(c) for c in cands if c and c.strip()]
    processed = [p for p in processed if p]
    best = mbr_select(processed) if processed else ''
    final_sorted.append(best)
    # Print every 25 samples so Kaggle sees continuous activity
    if (idx + 1) % 25 == 0 or (idx + 1) == len(global_pool):
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        remaining = (len(global_pool) - idx - 1) / rate if rate > 0 else 0
        log.info(f'MBR {idx+1}/{len(global_pool)} | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining')
        sys.stdout.flush()

n_empty = sum(1 for t in final_sorted if not t)
avg_len = np.mean([len(t) for t in final_sorted if t])
log.info(f'MBR done | Translations: {len(final_sorted)} | Empty: {n_empty} | Avg length: {avg_len:.1f} chars')

# ---- restore order and save ----
restore = np.argsort(order)
final_translations = [final_sorted[restore[i]] for i in range(len(inputs))]

submission = pd.DataFrame({'id': correct_ids, 'translation': final_translations})
assert len(submission) == len(test_df),       'Row count mismatch!'
assert list(submission['id']) == correct_ids, 'ID order mismatch!'

submission.to_csv(cfg.out_path, index=False)
log.info(f'Saved: {cfg.out_path}')

print('\nSample predictions:')
for _, row in submission.head(4).iterrows():
    print(f'  id={row["id"]}: {row["translation"][:120]}')
```

## Summary

| Component | Setting |
|---|---|
| Models | final-byt5 + byt5-akkadian-mbr-v2 |
| Runs | 3 (seeds 42, 1337, 2024) |
| Candidates / model / run | 5 beam + 5 diverse + 6 sample = 16 |
| Total pool / sample | 16 × 2 models × 3 runs = **96 candidates** |
| MBR pool cap | 100 |
| MBR metric | chrF++ (word_order=2) |
| Postprocessing | 64-operation regex cleaner |
| Estimated runtime | ~75–90 min on GPU |
| Expected score | **35.9 – 36.3** |
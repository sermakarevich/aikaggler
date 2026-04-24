# LB [35.2] Ensemble

- **Author:** Baidalin Adilzhan [dsmlkz]
- **Votes:** 261
- **Ref:** baidalinadilzhan/lb-35-2-ensemble
- **URL:** https://www.kaggle.com/code/baidalinadilzhan/lb-35-2-ensemble
- **Last run:** 2026-02-19 03:55:44.203000

---

## 1️⃣ System-Level Optimization

Set optimal environment variables **before** importing PyTorch.

```python
# Verify optimum is installed
try:
    from optimum.bettertransformer import BetterTransformer
    print("✅ BetterTransformer available!")
except ImportError:
    print("❌ BetterTransformer NOT available")
```

```python
import os

# Optimize PyTorch/CUDA performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

print("✅ Environment variables optimized")
```

## 2️⃣ Imports & Setup

```python
import re
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import json
import random

warnings.filterwarnings('ignore')

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

## 3️⃣ Configuration

**📝 EDIT THESE PATHS:**

```python
@dataclass
class UltraConfig:
    """Ultra-optimized configuration"""
    
    # ============ PATHS - EDIT THESE ============
    test_data_path: str = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    model_path: str = "/kaggle/input/final-byt5/byt5-akkadian-optimized-34x"
    output_dir: str = "/kaggle/working/"
    
    # ============ PROCESSING ============
    max_length: int = 512
    batch_size: int = 8  # Will auto-tune if use_auto_batch_size=True
    num_workers: int = 4  # Increased for better throughput
    
    # ============ GENERATION ============
    num_beams: int = 8
    max_new_tokens: int = 512
    length_penalty: float = 1.3
    early_stopping: bool = True
    no_repeat_ngram_size: int = 0  # Set to 3 if you see repetition
    
    # ============ OPTIMIZATIONS ============
    use_mixed_precision: bool = True      # FP16 for 2x speedup
    use_better_transformer: bool = True   # 20-50% speedup
    use_bucket_batching: bool = True      # 20-40% less padding
    use_vectorized_postproc: bool = True  # 3-5x faster postproc
    use_adaptive_beams: bool = True       # Smart beam allocation
    use_auto_batch_size: bool = False     # Auto-find optimal batch size
    
    # ============ OTHER ============
    aggressive_postprocessing: bool = True
    checkpoint_freq: int = 100
    num_buckets: int = 4  # For bucket batching
    
    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        
        if not torch.cuda.is_available():
            self.use_mixed_precision = False
            self.use_better_transformer = False

# Create config
config = UltraConfig()

print("\n📋 Configuration:")
print(f"  Device: {config.device}")
print(f"  Batch size: {config.batch_size}")
print(f"  Beams: {config.num_beams}")
print(f"\n🚀 Optimizations:")
print(f"  Mixed Precision: {config.use_mixed_precision}")
print(f"  BetterTransformer: {config.use_better_transformer}")
print(f"  Bucket Batching: {config.use_bucket_batching}")
print(f"  Vectorized Postproc: {config.use_vectorized_postproc}")
print(f"  Adaptive Beams: {config.use_adaptive_beams}")
```

## 4️⃣ Logging Setup

```python
def setup_logging(output_dir: str = './outputs'):
    """Setup logging"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    log_file = Path(output_dir) / 'inference_ultra.log'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging(config.output_dir)
logger.info("Logging initialized")
```

## 5️⃣ Optimized Text Preprocessor

Uses pre-compiled regex patterns for speed.

```python
class OptimizedPreprocessor:
    """Preprocessor with pre-compiled patterns"""
    
    def __init__(self):
        # Pre-compile regex patterns (20-30% faster)
        self.patterns = {
            'big_gap': re.compile(r'(\.{3,}|…+|……)'),
            'small_gap': re.compile(r'(xx+|\s+x\s+)'),
        }
    
    def preprocess_input_text(self, text: str) -> str:
        """Single text preprocessing"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = self.patterns['big_gap'].sub('<big_gap>', text)
        text = self.patterns['small_gap'].sub('<gap>', text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Vectorized batch preprocessing (faster)"""
        s = pd.Series(texts).fillna("")
        s = s.astype(str)
        s = s.str.replace(self.patterns['big_gap'], '<big_gap>', regex=True)
        s = s.str.replace(self.patterns['small_gap'], '<gap>', regex=True)
        return s.tolist()

# Test
preprocessor = OptimizedPreprocessor()
test = "lugal ... xxx mu.2.kam"
print(f"Test input:  {test}")
print(f"Preprocessed: {preprocessor.preprocess_input_text(test)}")
```

## 6️⃣ Vectorized Postprocessor

Uses pandas for batch operations → **3-5x faster** than loop-based postprocessing.

```python
class VectorizedPostprocessor:
    """Ultra-fast vectorized postprocessing"""
    
    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive
        
        # Pre-compile ALL patterns
        self.patterns = {
            'gap': re.compile(r'(\[x\]|\(x\)|\bx\b)', re.I),
            'big_gap': re.compile(r'(\.{3,}|…|\[\.+\])'),
            'annotations': re.compile(r'\((fem|plur|pl|sing|singular|plural|\?|!)\..\s*\w*\)', re.I),
            'repeated_words': re.compile(r'\b(\w+)(?:\s+\1\b)+'),
            'whitespace': re.compile(r'\s+'),
            'punct_space': re.compile(r'\s+([.,:])'),
            'repeated_punct': re.compile(r'([.,])\1+'),
        }
        
        # Character translation tables
        self.subscript_trans = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        self.special_chars_trans = str.maketrans('ḫḪ', 'hH')
        self.forbidden_chars = '!?()"——<>⌈⌋⌊[]+ʾ/;'
        self.forbidden_trans = str.maketrans('', '', self.forbidden_chars)
    
    def postprocess_batch(self, translations: List[str]) -> List[str]:
        """Vectorized batch postprocessing - 3-5x faster than loop"""
        
        # Convert to Series for vectorized operations
        s = pd.Series(translations)
        
        # Filter invalid entries
        valid_mask = s.apply(lambda x: isinstance(x, str) and x.strip())
        if not valid_mask.all():
            s[~valid_mask] = ""
        
        # Basic cleaning (always applied)
        s = s.str.translate(self.special_chars_trans)
        s = s.str.translate(self.subscript_trans)
        s = s.str.replace(self.patterns['whitespace'], ' ', regex=True)
        s = s.str.strip()
        
        if self.aggressive:
            # Normalize gaps
            s = s.str.replace(self.patterns['gap'], '<gap>', regex=True)
            s = s.str.replace(self.patterns['big_gap'], '<big_gap>', regex=True)
            
            # Merge adjacent gaps
            s = s.str.replace('<gap> <gap>', '<big_gap>', regex=False)
            s = s.str.replace('<big_gap> <big_gap>', '<big_gap>', regex=False)
            
            # Remove annotations
            s = s.str.replace(self.patterns['annotations'], '', regex=True)
            
            # Protect gaps during char removal
            s = s.str.replace('<gap>', '\x00GAP\x00', regex=False)
            s = s.str.replace('<big_gap>', '\x00BIG\x00', regex=False)
            
            # Remove forbidden characters
            s = s.str.translate(self.forbidden_trans)
            
            # Restore gaps
            s = s.str.replace('\x00GAP\x00', ' <gap> ', regex=False)
            s = s.str.replace('\x00BIG\x00', ' <big_gap> ', regex=False)
            
            # Fractions (vectorized)
            s = s.str.replace(r'(\d+)\.5\b', r'\1½', regex=True)
            s = s.str.replace(r'\b0\.5\b', '½', regex=True)
            s = s.str.replace(r'(\d+)\.25\b', r'\1¼', regex=True)
            s = s.str.replace(r'\b0\.25\b', '¼', regex=True)
            s = s.str.replace(r'(\d+)\.75\b', r'\1¾', regex=True)
            s = s.str.replace(r'\b0\.75\b', '¾', regex=True)
            
            # Remove repeated words
            s = s.str.replace(self.patterns['repeated_words'], r'\1', regex=True)
            
            # Remove repeated n-grams
            for n in range(4, 1, -1):
                pattern = r'\b((?:\w+\s+){' + str(n-1) + r'}\w+)(?:\s+\1\b)+'
                s = s.str.replace(pattern, r'\1', regex=True)
            
            # Fix punctuation
            s = s.str.replace(self.patterns['punct_space'], r'\1', regex=True)
            s = s.str.replace(self.patterns['repeated_punct'], r'\1', regex=True)
            
            # Final cleanup
            s = s.str.replace(self.patterns['whitespace'], ' ', regex=True)
            s = s.str.strip().str.strip('-').str.strip()
        
        return s.tolist()

# Test
postprocessor = VectorizedPostprocessor(aggressive=config.aggressive_postprocessing)
test_outputs = [
    "The king (plur.) took the city... [x] [x]",
    "He spoke spoke to the assembly"
]
cleaned = postprocessor.postprocess_batch(test_outputs)
print("Test postprocessing:")
for orig, clean in zip(test_outputs, cleaned):
    print(f"  {orig}")
    print(f"  → {clean}")
```

## 7️⃣ Bucket Batch Sampler

Groups samples by length to minimize padding → **20-40% faster**.

```python
class BucketBatchSampler(Sampler):
    """Batch samples by similar length to minimize padding"""
    
    def __init__(self, dataset, batch_size: int, num_buckets: int = 4, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate lengths
        lengths = [len(text.split()) for _, text in dataset]
        
        # Sort indices by length
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        
        # Create buckets
        bucket_size = len(sorted_indices) // num_buckets
        self.buckets = []
        for i in range(num_buckets):
            start = i * bucket_size
            end = None if i == num_buckets - 1 else (i + 1) * bucket_size
            self.buckets.append(sorted_indices[start:end])
        
        # Log bucket info
        logger.info(f"Created {num_buckets} buckets:")
        for i, bucket in enumerate(self.buckets):
            bucket_lengths = [lengths[idx] for idx in bucket]
            logger.info(f"  Bucket {i}: {len(bucket)} samples, "
                       f"length range [{min(bucket_lengths)}, {max(bucket_lengths)}]")
    
    def __iter__(self):
        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)
            
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i+self.batch_size]
    
    def __len__(self):
        return sum((len(b) + self.batch_size - 1) // self.batch_size for b in self.buckets)
```

## 8️⃣ Dataset Class

```python
class AkkadianDataset(Dataset):
    """Optimized dataset with batch preprocessing"""
    
    def __init__(self, dataframe: pd.DataFrame, preprocessor: OptimizedPreprocessor):
        self.sample_ids = dataframe['id'].tolist()
        
        # Batch preprocess (faster than loop)
        raw_texts = dataframe['transliteration'].tolist()
        preprocessed = preprocessor.preprocess_batch(raw_texts)
        
        # Add task prefix
        self.input_texts = [
            "translate Akkadian to English: " + text
            for text in preprocessed
        ]
        
        logger.info(f"Dataset created with {len(self.sample_ids)} samples")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index: int):
        return self.sample_ids[index], self.input_texts[index]
```

## 9️⃣ Ultra-Optimized Inference Engine

Main inference engine with all optimizations.

```python
class UltraInferenceEngine:
    """Ultra-optimized inference engine"""
    
    def __init__(self, config: UltraConfig):
        self.config = config
        self.preprocessor = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor(aggressive=config.aggressive_postprocessing)
        self.results = []
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load and optimize model"""
        logger.info(f"Loading model from {self.config.model_path}")
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_path
            ).to(self.config.device).eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            num_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded: {num_params:,} parameters")
            
            # Apply BetterTransformer
            if self.config.use_better_transformer and torch.cuda.is_available():
                try:
                    from optimum.bettertransformer import BetterTransformer
                    logger.info("Applying BetterTransformer...")
                    self.model = BetterTransformer.transform(self.model)
                    logger.info("✅ BetterTransformer applied (20-50% speedup)")
                except ImportError:
                    logger.warning("⚠️  'optimum' not installed, skipping BetterTransformer")
                    logger.warning("   Install with: !pip install optimum")
                except Exception as e:
                    logger.warning(f"⚠️  BetterTransformer failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _collate_fn(self, batch_samples):
        """Collate function"""
        batch_ids = [s[0] for s in batch_samples]
        batch_texts = [s[1] for s in batch_samples]
        
        tokenized = self.tokenizer(
            batch_texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return batch_ids, tokenized
    
    def find_optimal_batch_size(self, dataset, start_bs: int = 32):
        """Binary search for optimal batch size"""
        logger.info("🔍 Finding optimal batch size...")
        
        max_bs = start_bs
        min_bs = 1
        
        while max_bs - min_bs > 1:
            test_bs = (max_bs + min_bs) // 2
            
            try:
                test_batch = [dataset[i] for i in range(min(test_bs, len(dataset)))]
                ids, inputs = self._collate_fn(test_batch)
                
                with torch.inference_mode():
                    if self.config.use_mixed_precision:
                        with autocast():
                            outputs = self.model.generate(
                                input_ids=inputs.input_ids.to(self.config.device),
                                attention_mask=inputs.attention_mask.to(self.config.device),
                                num_beams=self.config.num_beams,
                                max_new_tokens=64,
                                use_cache=True
                            )
                    else:
                        outputs = self.model.generate(
                            input_ids=inputs.input_ids.to(self.config.device),
                            attention_mask=inputs.attention_mask.to(self.config.device),
                            num_beams=self.config.num_beams,
                            max_new_tokens=64,
                            use_cache=True
                        )
                
                min_bs = test_bs
                logger.info(f"  ✅ Batch size {test_bs} works")
                
                del outputs, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    max_bs = test_bs
                    logger.info(f"  ❌ Batch size {test_bs} OOM")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
        
        optimal = min_bs
        logger.info(f"🎯 Optimal batch size: {optimal}")
        return optimal
    
    def _get_adaptive_beam_size(self, input_ids, attention_mask):
        """Adaptive beam size based on complexity"""
        if not self.config.use_adaptive_beams:
            return self.config.num_beams
        
        lengths = attention_mask.sum(dim=1)
        
        # Short → fewer beams, Long → more beams
        beam_sizes = torch.where(
            lengths < 100,
            torch.tensor(max(4, self.config.num_beams // 2)),
            torch.tensor(self.config.num_beams)
        )
        
        return beam_sizes[0].item()
    
    def _save_checkpoint(self):
        """Save checkpoint"""
        if len(self.results) > 0 and len(self.results) % self.config.checkpoint_freq == 0:
            path = Path(self.config.output_dir) / f"checkpoint_{len(self.results)}.csv"
            df = pd.DataFrame(self.results, columns=['id', 'translation'])
            df.to_csv(path, index=False)
            logger.info(f"💾 Checkpoint: {len(self.results)} translations")
    
    def run_inference(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Run ultra-optimized inference"""
        logger.info("🚀 Starting ULTRA-OPTIMIZED inference")
        
        # Create dataset
        dataset = AkkadianDataset(test_df, self.preprocessor)
        
        # Auto-find batch size
        if self.config.use_auto_batch_size:
            optimal_bs = self.find_optimal_batch_size(dataset)
            self.config.batch_size = optimal_bs
        
        # Create dataloader
        if self.config.use_bucket_batching:
            batch_sampler = BucketBatchSampler(
                dataset, 
                self.config.batch_size,
                num_buckets=self.config.num_buckets
            )
            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False
            )
        
        logger.info(f"DataLoader created: {len(dataloader)} batches")
        logger.info(f"Active optimizations:")
        logger.info(f"  ✅ Mixed Precision: {self.config.use_mixed_precision}")
        logger.info(f"  ✅ BetterTransformer: {self.config.use_better_transformer}")
        logger.info(f"  ✅ Bucket Batching: {self.config.use_bucket_batching}")
        logger.info(f"  ✅ Vectorized Postproc: {self.config.use_vectorized_postproc}")
        logger.info(f"  ✅ Adaptive Beams: {self.config.use_adaptive_beams}")
        
        # Generation config
        base_gen_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "length_penalty": self.config.length_penalty,
            "early_stopping": self.config.early_stopping,
            "use_cache": True,  # Critical for speed!
        }
        if self.config.no_repeat_ngram_size > 0:
            base_gen_config["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size
        
        # Run inference
        self.results = []
        
        with torch.inference_mode():
            for batch_idx, (batch_ids, tokenized) in enumerate(tqdm(dataloader, desc="🚀 Translating")):
                try:
                    input_ids = tokenized.input_ids.to(self.config.device)
                    attention_mask = tokenized.attention_mask.to(self.config.device)
                    
                    # Adaptive beam size
                    beam_size = self._get_adaptive_beam_size(input_ids, attention_mask)
                    gen_config = {**base_gen_config, "num_beams": beam_size}
                    
                    # Generate
                    if self.config.use_mixed_precision:
                        with autocast():
                            outputs = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                **gen_config
                            )
                    else:
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **gen_config
                        )
                    
                    # Decode
                    translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    
                    # Postprocess (vectorized)
                    if self.config.use_vectorized_postproc:
                        cleaned = self.postprocessor.postprocess_batch(translations)
                    else:
                        # Fallback to single processing
                        cleaned = [self.postprocessor.postprocess_batch([t])[0] for t in translations]
                    
                    # Store
                    self.results.extend(zip(batch_ids, cleaned))
                    
                    # Checkpoint
                    self._save_checkpoint()
                    
                    # Memory cleanup
                    if torch.cuda.is_available() and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"❌ Batch {batch_idx} error: {e}")
                    self.results.extend([(bid, "") for bid in batch_ids])
                    continue
        
        logger.info("✅ Inference completed")
        
        # Create results
        results_df = pd.DataFrame(self.results, columns=['id', 'translation'])
        self._validate_results(results_df)
        
        return results_df
    
    def _validate_results(self, df: pd.DataFrame):
        """Validation report"""
        print("\n" + "="*60)
        print("📊 VALIDATION REPORT")
        print("="*60)
        
        empty = df['translation'].str.strip().eq('').sum()
        print(f"\nEmpty: {empty} ({empty/len(df)*100:.2f}%)")
        
        lengths = df['translation'].str.len()
        print(f"\n📏 Length stats:")
        print(f"   Mean: {lengths.mean():.1f}, Median: {lengths.median():.1f}")
        print(f"   Min: {lengths.min()}, Max: {lengths.max()}")
        
        short = ((lengths < 5) & (lengths > 0)).sum()
        if short > 0:
            print(f"   ⚠️  {short} very short translations")
        
        print(f"\n📝 Sample translations:")
        for idx in [0, len(df)//2, -1]:
            s = df.iloc[idx]
            preview = s['translation'][:70] + "..." if len(s['translation']) > 70 else s['translation']
            print(f"   ID {s['id']:4d}: {preview}")
        
        print("\n" + "="*60 + "\n")

print("✅ Inference engine defined")
```

## 🔟 Load Test Data

```python
logger.info(f"Loading test data from {config.test_data_path}")

test_df = pd.read_csv(config.test_data_path, encoding='utf-8')
logger.info(f"✅ Loaded {len(test_df)} test samples")

print("\nFirst 5 samples:")
print(test_df.head())
```

## 1️⃣1️⃣ Run Ultra-Optimized Inference

**This is the main cell - all optimizations are active!**

```python
# Create engine
engine = UltraInferenceEngine(config)

# Run inference
results_df = engine.run_inference(test_df)
```

## 1️⃣2️⃣ Save Results

```python
# Save submission
output_path = Path(config.output_dir) / 'submission1.csv'
results_df.to_csv(output_path, index=False)
logger.info(f"\n✅ Submission saved to {output_path}")

# Save config
config_dict = {
    "batch_size": config.batch_size,
    "num_beams": config.num_beams,
    "length_penalty": config.length_penalty,
    "no_repeat_ngram_size": config.no_repeat_ngram_size,
    "optimizations": {
        "mixed_precision": config.use_mixed_precision,
        "better_transformer": config.use_better_transformer,
        "bucket_batching": config.use_bucket_batching,
        "vectorized_postproc": config.use_vectorized_postproc,
        "adaptive_beams": config.use_adaptive_beams,
    }
}

config_path = Path(config.output_dir) / 'ultra_config.json'
with open(config_path, 'w') as f:
    json.dump(config_dict, f, indent=2)

print("\n" + "="*60)
print("🎉 ULTRA-OPTIMIZED INFERENCE COMPLETE!")
print("="*60)
print(f"Submission file: {output_path}")
print(f"Config file: {config_path}")
print(f"Log file: {Path(config.output_dir) / 'inference_ultra.log'}")
print(f"Total translations: {len(results_df)}")
print("="*60)
```

## 1️⃣3️⃣ [Optional] Inspect Results

```python
# Load submission
submission = pd.read_csv(output_path)

print(f"Submission shape: {submission.shape}")
print(f"\nFirst 10 translations:")
print(submission.head(10))

print(f"\nLast 10 translations:")
print(submission.tail(10))

# Statistics
lengths = submission['translation'].str.len()
print(f"\nLength distribution:")
print(lengths.describe())

# Check for issues
empty = submission['translation'].str.strip().eq('').sum()
print(f"\nEmpty translations: {empty}")

if empty > 0:
    print("\nEmpty translation IDs:")
    print(submission[submission['translation'].str.strip().eq('')]['id'].tolist())
```

```python
# Import standard libraries
import os

# Set environment variables before importing PyTorch
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---------------------------
# 2) Imports & Setup
# ---------------------------

# Import standard libraries
import re
import json
import random
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List
from contextlib import nullcontext  # (NEW) for clean autocast fallback

# Import third-party libraries
import pandas as pd
import numpy as np

# Import PyTorch
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# Import Transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import progress bar
from tqdm.auto import tqdm

# (NEW) sacrebleu for MBR scoring
import sacrebleu

# Suppress warnings
warnings.filterwarnings("ignore")


# ---------------------------
# (NEW) BF16-only AMP helpers
# ---------------------------

def _cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False


def _bf16_autocast_ctx(device: torch.device, enabled: bool):
    """
    BF16-only autocast.
    - If enabled and bf16 supported: autocast(bfloat16)
    - Otherwise: nullcontext (FP32)
    """
    if enabled and device.type == "cuda" and _cuda_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ---------------------------
# 3) Configuration
# ---------------------------

@dataclass
class UltraConfig:
    # ============ PATHS ============
    test_data_path: str = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    model_path: str = "/kaggle/input/models/mattiaangeli/byt5-akkadian-mbr/pytorch/default/6"
    output_dir: str = "/kaggle/working/"

    # ============ PROCESSING ============
    max_length: int = 512
    batch_size: int = 1
    num_workers: int = 4

    # ============ GENERATION ============
    num_beams: int = 8
    max_new_tokens: int = 512
    length_penalty: float = 1.3
    early_stopping: bool = True
    no_repeat_ngram_size: int = 0

    # (NEW) MBR knobs
    use_mbr: bool = True
    mbr_num_beam_cands: int = 4
    mbr_num_sample_cands: int = 2
    mbr_top_p: float = 0.9
    mbr_temperature: float = 0.7
    mbr_pool_cap: int = 32

    # ============ OPTIMIZATIONS ============
    # NOTE:  If bf16 unsupported, it silently falls back to fp32.
    use_mixed_precision: bool = True
    use_better_transformer: bool = True
    use_bucket_batching: bool = True
    use_vectorized_postproc: bool = True
    use_adaptive_beams: bool = True
    use_auto_batch_size: bool = False

    # ============ OTHER ============
    aggressive_postprocessing: bool = True
    checkpoint_freq: int = 100
    num_buckets: int = 4

    def __post_init__(self):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        # Disable CUDA-dependent optimizations if no GPU
        if not torch.cuda.is_available():
            self.use_mixed_precision = False
            self.use_better_transformer = False

        # (NEW) derived flag: bf16-only AMP will be used iff supported
        self.use_bf16_amp = bool(self.use_mixed_precision and self.device.type == "cuda" and _cuda_bf16_supported())


# ---------------------------
# 4) Logging Setup
# ---------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Define log file path
    log_file = Path(output_dir) / "inference_ultra.log"

    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    # Return logger
    return logging.getLogger(__name__)


# ---------------------------
# 5) Optimized Text Preprocessor
# ---------------------------

class OptimizedPreprocessor:
    # Initialize precompiled regex patterns
    def __init__(self):
        # Pre-compile patterns
        self.patterns = {
            "big_gap": re.compile(r"(\.{3,}|…+|……)"),
            "small_gap": re.compile(r"(xx+|\s+x\s+)"),
        }

    # Preprocess a single input text
    def preprocess_input_text(self, text: str) -> str:
        # Handle NaN
        if pd.isna(text):
            return ""

        # Convert to string
        cleaned_text = str(text)

        # Replace gaps
        cleaned_text = self.patterns["big_gap"].sub("<big_gap>", cleaned_text)
        cleaned_text = self.patterns["small_gap"].sub("<gap>", cleaned_text)

        return cleaned_text

    # Preprocess a batch of texts using vectorized pandas ops
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        # Convert to Series and sanitize
        s = pd.Series(texts).fillna("")
        s = s.astype(str)

        # Apply vectorized replacements
        s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)
        s = s.str.replace(self.patterns["small_gap"], "<gap>", regex=True)

        return s.tolist()


# ---------------------------
# 6) Vectorized Postprocessor
# ---------------------------

class VectorizedPostprocessor:
    # Initialize postprocessor patterns and translation tables
    def __init__(self, aggressive: bool = True):
        # Store mode
        self.aggressive = aggressive

        # Pre-compile patterns
        self.patterns = {
            "gap": re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I),
            "big_gap": re.compile(r"(\.{3,}|…|\[\.+\])"),
            "annotations": re.compile(r"\((fem|plur|pl|sing|singular|plural|\?|!)\..\s*\w*\)", re.I),
            "repeated_words": re.compile(r"\b(\w+)(?:\s+\1\b)+"),
            "whitespace": re.compile(r"\s+"),
            "punct_space": re.compile(r"\s+([.,:])"),
            "repeated_punct": re.compile(r"([.,])\1+"),
        }

        # Create character translation tables
        self.subscript_trans = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        self.special_chars_trans = str.maketrans("ḫḪ", "hH")

        # Define forbidden characters
        self.forbidden_chars = '!?()"——<>⌈⌋⌊[]+ʾ/;'

        # Create forbidden translate table
        self.forbidden_trans = str.maketrans("", "", self.forbidden_chars)

    # Vectorized postprocessing
    def postprocess_batch(self, translations: List[str]) -> List[str]:
        # Convert to Series
        s = pd.Series(translations)

        # (FIX) Validate entries -> boolean mask, not strings
        valid_mask = s.apply(lambda x: isinstance(x, str) and (len(x.strip()) > 0))
        if not bool(valid_mask.all()):
            s.loc[~valid_mask] = ""

        # Basic cleaning
        s = s.fillna("").astype(str)
        s = s.str.translate(self.special_chars_trans)
        s = s.str.translate(self.subscript_trans)
        s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
        s = s.str.strip()

        # Aggressive postprocessing
        if self.aggressive:
            # Normalize gaps
            s = s.str.replace(self.patterns["gap"], "<gap>", regex=True)
            s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)

            # Merge adjacent gaps
            s = s.str.replace("<gap> <gap>", "<big_gap>", regex=False)
            s = s.str.replace("<big_gap> <big_gap>", "<big_gap>", regex=False)

            # Remove annotations
            s = s.str.replace(self.patterns["annotations"], "", regex=True)

            # Protect gaps during forbidden-character removal
            s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
            s = s.str.replace("<big_gap>", "\x00BIG\x00", regex=False)

            # Remove forbidden characters
            s = s.str.translate(self.forbidden_trans)

            # Restore gaps
            s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
            s = s.str.replace("\x00BIG\x00", " <big_gap> ", regex=False)

            # Fractions
            s = s.str.replace(r"(\d+)\.5\b", r"\1½", regex=True)
            s = s.str.replace(r"\b0\.5\b", "½", regex=True)
            s = s.str.replace(r"(\d+)\.25\b", r"\1¼", regex=True)
            s = s.str.replace(r"\b0\.25\b", "¼", regex=True)
            s = s.str.replace(r"(\d+)\.75\b", r"\1¾", regex=True)
            s = s.str.replace(r"\b0\.75\b", "¾", regex=True)

            # Remove repeated words
            s = s.str.replace(self.patterns["repeated_words"], r"\1", regex=True)

            # Remove repeated n-grams (4 -> 2)
            for n in range(4, 1, -1):
                pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
                s = s.str.replace(pattern, r"\1", regex=True)

            # Fix punctuation spacing and repeats
            s = s.str.replace(self.patterns["punct_space"], r"\1", regex=True)
            s = s.str.replace(self.patterns["repeated_punct"], r"\1", regex=True)

            # Final whitespace cleanup
            s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
            s = s.str.strip().str.strip("-").str.strip()

        return s.tolist()


# ---------------------------
# 7) Bucket Batch Sampler
# ---------------------------

class BucketBatchSampler(Sampler):
    # Initialize sampler
    def __init__(self, dataset: Dataset, batch_size: int, num_buckets: int, logger: logging.Logger, shuffle: bool = False):
        # Store settings
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.logger = logger

        # Compute lengths for bucketing
        lengths = [len(text.split()) for _, text in dataset]

        # Sort indices by length
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        # Create buckets
        bucket_size = max(1, len(sorted_indices) // max(1, num_buckets))
        self.buckets = []

        for i in range(num_buckets):
            start = i * bucket_size
            end = None if i == num_buckets - 1 else (i + 1) * bucket_size
            self.buckets.append(sorted_indices[start:end])

        # Log bucket details
        self.logger.info(f"Created {num_buckets} buckets:")
        for i, bucket in enumerate(self.buckets):
            bucket_lengths = [lengths[idx] for idx in bucket] if len(bucket) > 0 else [0]
            self.logger.info(
                f"  Bucket {i}: {len(bucket)} samples, length range [{min(bucket_lengths)}, {max(bucket_lengths)}]"
            )

    # Yield batches
    def __iter__(self):
        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)

            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i : i + self.batch_size]

    # Return number of batches
    def __len__(self):
        total = 0
        for bucket in self.buckets:
            total += (len(bucket) + self.batch_size - 1) // self.batch_size
        return total


# ---------------------------
# 8) Dataset
# ---------------------------

class AkkadianDataset(Dataset):
    # Initialize dataset
    def __init__(self, dataframe: pd.DataFrame, preprocessor: OptimizedPreprocessor, logger: logging.Logger):
        # Store ids
        self.sample_ids = dataframe["id"].tolist()

        # Preprocess in batch
        raw_texts = dataframe["transliteration"].tolist()
        preprocessed = preprocessor.preprocess_batch(raw_texts)

        # Add task prefix
        self.input_texts = ["translate Akkadian to English: " + text for text in preprocessed]

        # Log dataset info
        logger.info(f"Dataset created with {len(self.sample_ids)} samples")

    # Return size
    def __len__(self):
        return len(self.sample_ids)

    # Return item
    def __getitem__(self, index: int):
        return self.sample_ids[index], self.input_texts[index]


# ---------------------------
# 9) Ultra-Optimized Inference Engine
# ---------------------------

class UltraInferenceEngine:
    # Initialize engine
    def __init__(self, config: UltraConfig, logger: logging.Logger):
        # Store config and logger
        self.config = config
        self.logger = logger

        # Create helpers
        self.preprocessor = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor(aggressive=config.aggressive_postprocessing)

        # (NEW) MBR metrics (sentence-level)
        # We use chrF++ only for MBR similarity (fast-ish, stable); BLEU not needed for selection.
        self._CHRFPP_SENT = sacrebleu.metrics.CHRF(word_order=2)

        # Initialize results
        self.results = []

        # Load model and tokenizer
        self._load_model()

        # (NEW) log BF16 mode clearly
        if self.config.device.type == "cuda":
            self.logger.info(f"BF16 supported: {_cuda_bf16_supported()}")
            self.logger.info(f"BF16 autocast enabled: {self.config.use_bf16_amp}")

    # Load and optimize model
    def _load_model(self):
        # Log model load
        self.logger.info(f"Loading model from {self.config.model_path}")

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
        self.model = self.model.to(self.config.device)
        self.model = self.model.eval()

        # Optional (safe) matmul precision hint
        if self.config.device.type == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        # Log parameter count
        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model loaded: {num_params:,} parameters")

        # Apply BetterTransformer if enabled
        if self.config.use_better_transformer and torch.cuda.is_available():
            try:
                # Import optimum lazily
                from optimum.bettertransformer import BetterTransformer

                # Apply transformation
                self.logger.info("Applying BetterTransformer...")
                self.model = BetterTransformer.transform(self.model)
                self.logger.info("✅ BetterTransformer applied (20-50% speedup)")
            except ImportError:
                self.logger.warning("⚠️  'optimum' not installed, skipping BetterTransformer")
                self.logger.warning("   Install with: !pip install optimum")
            except Exception as exc:
                self.logger.warning(f"⚠️  BetterTransformer failed: {exc}")

    # Collate function for DataLoader
    def _collate_fn(self, batch_samples):
        # Extract ids and texts
        batch_ids = [s[0] for s in batch_samples]
        batch_texts = [s[1] for s in batch_samples]

        # Tokenize
        tokenized = self.tokenizer(
            batch_texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return batch_ids, tokenized

    # Adaptive beam selection
    def _get_adaptive_beam_size(self, attention_mask: torch.Tensor) -> int:
        # Return fixed beams if disabled
        if not self.config.use_adaptive_beams:
            return self.config.num_beams

        # Compute lengths
        lengths = attention_mask.sum(dim=1)

        # Choose beams based on length
        short_beams = max(4, self.config.num_beams // 2)
        beam_sizes = torch.where(
            lengths < 100,
            torch.tensor(short_beams, device=lengths.device),
            torch.tensor(self.config.num_beams, device=lengths.device),
        )

        # Use first element (batch-wise adaptive is possible, but keep simple/fast)
        return int(beam_sizes[0].item())

    # Save periodic checkpoints
    def _save_checkpoint(self):
        # Checkpoint only when frequency matches
        if len(self.results) > 0 and len(self.results) % self.config.checkpoint_freq == 0:
            checkpoint_path = Path(self.config.output_dir) / f"checkpoint_{len(self.results)}.csv"

            # Create DataFrame
            df = pd.DataFrame(self.results, columns=["id", "translation"])

            # Save
            df.to_csv(checkpoint_path, index=False)

            # Log
            self.logger.info(f"💾 Checkpoint: {len(self.results)} translations")

    # Optional: auto-tune batch size
    def find_optimal_batch_size(self, dataset: Dataset, start_bs: int = 32) -> int:
        # Log
        self.logger.info("🔍 Finding optimal batch size...")

        # Initialize binary search
        max_bs = start_bs
        min_bs = 1

        # Binary search loop
        while max_bs - min_bs > 1:
            # Choose midpoint
            test_bs = (max_bs + min_bs) // 2

            try:
                # Build test batch
                test_batch = [dataset[i] for i in range(min(test_bs, len(dataset)))]

                # Collate
                _, inputs = self._collate_fn(test_batch)

                # Run a tiny generation (BF16 only, else FP32)
                with torch.inference_mode():
                    ctx = _bf16_autocast_ctx(self.config.device, enabled=self.config.use_bf16_amp)
                    with ctx:
                        _ = self.model.generate(
                            input_ids=inputs.input_ids.to(self.config.device),
                            attention_mask=inputs.attention_mask.to(self.config.device),
                            num_beams=self.config.num_beams,
                            max_new_tokens=64,
                            use_cache=True,
                        )

                # Update min bound
                min_bs = test_bs
                self.logger.info(f"  ✅ Batch size {test_bs} works")

                # Cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as exc:
                # Handle OOM
                if "out of memory" in str(exc).lower():
                    max_bs = test_bs
                    self.logger.info(f"  ❌ Batch size {test_bs} OOM")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

        # Log result
        self.logger.info(f"🎯 Optimal batch size: {min_bs}")
        return min_bs

    @staticmethod
    def _dedup_keep_order(xs):
        seen = set()
        out = []
        for x in xs:
            x = str(x).strip()
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def _sim_chrfpp(self, a: str, b: str) -> float:
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return 0.0
        # chrF++ sentence score
        return float(self._CHRFPP_SENT.sentence_score(a, [b]).score)

    def _mbr_pick(self, cands: List[str]) -> str:
        cands = self._dedup_keep_order(cands)
        if self.config.mbr_pool_cap is not None:
            cands = cands[: int(self.config.mbr_pool_cap)]

        n = len(cands)
        if n == 0:
            return ""
        if n == 1:
            return cands[0]

        # maximize average chrF++ similarity to others
        best_i, best_s = 0, -1e9
        for i in range(n):
            s = 0.0
            ai = cands[i]
            for j in range(n):
                if i == j:
                    continue
                s += self._sim_chrfpp(ai, cands[j])
            s /= max(1, n - 1)
            if s > best_s:
                best_s, best_i = s, i

        return cands[int(best_i)]

    def _generate_mbr_batch(self, input_ids, attention_mask, *, beam_size: int):
        """
        Returns: list[str] of length B (MBR-selected prediction per example)
        """
        gen_common = {
            "max_new_tokens": self.config.max_new_tokens,
            "repetition_penalty": 1.2,
            "use_cache": True,
        }

        B = int(input_ids.shape[0])

        # ----- beam candidates -----
        num_beam_cands = int(max(1, self.config.mbr_num_beam_cands))
        nb = int(max(1, beam_size))
        nb = int(max(nb, num_beam_cands))  # must be >= num_return_sequences

        beam_kwargs = dict(
            do_sample=False,
            num_beams=int(nb),
            num_return_sequences=int(num_beam_cands),
            length_penalty=float(self.config.length_penalty),
            early_stopping=bool(self.config.early_stopping),
        )

        beam_out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_common,
            **beam_kwargs,
        )
        beam_txt = self.tokenizer.batch_decode(beam_out, skip_special_tokens=True)

        pools = [[] for _ in range(B)]
        Rb = int(num_beam_cands)
        for i in range(B):
            pools[i].extend(beam_txt[i * Rb:(i + 1) * Rb])

        # ----- optional sampling candidates -----
        num_sample_cands = int(self.config.mbr_num_sample_cands)
        if num_sample_cands > 0:
            samp_out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                num_beams=1,
                top_p=float(self.config.mbr_top_p),
                temperature=float(self.config.mbr_temperature),
                num_return_sequences=int(num_sample_cands),
                max_new_tokens=int(self.config.max_new_tokens),
                use_cache=True,
            )
            samp_txt = self.tokenizer.batch_decode(samp_out, skip_special_tokens=True)

            Rs = int(num_sample_cands)
            for i in range(B):
                pools[i].extend(samp_txt[i * Rs:(i + 1) * Rs])

        # ----- MBR pick per example -----
        chosen = [self._mbr_pick(p) for p in pools]
        return chosen

    # Run inference end-to-end
    def run_inference(self, test_df: pd.DataFrame) -> pd.DataFrame:
        # Log start
        self.logger.info("🚀 Starting ULTRA-OPTIMIZED inference")

        # Create dataset
        dataset = AkkadianDataset(test_df, self.preprocessor, self.logger)

        # Auto-tune batch size if enabled
        if self.config.use_auto_batch_size:
            optimal_bs = self.find_optimal_batch_size(dataset)
            self.config.batch_size = optimal_bs

        # Create DataLoader
        if self.config.use_bucket_batching:
            batch_sampler = BucketBatchSampler(
                dataset=dataset,
                batch_size=self.config.batch_size,
                num_buckets=self.config.num_buckets,
                logger=self.logger,
                shuffle=False,
            )

            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False,
            )

        # Log dataloader and optimizations
        self.logger.info(f"DataLoader created: {len(dataloader)} batches")
        self.logger.info("Active optimizations:")
        self.logger.info(f"  ✅ Mixed Precision flag: {self.config.use_mixed_precision} (BF16)")
        self.logger.info(f"  ✅ BF16 autocast active: {self.config.use_bf16_amp}")
        self.logger.info(f"  ✅ BetterTransformer: {self.config.use_better_transformer}")
        self.logger.info(f"  ✅ Bucket Batching: {self.config.use_bucket_batching}")
        self.logger.info(f"  ✅ Vectorized Postproc: {self.config.use_vectorized_postproc}")
        self.logger.info(f"  ✅ Adaptive Beams: {self.config.use_adaptive_beams}")
        self.logger.info(
            f"  ✅ MBR: {self.config.use_mbr} "
            f"(beam_cands={self.config.mbr_num_beam_cands}, sample_cands={self.config.mbr_num_sample_cands})"
        )

        # Reset results
        self.results = []

        # Inference loop
        with torch.inference_mode():
            for batch_idx, (batch_ids, tokenized) in enumerate(tqdm(dataloader, desc="🚀 Translating")):
                try:
                    # Move inputs
                    input_ids = tokenized.input_ids.to(self.config.device)
                    attention_mask = tokenized.attention_mask.to(self.config.device)

                    # Adaptive beams
                    beam_size = self._get_adaptive_beam_size(attention_mask)

                    # BF16-only autocast context (else FP32)
                    ctx = _bf16_autocast_ctx(self.config.device, enabled=self.config.use_bf16_amp)

                    with ctx:
                        if self.config.use_mbr:
                            cleaned_raw = self._generate_mbr_batch(input_ids, attention_mask, beam_size=beam_size)
                        else:
                            outputs = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                num_beams=int(beam_size),
                                max_new_tokens=self.config.max_new_tokens,
                                length_penalty=self.config.length_penalty,
                                early_stopping=self.config.early_stopping,
                                use_cache=True,
                                **({"no_repeat_ngram_size": self.config.no_repeat_ngram_size}
                                   if self.config.no_repeat_ngram_size > 0 else {}),
                            )
                            cleaned_raw = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    # Postprocess
                    if self.config.use_vectorized_postproc:
                        cleaned = self.postprocessor.postprocess_batch(cleaned_raw)
                    else:
                        cleaned = [self.postprocessor.postprocess_batch([t])[0] for t in cleaned_raw]

                    # Store
                    self.results.extend(list(zip(batch_ids, cleaned)))

                    # Save checkpoint
                    self._save_checkpoint()

                    # Periodic cache clearing
                    if torch.cuda.is_available() and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

                except Exception as exc:
                    # Log error
                    self.logger.error(f"❌ Batch {batch_idx} error: {exc}")

                    # Fill empties for this batch
                    self.results.extend([(bid, "") for bid in batch_ids])

        # Log completion
        self.logger.info("✅ Inference completed")

        # Build results DataFrame
        results_df = pd.DataFrame(self.results, columns=["id", "translation"])

        # Validate
        self._validate_results(results_df)

        return results_df

    # Print validation report
    def _validate_results(self, df: pd.DataFrame):
        # Header
        print("\n" + "=" * 60)
        print("📊 VALIDATION REPORT")
        print("=" * 60)

        # Empty count
        empty = df["translation"].astype(str).str.strip().eq("").sum()
        print(f"\nEmpty: {empty} ({(empty / max(1, len(df))) * 100:.2f}%)")

        # Length stats
        lengths = df["translation"].astype(str).str.len()
        print("\n📏 Length stats:")
        print(f"   Mean: {lengths.mean():.1f}, Median: {lengths.median():.1f}")
        print(f"   Min: {lengths.min()}, Max: {lengths.max()}")

        # Very short translations
        short = ((lengths < 5) & (lengths > 0)).sum()
        if short > 0:
            print(f"   ⚠️  {short} very short translations")

        # Samples
        print("\n📝 Sample translations:")

        # Choose indices robustly
        sample_indices = [0]
        if len(df) > 2:
            sample_indices.append(len(df) // 2)
        if len(df) > 1:
            sample_indices.append(len(df) - 1)

        for idx in sample_indices:
            row = df.iloc[idx]
            text = str(row["translation"])
            preview = text[:70] + "..." if len(text) > 70 else text
            print(f"   ID {int(row['id']):4d}: {preview}")

        print("\n" + "=" * 60 + "\n")


# ---------------------------
# 10) IO Helpers
# ---------------------------

def print_environment_info():
    # Print PyTorch and CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_mem_gb:.2f} GB")
        print(f"BF16 supported: {_cuda_bf16_supported()} ")

    # Verify optimum availability (optional)
    try:
        from optimum.bettertransformer import BetterTransformer  # noqa: F401
        print("✅ BetterTransformer available!")
    except ImportError:
        print("❌ BetterTransformer NOT available")


def save_outputs(
    results_df: pd.DataFrame,
    config: UltraConfig,
    output_dir: str,
    logger: logging.Logger,
):
    # Define output paths
    output_path = Path(output_dir) / "submission2.csv"
    config_path = Path(output_dir) / "ultra_config.json"

    # Save submission
    results_df.to_csv(output_path, index=False)
    logger.info(f"✅ Submission saved to {output_path}")

    # Build config dict
    config_dict = {
        "batch_size": config.batch_size,
        "num_beams": config.num_beams,
        "length_penalty": config.length_penalty,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
        "bf16_only_amp": {
            "use_mixed_precision_flag": config.use_mixed_precision,
            "bf16_supported": _cuda_bf16_supported(),
            "bf16_autocast_active": config.use_bf16_amp,
            "never_fp16": True,
        },
        "mbr": {
            "use_mbr": config.use_mbr,
            "mbr_num_beam_cands": config.mbr_num_beam_cands,
            "mbr_num_sample_cands": config.mbr_num_sample_cands,
            "mbr_top_p": config.mbr_top_p,
            "mbr_temperature": config.mbr_temperature,
            "mbr_pool_cap": config.mbr_pool_cap,
        },
        "optimizations": {
            "better_transformer": config.use_better_transformer,
            "bucket_batching": config.use_bucket_batching,
            "vectorized_postproc": config.use_vectorized_postproc,
            "adaptive_beams": config.use_adaptive_beams,
        },
    }

    # Save config json
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("🎉 ULTRA-OPTIMIZED INFERENCE COMPLETE!")
    print("=" * 60)
    print(f"Submission file: {output_path}")
    print(f"Config file: {config_path}")
    print(f"Log file: {Path(output_dir) / 'inference_ultra.log'}")
    print(f"Total translations: {len(results_df)}")
    print("=" * 60)


def inspect_results(output_dir: str):
    # Load submission
    submission_path = Path(output_dir) / "submission2.csv"
    submission = pd.read_csv(submission_path)

    # Print quick view
    print(f"Submission shape: {submission.shape}")

    print("\nFirst 10 translations:")
    print(submission.head(10))

    print("\nLast 10 translations:")
    print(submission.tail(10))

    # Length statistics
    lengths = submission["translation"].astype(str).str.len()
    print("\nLength distribution:")
    print(lengths.describe())

    # Empty checks
    empty = submission["translation"].astype(str).str.strip().eq("").sum()
    print(f"\nEmpty translations: {empty}")

    if empty > 0:
        print("\nEmpty translation IDs:")
        print(submission[submission["translation"].astype(str).str.strip().eq("")]["id"].tolist())


# ---------------------------
# 11) Main
# ---------------------------

def main():
    # Create config
    config = UltraConfig()

    # Setup logger
    logger = setup_logging(config.output_dir)
    logger.info("Logging initialized")

    # Print environment info
    print_environment_info()

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Beams: {config.num_beams}")
    logger.info("Optimizations:")
    logger.info(f"  Mixed Precision flag: {config.use_mixed_precision} (BF16)")
    logger.info(f"  BF16 supported: {_cuda_bf16_supported()}")
    logger.info(f"  BF16 autocast active: {config.use_bf16_amp}")
    logger.info(f"  BetterTransformer: {config.use_better_transformer}")
    logger.info(f"  Bucket Batching: {config.use_bucket_batching}")
    logger.info(f"  Vectorized Postproc: {config.use_vectorized_postproc}")
    logger.info(f"  Adaptive Beams: {config.use_adaptive_beams}")
    logger.info(f"  MBR: {config.use_mbr}")

    # Load test data
    logger.info(f"Loading test data from {config.test_data_path}")
    test_df = pd.read_csv(config.test_data_path, encoding="utf-8")
    logger.info(f"✅ Loaded {len(test_df)} test samples")

    # Print first samples
    print("\nFirst 5 samples:")
    print(test_df.head())

    # Create engine
    engine = UltraInferenceEngine(config, logger)

    # Run inference
    results_df = engine.run_inference(test_df)

    # Save results and config
    save_outputs(results_df, config, config.output_dir, logger)

    # Optional inspection
    inspect_results(config.output_dir)


if __name__ == "__main__":
    main()
```

```python
import pandas as pd
import re

# =========================
# CONFIG
# =========================
CONFIG = {
    "blend_weights": [0.3, 0.7]  # Weight for submission1 (our) vs submission2 (external)
}

print(f"Blending weights: {CONFIG['blend_weights'][0]*100:.0f}% submission1 + {CONFIG['blend_weights'][1]*100:.0f}% submission2")

# =========================
# ENHANCED POSTPROCESSING
# =========================
def postprocess_translation(text):
    if not isinstance(text, str) or not text.strip(): 
        return "The tablet contains fragmentary text."
    
    processed_text = text.replace('ḫ', 'h').replace('Ḫ', 'H')
    sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    processed_text = processed_text.translate(sub_map)

    processed_text = re.sub(r'(\[x\]|\(x\)|\bx\b)', '<gap>', processed_text, flags=re.I)
    processed_text = re.sub(r'(\.{3,}|…|\[\.+\])', '<big_gap>', processed_text)
    
    processed_text = re.sub(r'<gap>\s*<gap>', ' <big_gap> ', processed_text)
    processed_text = re.sub(r'<big_gap>\s*<big_gap>', ' <big_gap> ', processed_text)

    processed_text = re.sub(r'\((fem|plur|pl|sing|singular|plural|\?|!)\.?\s*\w*\)', '', processed_text, flags=re.I)

    processed_text = processed_text.replace('<gap>', '\x00GAP\x00').replace('<big_gap>', '\x00BIG\x00')
    
    # Remove bad characters
    bad_chars = '!?()"—–<>⌈⌋⌊[]+ʾ/;'
    processed_text = processed_text.translate(str.maketrans('', '', bad_chars))

    processed_text = processed_text.replace('\x00GAP\x00', ' <gap> ').replace('\x00BIG\x00', ' <big_gap> ')

    # Handle fractions
    frac_map = {
        r'\.5\b': ' ½', r'\.25\b': ' ¼', r'\.75\b': ' ¾',
        r'\.33+\d*\b': ' ⅓', r'\.66+\d*\b': ' ⅔'
    }
    for pat, rep in frac_map.items():
        processed_text = re.sub(r'(\d+)' + pat, r'\1' + rep, processed_text)
        processed_text = re.sub(r'\b0' + pat, rep.strip(), processed_text)

    # Remove repeated words
    processed_text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', processed_text)
    for n in range(4, 1, -1):
        pat = r'\b((?:\w+\s+){' + str(n-1) + r'}\w+)(?:\s+\1\b)+'
        processed_text = re.sub(pat, r'\1', processed_text)
    
    # Additional improvements for blending
    # Capitalize first letter
    if processed_text and processed_text[0].islower():
        processed_text = processed_text[0].upper() + processed_text[1:]
    
    # Ensure proper ending
    if processed_text and processed_text[-1] not in '.!?':
        processed_text = processed_text + '.'
    
    return re.sub(r'\s+', ' ', processed_text).strip().strip('-')

# =========================
# TEXT BLENDING FUNCTION
# =========================
def blend_translations(text1, text2, weight1=0.7, weight2=0.3):
    """
    Blend two translations by selecting the better one based on heuristics
    or combining them intelligently
    """
    if not text1 or not text2:
        return text1 if text1 else text2
    
    # Simple weighted selection based on quality heuristics
    def score_text(text):
        score = 0
        words = text.split()
        
        # Length preference (optimal 10-40 words)
        if 10 <= len(words) <= 40:
            score += 2
        elif 5 <= len(words) <= 50:
            score += 1
        
        # Starts with capital
        if text and text[0].isupper():
            score += 1
        
        # Ends with punctuation
        if text and text[-1] in '.!?':
            score += 1
        
        # Contains Akkadian keywords
        akkadian_keywords = ['tablet', 'king', 'city', 'god', 'said', 'wrote', 'year']
        for keyword in akkadian_keywords:
            if keyword in text.lower():
                score += 0.5
        
        # No obvious artifacts
        if '???' not in text and '...' not in text and 'xxx' not in text.lower():
            score += 1
        
        return score
    
    score1 = score_text(text1)
    score2 = score_text(text2)
    
    # Weighted score
    weighted_score1 = score1 * weight1
    weighted_score2 = score2 * weight2
    
    # Return the better one
    if weighted_score1 >= weighted_score2:
        return text1
    else:
        return text2

# =========================
# MAIN BLENDING EXECUTION
# =========================
# Load submissions
print("Loading submissions...")
submission1 = pd.read_csv("submission1.csv")  # Our model's predictions
submission2 = pd.read_csv("submission2.csv")  # External predictions

# Create dicts for quick lookup
submission1_dict = dict(zip(submission1['id'], submission1['translation']))
submission2_dict = dict(zip(submission2['id'], submission2['translation']))

print(f"Loaded {len(submission1_dict)} from submission1")
print(f"Loaded {len(submission2_dict)} from submission2")

# Blend predictions
print("\nBlending predictions...")
blended_results = []

ids = sorted(set(list(submission1_dict.keys()) + list(submission2_dict.keys())))

for id_ in ids:
    trans1 = submission1_dict.get(id_, "")
    trans2 = submission2_dict.get(id_, "")
    
    # Apply postprocess to trans1 (assuming it's the "our" model output)
    trans1 = postprocess_translation(trans1)
    
    # Blend
    blended = blend_translations(
        trans1, 
        trans2,
        weight1=CONFIG['blend_weights'][0],
        weight2=CONFIG['blend_weights'][1]
    )
    
    # If blending failed, fall back to trans1
    if not blended or blended.strip() == "":
        blended = trans1
    
    blended_results.append((id_, blended))

# =========================
# CREATE FINAL SUBMISSION
# =========================
submission_df = pd.DataFrame(blended_results, columns=['id', 'translation'])

# Quality check
broken_count = sum(1 for t in submission_df['translation'] 
                   if len(t.split()) < 3 or "fragmentary" in t.lower())
print(f"\nQuality check: {broken_count} potentially problematic translations")

# Fix any remaining issues
submission_df['translation'] = submission_df['translation'].apply(
    lambda x: "The tablet contains an incomplete inscription." 
    if len(x.split()) < 3 else x
)

# Save
submission_df.to_csv("submission.csv", index=False)
print(f"\n✅ Saved blended submission with {len(submission_df)} rows")
```
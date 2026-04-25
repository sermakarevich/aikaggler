# Fork of 0.606_AFTER_LLM 993720

- **Author:** @🤞@
- **Votes:** 358
- **Ref:** anglolodorf/fork-of-0-606-after-llm-993720
- **URL:** https://www.kaggle.com/code/anglolodorf/fork-of-0-606-after-llm-993720
- **Last run:** 2025-09-02 19:37:22.317000

---

## add
- post_validate.py : remove fp
- predict.py : use llb to predict primary or secondary

```python
! uv pip uninstall --system 'tensorflow'
! uv pip install --system --no-index --find-links='/kaggle/input/latest-mdc-whls/whls' 'pymupdf' 'vllm' 'triton' 'logits-processor-zoo' 'numpy<2'
! mkdir -p /tmp/src
```

```python
%%writefile /tmp/src/helpers.py
import logging, os, kagglehub, inspect
from pathlib import Path
import polars as pl

IS_KAGGLE_ENV = sum(['KAGGLE' in k for k in os.environ]) > 0
IS_KAGGLE_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
COMP_DIR = Path(('/kaggle/input/make-data-count-finding-data-references' if IS_KAGGLE_SUBMISSION else kagglehub.competition_download('make-data-count-finding-data-references')))
PDF_DIR = COMP_DIR / ('test' if IS_KAGGLE_SUBMISSION else 'train') / 'PDF'
WORKING_DIR = Path(('/kaggle/working/' if IS_KAGGLE_ENV else '.working/'))

DOI_LINK = 'https://doi.org/'

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper() if not IS_KAGGLE_SUBMISSION else "WARNING"
LOG_FILE_PATH = os.getenv("LOG_FILE", "logs/project.log")
LOG_DIR = Path(LOG_FILE_PATH).parent

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(levelname)s %(asctime)s  [%(filename)s:%(lineno)d - %(funcName)s()] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def get_logger(name=None):
    if name is None:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            name = "__main__"
        else:
            name = frame.f_back.f_globals.get("__name__", "__main__")

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(DEFAULT_LOG_LEVEL)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
        ch = logging.StreamHandler()
        ch.setLevel(DEFAULT_LOG_LEVEL)
        ch.setFormatter(formatter)
        fh = logging.FileHandler(LOG_FILE_PATH)
        fh.setLevel(DEFAULT_LOG_LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
        logger.propagate = False
    return logger

def is_doi_link(name: str) -> pl.Expr:
    return pl.col(name).str.starts_with(DOI_LINK).and_(
        ~pl.col(name).str.contains(r"/dl\.")
    )

def string_normalization(name: str) -> pl.Expr:
    return pl.col(name).str.normalize("NFKC").str.replace_all(r"[^\p{Ascii}]", '').str.replace_all(r"https?://zenodo\.org/record/(\d+)", r" 10.5281/zenodo.$1 ")

def get_df(parse_dir: str):
    records = []
    txt_files = list(Path(parse_dir).glob('*.txt'))
    for txt_file in txt_files:
        id_ = txt_file.stem
        with open(txt_file, 'r') as f:
            text = f.read()
        records.append({'article_id': id_, 'text': text})
    return pl.DataFrame(records).with_columns(string_normalization('text').alias('text'))

def assume_type(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.when(is_doi_link('dataset_id').or_(pl.col('dataset_id').str.starts_with('SAMN'))).then(pl.lit('Primary')).otherwise(pl.lit('Secondary')).alias('type'))
    )

def score(df, gt, on, tag='all'):
    hits = gt.join(df, on=on)
    tp = hits.height
    fp = df.height - tp
    fn = gt.height - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    return f"{tag} - f1: {f1:.4f} [{tp}/{fp}/{fn}]"

def evaluate(df, on=['article_id', 'dataset_id']):
    gt = pl.read_csv(COMP_DIR/'train_labels.csv').filter(pl.col('type')!='Missing')
    return (
        score(df, gt, on),
        score(df.filter(is_doi_link('dataset_id')), gt.filter(is_doi_link('dataset_id')), on, 'doi'),
        score(df.filter(~is_doi_link('dataset_id')), gt.filter(~is_doi_link('dataset_id')), on, 'acc'),
    )
```

```python
%%writefile /tmp/src/parse.py
import argparse
from pathlib import Path
import pymupdf
from helpers import get_logger, PDF_DIR

l = get_logger()

def pdf_to_txt(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(PDF_DIR.glob("*.pdf")) + list(PDF_DIR.glob("*.PDF"))
    existing_txt_files = {f.stem for f in output_dir.glob("*.txt")}
    for pdf_file in pdf_files:
        txt_file = output_dir / f"{pdf_file.stem}.txt"
        if pdf_file.stem in existing_txt_files:
            continue
        try:
            text = ""
            with pymupdf.open(pdf_file) as doc:
                for page in doc:
                    text += page.get_text()
            txt_file.write_text(text, encoding='utf-8')
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=Path, help='Directory to save text files')
    args = parser.parse_args()
    pdf_to_txt(args.output_dir)

if __name__ == "__main__":
    main()
```

```python
%%writefile /tmp/src/check_parse.py
import polars as pl
from pathlib import Path
from helpers import *

l=get_logger()

def gt_dataset_id_normalization(name:str) -> pl.Expr:
    return (
        pl.when(is_doi_link(name))
        .then(pl.col(name).str.split(DOI_LINK).list.last())
        .otherwise(name)
        .str.to_lowercase()
    )

def main():
    if IS_KAGGLE_SUBMISSION:
        l.debug('skipping check_parse for submission')
        return
    df = (
        get_df('/tmp/train_parse')
        .with_columns(pl.col('text').str.replace_all('\s+', '').str.to_lowercase().alias('text'))
    )

    gt = (
        pl.read_csv(COMP_DIR/'train_labels.csv')
        .filter(pl.col('article_id').is_in(df['article_id']))
        .filter(pl.col('type')!='Missing')
        .with_columns(gt_dataset_id_normalization('dataset_id').alias('norm_id'))
    )

    l.info(f"pymupdf misses: {gt.join(df, on='article_id').with_columns(hit=pl.col('text').str.contains(pl.col('norm_id'), literal=True)).filter(~pl.col('hit')).height} dataset_ids")

if __name__=='__main__': main()
```

```python
%%writefile /tmp/src/getid.py
import re
import polars as pl
from typing import Optional, Tuple

from helpers import *

COMPILED_PATTERNS = {
    'ref_header_patterns': [re.compile(r'\b(R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S|BIBLIOGRAPHY|LITERATURE CITED|WORKS CITED|CITED WORKS|ACKNOWLEDGEMENTS)\b[:\s]*', re.IGNORECASE)],    
    'citation_pattern': re.compile(r'^\s*(\[\d+\]|\(\d+\)|\d+\.|\d+\)|\d+(?=\s|$))\s*'),
    'first_citation_patterns': [
        re.compile(r'^\s*\[1\]\s*'),
        re.compile(r'^\s*\(1\)\s*'),
        re.compile(r'^\s*1\.\s*'),
        re.compile(r'^\s*1\)\s*'),
        re.compile(r'^\s*1(?=\s|$)'),
    ],
}

l = get_logger()

def find_last_reference_header(text: str, header_patterns: list[re.Pattern]) -> Optional[int]:
    last_match_idx = None
    for pattern in header_patterns:
        matches = list(pattern.finditer(text))
        if matches:
            last_match_idx = matches[-1].start()
    return last_match_idx

def find_last_first_citation(text: str) -> Optional[int]:
    lines = text.splitlines()
    last_match_line = None
    for line_num, line in enumerate(lines):
        line = line.strip()
        for pattern in COMPILED_PATTERNS['first_citation_patterns']:
            if pattern.match(line):
                next_lines = lines[line_num:line_num+3]
                if any(COMPILED_PATTERNS['citation_pattern'].match(l.strip()) for l in next_lines[1:]):
                    last_match_line = line_num
                break
    return last_match_line

def find_reference_start(text: str) -> Optional[int]:
    lines = text.splitlines()
    last_first_citation = find_last_first_citation(text)
    if last_first_citation is not None:
        return last_first_citation
    start_search_idx = int(len(lines) * 0.5)
    for i in range(start_search_idx, len(lines)):
        line = lines[i].strip()
        if COMPILED_PATTERNS['citation_pattern'].match(line):
            next_lines = lines[i:i+3]
            if sum(1 for l in next_lines if COMPILED_PATTERNS['citation_pattern'].match(l.strip())) >= 2:
                for j in range(i, max(-1, i-10), -1):
                    if not COMPILED_PATTERNS['citation_pattern'].match(lines[j].strip()):
                        return j + 1
                return max(0, i-10)
    return None

def split_text_and_references(text: str) -> Tuple[str, str]:
    header_idx = find_last_reference_header(text, COMPILED_PATTERNS['ref_header_patterns'])
    if header_idx is not None:
        header_idx2 = find_last_reference_header(text[:header_idx].strip(), COMPILED_PATTERNS['ref_header_patterns'])
        if header_idx2 is not None:
            header_idx3 = find_last_reference_header(text[:header_idx2].strip(), COMPILED_PATTERNS['ref_header_patterns'])
            if header_idx3 is not None:
                return text[:header_idx3].strip(), text[header_idx3:].strip()
            return text[:header_idx2].strip(), text[header_idx2:].strip()
        return text[:header_idx].strip(), text[header_idx:].strip()
    ref_start_line = find_reference_start(text)
    if ref_start_line is not None:
        lines = text.splitlines()
        body = '\n'.join(lines[:ref_start_line])
        refs = '\n'.join(lines[ref_start_line:])
        return body.strip(), refs.strip()
    return text.strip(), ''

def get_splits(df: pl.DataFrame) -> pl.DataFrame:
    bodies, refs = [], []
    for raw_text in df['text']:
        main, ref = split_text_and_references(raw_text)
        bodies.append(main)
        refs.append(ref)
    return df.with_columns(pl.Series('body', bodies), pl.Series('ref', refs))

def doi_gbif_ids(df):
    doi_pattern = re.compile(r'10\s*\.\s*\d{4,9}\s*/\s*\S+')
    DOI_LINK="https://doi.org/"
    records = []
    for article_id, ref_text in df.select(["article_id","ref"]).rows():
        idx = ref_text.find("GBIF Occurrence")
        if idx == -1:
            continue
        context_window = []
        context_len = 110
        rem_ref = ref_text[idx:]
        while idx != -1 :
            if len(rem_ref) < context_len:
                context_window.append(rem_ref)
                doi_matches = doi_pattern.findall(rem_ref)
            else:
                context_window.append(rem_ref[:context_len])
                doi_matches = doi_pattern.findall(rem_ref[:context_len])
    
            for match in doi_matches:
                cleaned = re.sub(r'\s+', '', match)         
                cleaned = re.sub(r'[^A-Za-z0-9]+$', '', cleaned)  
                cleaned = cleaned.lower()
                dataset_id = DOI_LINK + cleaned
                records.append((article_id, dataset_id, match))
            rem_ref = rem_ref[1:]
            idx = rem_ref.find("GBIF Occurrence")
            if idx != -1 :
                rem_ref = rem_ref[idx:]
    doi_df = pl.DataFrame(records, schema=["article_id", "dataset_id", "match"], orient="row").unique(subset=["article_id", "dataset_id"])
    doi_df= doi_df.with_columns(
        pl.col("match").map_elements(lambda x: [x] if x is not None else [], return_dtype=pl.List(pl.Utf8))
    )
    return doi_df

def tidy_extraction(df) -> pl.DataFrame:
    bad_ids = [f'{DOI_LINK}{e}' for e in ['10.5061/dryad', '10.5281/zenodo', '10.6073/pasta']]

    doi_df = (
        df.with_columns(pl.col('body').str.extract_all(r'10\s*\.\s*\d{4,9}\s*/\s*\S+').alias('match'))
          .explode('match')
          .drop_nulls('match')
          .with_columns(
              pl.col('match').str.replace_all(r'\s+', '')
                             .str.replace(r'[^A-Za-z0-9]+$', '')
                             .str.to_lowercase()
                             .alias('dataset_id')
          )
          .group_by('article_id', 'dataset_id')
          .agg('match')
          .with_columns((DOI_LINK + pl.col('dataset_id')).alias('dataset_id'))
    )

    extra_doi = doi_gbif_ids(df)
    doi_df = pl.concat([doi_df,extra_doi])

    ref_doi = (
        df.with_columns(pl.col('ref').str.extract_all(r'10\s*\.\s*\d{4,9}\s*/\s*\S+').alias('match'))
          .explode('match')
          .drop_nulls('match')
          .with_columns(
              pl.col('match').str.replace_all(r'\s+', '')
                             .str.replace(r'[^A-Za-z0-9]+$', '')
                             .str.to_lowercase()
                             .alias('dataset_id')
          )
          .group_by('article_id', 'dataset_id')
          .agg('match')
          .with_columns((DOI_LINK + pl.col('dataset_id')).alias('dataset_id'))
    )

    doi_df = pl.concat([doi_df,ref_doi])

    REGEX_IDS = (
        r"(?i)\b(?:"
        r"CHEMBL\d+|"
        r"E-GEOD-\d+|E-PROT-\d+|E-MTAB-\d+|E-MEXP-\d+|EMPIAR-\d+|"
        r"ENSBTAG\d+|ENSOARG\d+|"
        r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
        r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|BX\d{6}|KX\d{6}|K0\d{4}|CAB\d{6}|"
        r"NC_\d{6}\.\d{1}|NM_\d{9}|"
        r"PRJNA\d+|PRJEB\d+|PRJDB\d+|PXD\d+|SAMN\d+|"
        r"GSE\d+|GSM\d+|GPL\d+|"
        r"PDB\s?[1-9][A-Z0-9]{3}|HMDB\d+|"
        r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+|"
        r"(?:SR[PRX]|STH|ERR|DRR|DRX|DRP|ERP|ERX)\d+|"
        r"CVCL_[A-Z0-9]{4}|"
        r"[1-5]\.(?:10|20|30|40|50|60|70|80|90)\.\d{2,4}\.\d{2,4}"
        r")"
    )

    acc_df = (
        df.with_columns(
            pl.col('text').str.extract_all(REGEX_IDS).alias('match')
        )
        .explode('match')
        .drop_nulls('match')
        .with_columns(
            pl.col('match').str.replace_all(r'\s+', '')
                           .str.replace(r'[^A-Za-z0-9]+$', '')
                           .str.replace(r'(?i)^PDB', '')
                           .alias('dataset_id')
        )
        .group_by('article_id', 'dataset_id')
        .agg('match')
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('dryad.'))
              .then(f'{DOI_LINK}10.5061/' + pl.col('dataset_id'))
              .otherwise('dataset_id')
              .alias('dataset_id')
        )
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('pasta/'))
              .then(f'{DOI_LINK}10.6073/' + pl.col('dataset_id'))
              .otherwise('dataset_id')
              .alias('dataset_id')
        )
    )

    df = pl.concat([doi_df, acc_df])

    df = (
        df.unique(['article_id', 'dataset_id'])  # CHANGED
          .filter(~pl.col('article_id').str.replace('_','/').str.contains(pl.col('dataset_id').str.split(DOI_LINK).list.last().str.escape_regex()))
          .filter(~pl.col('dataset_id').str.contains(pl.col('article_id').str.replace('_','/').str.escape_regex()))
          .filter(~pl.col('dataset_id').str.contains('figshare', literal=True))
          .filter(~pl.col('dataset_id').is_in(bad_ids))
          .filter(
              pl.when(is_doi_link('dataset_id') &
                      (pl.col('dataset_id').str.split('/').list.last().str.len_chars() < 5))
               .then(False)
               .otherwise(True)
          )
          .with_columns(pl.col('match').list.unique())
    )
    return df

def get_context_window(text: str, substring: str, window: int = 100) -> str:
    idx = text.find(substring)
    if idx == -1:
        raise ValueError
    start = max(idx - window, 0)
    end = min(idx + len(substring) + window, len(text))
    return text[start:end]

def get_window_df(text_df, ids_df):
    df = ids_df.join(text_df, on='article_id')
    windows = []
    for text, match_ids in df.select('text', 'match').rows():
        windows.append(get_context_window(text, match_ids[0]))
    return df.with_columns(pl.Series('window', windows)).select('article_id', 'dataset_id', 'window')

def write_the_match(text_df,id_df):
    df = id_df.join(text_df,on=['article_id'])
    records=[]
    for art_id,dataset_id,match_ids,text in df.select('article_id','dataset_id','match','text').rows():
        records.append({'article_id':art_id,'dataset_id':dataset_id,'match':match_ids[0],'text':text})

    pl.DataFrame(records).write_parquet('/tmp/context_data.parquet')

def main():
    text_df = get_df('/tmp/train_parse')
    df = get_splits(text_df)
    df = tidy_extraction(df)
        
    write_the_match(text_df,df)
    
    df = get_window_df(text_df, df)
    df.write_parquet('/tmp/extracted.parquet')
    df = assume_type(df)
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
    if not IS_KAGGLE_SUBMISSION:
        results = evaluate(df)
        for r in results: l.info(r)
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results: l.info(r)

if __name__=='__main__': main()
```

```python
%%writefile /tmp/src/llm_validate.py
import polars as pl
import os

from helpers import *

l = get_logger()

SYS_PROMPT_CLASSIFY_DOI = """
1. Priority Rules (highest → lowest)
1.1 Always classify as A (Data) if:
DOI prefix matches a known data repository:

Dryad: 10.5061

Zenodo: 10.5281

Dl: 10.15468

ICPSR: 10.3886

USGS data: 10.5066

Mendeley Data: 10.17632

Dataverse: 10.7910/DVN

OpenNeuro: 10.18112/openneuro.

PANGAEA: 10.1594/PANGAEA.


2. Classify as B (Literature) if:
DOI prefix belongs to a publisher (e.g., 10.1038, 10.1007, 10.1126, 10.1016, 10.1101, 10.1021, 10.1145, 10.1177, 10.1093, 10.1080, 10.1111, etc.).

Context indicates a journal article, book, conference paper, preprint, protocol, or method paper, without any repository/data storage signal.

Mentions only “supplementary material” or “supplementary information” without a repository.

3. Ambiguous cases
No repository prefix and no clear context → default to B.


4. Output
Only output:

A → data repository / dataset

B → literature / non-data resource

Few-shot examples

“Raw images are stored on Figshare (DOI 10.6084/m9.figshare.1234567).” → A

“Sequence reads available under BioProject accession PRJNA765432.” → A

“As described in Nature Methods (DOI 10.1038/s41592-020-0793-2).” → B

“See Supplementary Data at Zenodo (10.5281/zenodo.987654).” → A

“Method details published in J. Proteome Res. DOI: 10.1021/acs.jproteome.0c00845.” → B

“Data uploaded to Dryad (10.5061/dryad.x1y2z3).” → A

“Referenced paper: DOI 10.1101/2020.01.01.123456 (bioRxiv preprint).” → B

“Metabolomics data in MetaboLights MTBLS1234.” → A

“The MRI scans are deposited at OpenNeuro (DOI 10.18112/openneuro.ds000001.v1.0.0).” → A

“Protein structure described in Science (DOI 10.1126/science.abc1234).” → B
""".strip()

def build_df():
    df = pl.read_parquet('/tmp/extracted.parquet')
    df.filter(~is_doi_link('dataset_id')).select('article_id', 'dataset_id').write_csv('/tmp/accid_sub.csv')
    return df.filter(is_doi_link('dataset_id'))

def build_prompt(tokenizer, df):
    prompts = []
    for doi, text in df.select('dataset_id', 'window').rows():
        messages = [{'role':'system','content': SYS_PROMPT_CLASSIFY_DOI}, {'role':'user', 'content': text}]
        prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
    return df.with_columns(pl.Series('prompt', prompts))

if __name__=='__main__':
    os.environ["VLLM_USE_V1"] = "0"
    import vllm
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
    model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
    llm = vllm.LLM(model_path, quantization='awq', tensor_parallel_size=2, gpu_memory_utilization=0.9, trust_remote_code=True, dtype="half", enforce_eager=True, max_model_len=2048, disable_log_stats=True, disable_custom_all_reduce=True, enable_prefix_caching=True, task='generate')
    tokenizer = llm.get_tokenizer()
    df = build_df()
    df = build_prompt(tokenizer, df)
    prompts = df['prompt'].to_list()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])
    outputs = llm.generate(prompts, vllm.SamplingParams(seed=777, temperature=0.2, skip_special_tokens=True, max_tokens=1, logits_processors=[mclp], logprobs=len(mclp.choices)), use_tqdm=True)
    logprobs = [{lp.decoded_token: lp.logprob for lp in list(lps)} for lps in [output.outputs[0].logprobs[0].values() for output in outputs]]
    choices = [max(d, key=d.get) for d in logprobs]
    types = {'A': True, 'B': False}
    choices = [types[c] for c in choices]
    df = df.with_columns(pl.Series('type', choices))
    df.filter(pl.col('type')).select('article_id', 'dataset_id').write_csv('/tmp/doi_sub.csv')
    df = pl.concat([pl.read_csv('/tmp/doi_sub.csv'), pl.read_csv('/tmp/accid_sub.csv')])
    df = assume_type(df)
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
    if not IS_KAGGLE_SUBMISSION:
        results = evaluate(df)
        for r in results: l.info(r) 
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results: l.info(r)


    
    try:
        del llm, tokenizer
    except:
        pass
    
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
```

```python
%%writefile /tmp/src/post_filter.py
import polars as pl
from helpers import *

"""
Fourth essence: Post-filter to cut FP DOIs that look like literature.
- Read /kaggle/working/submission.csv (output of llm_validate.py)
- Join with /tmp/extracted.parquet to get context window
- Drop DOI rows that (1) start with typical publisher prefixes AND (2) have no data-ish words nearby
- Keep accessions untouched
"""

l = get_logger()

PAPER_PREFIXES = [
    "10.5061","10.5281","10.17632","10.1594","10.15468","10.17882","10.7937","10.7910","10.6073",
    "10.3886","10.3334","10.4121","10.5066","10.5067","10.18150","10.25377","10.25387","10.23642","10.24381","10.22033"
]

CONTEXT_RE = r"(?i)\b(data(?:set)?|repository|archive|deposited|available|supplementary|raw(?:\s+data)?|uploaded|hosted|stored|accession)\b"

def is_paper_prefix(col: str = "dataset_id") -> pl.Expr:
    expr = pl.lit(False)
    for p in PAPER_PREFIXES:
        expr = expr | pl.col(col).str.starts_with(f"{DOI_LINK}{p}")
    return expr

def main():
    sub = pl.read_csv("/kaggle/working/submission.csv")

    # Normalize columns: drop row_id if present so concat widths match
    if "row_id" in sub.columns:
        sub = sub.drop("row_id")

    # Context windows
    win = pl.read_parquet("/tmp/extracted.parquet").select("article_id", "dataset_id", "window")

    # DOI & ACC split
    doi_rows = sub.filter(is_doi_link("dataset_id")).join(win, on=["article_id", "dataset_id"], how="left")
    acc_rows = sub.filter(~is_doi_link("dataset_id"))

    keep_mask = (
        (~is_paper_prefix("dataset_id"))  # not a known paper prefix
        | doi_rows["window"].fill_null("").str.contains(CONTEXT_RE)
    )

    kept_doi = doi_rows.filter(keep_mask).select("article_id", "dataset_id", "type")
    final = pl.concat([kept_doi, acc_rows.select("article_id", "dataset_id", "type")])

    # Re-eval & save
    if not IS_KAGGLE_SUBMISSION:
        for r in evaluate(final): l.info(r)
        for r in evaluate(final, on=["article_id", "dataset_id", "type"]): l.info(r)

    final.with_row_index("row_id").write_csv("/kaggle/working/submission.csv")

if __name__ == "__main__":
    main()
```

```python
%%writefile /tmp/src/post_validate.py

from helpers import *
import polars as pl
import os


l = get_logger()


PROMPT_CLASSIFY_CITATION_TYPE = '''
# Role & Task
You are an expert data citation analyst. Your task is to classify a given citation from a scientific paper into one of two categories: **A** (Data) or **B** (Not Data). Base your decision strictly on the provided abstract and the context of the citation.

## Instructions
1.  **Read the provided abstract** to understand the research context.
2.  **Analyze the citation context** for key linguistic cues.
3.  **Classify the citation** as either **A** or **B** based on the definitions below.
4.  **Output only a single letter: A or B.** Do not output any other text, explanation, or formatting.

## Category Definitions

### **Category A: DATA**
The citation points to a dataset. This includes:
*   **Primary Data:** Raw or processed data that the current study's authors collected, generated, or created.
*   **Secondary Data:** Data that was originally produced by other researchers but is being *used as a dataset* in the current study.
*   **Key Phrases:** "data are available at", "we collected", "we measured", "data were obtained from", "dataset", "downloaded from", "deposited in", repository names (e.g., GenBank, Zenodo, Figshare, TCIA).

### **Category B: NOT DATA**
The citation points to a traditional scholarly publication or other non-data resource. This includes:
*   Journal articles, books, conference proceedings, preprints, protocols, methods papers.
*   **Key Phrases:** "as described in", "according to", "previous study", "et al.", "paper", "article", "methodology", "was used for analysis" (without indicating data access).
*   Citations that provide background context or methodological description but do not serve as the source of the data used in the analysis.

## Input Format
You will be provided with the following three pieces of information:
Paper Abstract: {abstract}
Citation: {dataset_id}
Citation Context: {context}

## Critical Thinking Guidelines
*   A DOI or URL can point to either data (A) or a paper (B). The context determines the classification.
*   If the citation is used to describe the *source* of the data for the current study's analysis, it is likely **A**.
*   If the citation is used to provide background, justify a method, or compare results, it is likely **B** (a reference to another paper).
*   When in doubt, rely on the linguistic cues in the "Citation Context".

## Examples for Pattern Recognition

**Example 1 (Classify as A):**
*   Context: "Three out of four cohorts used in this study can be found on The Cancer Imaging Archive (TCIA)24: Canadian benchmark dataset23: https://doi.org/10.7937/K9/TCIA.2017.8oje5q00."
*   **Reasoning:** The text states cohorts are "used in this study" and provides direct repository links. This is a clear case of citing external data for use.
*   **Output:** A

**Example 2 (Classify as B):**
*   Context: "data presented here are available at the SEANOE dataportal: https://doi.org/10.17882/94052 (ZooScan dataset Grandremy et al. 2023c)"
*   **Reasoning:** The phrase "data presented here" indicates this is the authors' own data being deposited, not a citation to an external source they are using. The "(Author et al. Year)" format is a classic literature citation style.
*   **Output:** B

**Example 3 (Classify as A):**
*   Context: "GBIF occurrence data: Vulpes vulpes: https://doi.org/10.15468/dl.wgtneb (28 May 2021)."
*   **Reasoning:** Explicitly names the data source (GBIF) and provides a direct access link/DOI for the specific dataset used.
*   **Output:** A

**Example 4 (Classify as B):**
*   Context: "North American soil NCBI SRA SRP035367 Smith & Peay [36] ITS2-Soil"
*   **Reasoning:** While it mentions a data repository ID (SRP035367), it couples it with a standard literature citation "[36]". The context suggests it is referencing the *paper* by Smith & Peay that describes the data, not directly citing the dataset itself for use.
*   **Output:** B

## Ready for Input
Begin your analysis. Remember: Output only **A** or **B**.
'''

def get_context_window(text: str, substring: str, window: int = 600) -> str:
    idx = text.find(substring)
    if idx == -1:
        return "no context", "no abstraction"
    start = max(idx - window, 0)
    end = min(idx + len(substring) + window, len(text))
    return text[start:end] , text[:1000]




def find_context_win(tokenizer,df):
    text_df = pl.read_parquet('/tmp/context_data.parquet')
    # print(text_df)
    df = df.join(text_df, on=["article_id","dataset_id"], how="inner")
    df = df.drop("type")
    print(df)

    prompts = []
    
    for article_id,dataset_id,text,match in df.select(["article_id","dataset_id","text",'match']).rows():

        context, abstract = get_context_window(text,match)
        user_content = f"""
        Paper Abstract: {abstract}
        
        Citation: {dataset_id}

        
        Citation Context: {context}
        """
        messages = [
            {"role": "system", "content": PROMPT_CLASSIFY_CITATION_TYPE},
            {"role": "user", "content": user_content.strip()}
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        )
        
    return df.with_columns(pl.Series("prompt", prompts))

    

if __name__=="__main__":
    os.environ["VLLM_USE_V1"] = "0"
    MODEL_PATH = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
    import vllm
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor

    llm = vllm.LLM(
        MODEL_PATH,
        quantization='awq',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=16384,
        disable_log_stats=True, 
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
        task='generate')

    tokenizer = llm.get_tokenizer()

    df=pl.read_csv("/kaggle/working/submission.csv")
    
    if "row_id" in df.columns:
        df = df.drop("row_id")

    # print(df)

    doi_df = df.filter(is_doi_link("dataset_id"))
    acc_df = df.filter(~is_doi_link("dataset_id"))

    # print(doi_df)

    df = find_context_win(tokenizer,doi_df)

    
    
    prompts = df['prompt'].to_list()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B","C"])
    outputs = llm.generate(prompts, vllm.SamplingParams(seed=777, temperature=0.7, skip_special_tokens=True, max_tokens=1, logits_processors=[mclp], logprobs=len(mclp.choices)), use_tqdm=True)
    logprobs = [{lp.decoded_token: lp.logprob for lp in list(lps)} for lps in [output.outputs[0].logprobs[0].values() for output in outputs]]
    choices = [max(d, key=d.get) for d in logprobs]
    types = {'A': True, 'B': False}
    choices = [types[c] for c in choices]
    df = df.with_columns(pl.Series('type', choices))
    df.filter(pl.col('type')).select('article_id', 'dataset_id').write_csv('/tmp/doi_sub.csv')
    df = pl.concat([pl.read_csv('/tmp/doi_sub.csv'), pl.read_csv('/tmp/accid_sub.csv')])
    df = assume_type(df)
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
    # print(df)
    if not IS_KAGGLE_SUBMISSION:
        results = evaluate(df)
        for r in results: l.info(r) 
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results: l.info(r)
    
    
    try:
        del llm, tokenizer
    except:
        pass
    
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
```

```python
%%writefile /tmp/src/predict.py

from helpers import *
import polars as pl
import os


l = get_logger()


PROMPT_CLASSIFY_CITATION_TYPE = '''
# Role & Task
You are an expert data citation analyst. Your task is to classify a given doi/accession ID from a scientific paper into one of two categories based on the context: **A (Primary Data)** or **B (Secondary Data)**.

## Instructions
1.  **Read the provided abstract** to understand the research context.
2.  **Analyze the citation context** for key linguistic cues.
3.  **Classify the citation** as either **A** or **B** based on the definitions below.
4.  **Output only a single letter: A or B.** Do not output any other text, explanation, or formatting.

## Category Definitions

### **Category A: PRIMARY DATA**
The data was generated, collected,submitted,uploaded or created by the **authors of the current study**. This is *their* data.
*   **Key Phrases:** "we collected", "we generated", "submitted","uploaded","our data", "data are available at [URL/DOI]", 
"data have been deposited", "this study presents", "supplementary data","data have been submitted","data have been uploaded",.

### **Category B: SECONDARY DATA**
The data was produced by **other researchers** or external sources and is being reused or analyzed by the current study's authors.
*   **Key Phrases:** "data were obtained from", "publicly available data", "previously published data", "retrieved from", "downloaded from", "[Dataset Name] dataset", "database", citing a specific external source.

## Input Format
You will be provided with the following three pieces of information:
Paper Abstract: {abstract}
Citation: {dataset_id}
Citation Context: {context}


## Decision Framework
Answer these questions based on the **Citation Context**:

1.  **Who is the source of the data?**
    *   If the context implies the **authors themselves** are the source (e.g., "we," "our"), classify as **A**.
    *   If the context names an **external source** (e.g., a repository, another study, a database), classify as **B**.

2.  **What is the action being described?**
    *   **A (Primary)** actions: *depositing, making available, presenting* their own data.
    *   **B (Secondary)** actions: *using, obtaining, accessing, downloading, analyzing* existing data from elsewhere.

## Examples for Pattern Recognition

**Example 1 (Classify as B):**
*   Context: "Three out of four cohorts **used in this study** can be found on The Cancer Imaging Archive (TCIA)24: Canadian benchmark dataset23: https://doi.org/10.7937/K9/TCIA.2017.8oje5q00."
*   **Reasoning:** The authors are describing external datasets they **used** (a Secondary action). The source is TCIA, not themselves.
*   **Output:** B

**Example 2 (Classify as A):**
*   Context: "Additional research data **supporting this publication are available** at 10.25377/sussex.21184705."
*   **Reasoning:** The authors are stating the availability of data that **supports their own publication**. The source is implied to be themselves.
*   **Output:** A

**Example 3 (Classify as B):**
*   Context: "GBIF occurrence data: Vulpes vulpes: https://doi.org/10.15468/dl.wgtneb (28 May 2021)."
*   **Reasoning:** The data is explicitly sourced from an external repository (GBIF). The authors are referring to data they reused.
*   **Output:** B

**Example 4 (Classify as A):**
*   Context: "Data referring to Barbieux et al. (2017; https://doi.org/10.17882/49388) are freely available on SEANOE."
*   **Reasoning:** This is a tricky case. The citation format "(Author et al. Year)" suggests a literature reference. However, the phrase "Data referring to" and the direct data DOI indicate the authors are citing **their own previously published dataset** (from a 2017 paper) that is now available. This is their Primary data.
*   **Output:** A

## Ready for Input
Begin your analysis. Remember: Output only **A** or **B**.

'''

def get_context_window(text: str, substring: str, window: int = 600) -> str:
    idx = text.find(substring)
    if idx == -1:
        return "no context", "no abstraction"
    start = max(idx - window, 0)
    end = min(idx + len(substring) + window, len(text))
    return text[start:end] , text[:1000]

def assume_type1(df):
    """
    Ajoute une colonne 'type' au DataFrame df :
    - SAMN -> toujours Primary
    - Sinon -> on regarde la colonne 'prompt' pour déterminer Primary/Secondary
    """
    def _decide(row):
        ds = str(row["dataset_id"]).lower()
        ctx = str(row.get("prompt", "")).lower()

        # Règle 1 : SAMN
        if ds.startswith("samn"):
            return "Primary"

        # Règle 2 : mots-clés Primary
        if any(word in ctx for word in [
            "sequencing", "rna-seq", "genome", "raw data",
            "bioproject", "experiment", "fastq", "reads", "sample"
        ]):
            return "Primary"

        # Règle 3 : mots-clés Secondary
        if any(word in ctx for word in [
            "supplementary", "analysis", "processed", "meta-analysis",
            "aggregated", "derived", "study"
        ]):
            return "Secondary"

        # Fallback
        return "Secondary"

    df = df.copy()
    df["type"] = df.apply(_decide, axis=1)
    return df



def find_context_win(tokenizer,df):
    text_df = pl.read_parquet('/tmp/context_data.parquet')
    # print(text_df)
    df = df.join(text_df, on=["article_id","dataset_id"], how="inner")
    df = df.drop("type")
    print(df)

    prompts = []
    
    for article_id,dataset_id,text,match in df.select(["article_id","dataset_id","text",'match']).rows():

        context, abstract = get_context_window(text,match)
        user_content = f"""
        Paper Abstract: {abstract}
        
        Citation: {dataset_id}

        
        Citation Context: {context}
        """
        messages = [
            {"role": "system", "content": PROMPT_CLASSIFY_CITATION_TYPE},
            {"role": "user", "content": user_content.strip()}
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        )
        
    return df.with_columns(pl.Series("prompt", prompts))

 def find_context_win1(df):
    text_df = pl.read_parquet('/tmp/context_data.parquet')
    # print(text_df)
    df = df.join(text_df, on=["article_id","dataset_id"], how="inner")
    df = df.drop("type")
    print(df)

    prompts = []
    
    for article_id,dataset_id,text,match in df.select(["article_id","dataset_id","text",'match']).rows():

        context, abstract = get_context_window(text,match)
        user_content = f"""
        Paper Abstract: {abstract}
        
        Citation: {dataset_id}

        
        Citation Context: {context}
        """
        messages = [
            {"role": "sys
        ]
        prompts.append(
            user_content
        )
        
    return df.with_columns(pl.Series("prompt", prompts))   

if __name__=="__main__":
    os.environ["VLLM_USE_V1"] = "0"
    MODEL_PATH = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
    import vllm
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor

    llm = vllm.LLM(
        MODEL_PATH,
        quantization='awq',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=16384,
        disable_log_stats=True, 
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
        task='generate')

    tokenizer = llm.get_tokenizer()

    df=pl.read_csv("/kaggle/working/submission.csv")
    
    if "row_id" in df.columns:
        df = df.drop("row_id")


    doi_df = df.filter(is_doi_link("dataset_id"))
    acc_df = df.filter(~is_doi_link("dataset_id"))
    acc_df = find_context_win1(acc_df)



    df = find_context_win(tokenizer,doi_df)

    
    
    prompts = df['prompt'].to_list()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])
    outputs = llm.generate(prompts, vllm.SamplingParams(seed=777, temperature=0.8, skip_special_tokens=True, max_tokens=1, logits_processors=[mclp], logprobs=len(mclp.choices)), use_tqdm=True)
    logprobs = [{lp.decoded_token: lp.logprob for lp in list(lps)} for lps in [output.outputs[0].logprobs[0].values() for output in outputs]]
    choices = [max(d, key=d.get) for d in logprobs]
    types = {'A':'Primary', 'B':'Secondary'}
    choices = [types[c] for c in choices]


    
    df = df.with_columns(pl.Series('type', choices))
    df.select('article_id', 'dataset_id','type').write_csv('/tmp/doi_sub.csv')

    acc_df = assume_type1(acc_df)
    acc_df.select('article_id','dataset_id','type').write_csv("/tmp/accid_sub.csv")
    df = pl.concat([pl.read_csv('/tmp/doi_sub.csv'), pl.read_csv('/tmp/accid_sub.csv')])
    
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('/kaggle/working/submission.csv')
    # print(df)
    if not IS_KAGGLE_SUBMISSION:
        results = evaluate(df)
        for r in results: l.info(r) 
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results: l.info(r)
    
    
    try:
        del llm, tokenizer
    except:
        pass
    
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
```

```python
%cd /tmp
!LOG_LEVEL=INFO python src/parse.py /tmp/train_parse
! python src/check_parse.py
! python src/getid.py
! python src/llm_validate.py
```

```python
! python src/post_validate.py
```

```python
! python src/predict.py
```

```python
! grep "f1:" /tmp/logs/project.log
```
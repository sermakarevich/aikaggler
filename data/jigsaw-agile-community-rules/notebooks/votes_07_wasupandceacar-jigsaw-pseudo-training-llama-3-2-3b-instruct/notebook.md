# jigsaw_pseudo_training_llama-3.2-3b-instruct

- **Author:** wasupandceacar
- **Votes:** 382
- **Ref:** wasupandceacar/jigsaw-pseudo-training-llama-3-2-3b-instruct
- **URL:** https://www.kaggle.com/code/wasupandceacar/jigsaw-pseudo-training-llama-3-2-3b-instruct
- **Last run:** 2025-10-12 16:58:52.783000

---

Base idea from [https://www.kaggle.com/code/barnobarno/test-time-training-tt](https://www.kaggle.com/code/barnobarno/test-time-training-tt)

25/10/13 Update: remove all private resources

Changes:

1. Base model is llama-3.2-3b-instruct
2. Use 5% of hidden test data for pseudo training
3. r=64, lora_alpha=128
4. Use bf16 in deepspeed
5. Add bnb config for non gptq model

```python
!uv pip install --system --no-index --find-links='/kaggle/input/jigsaw-packages2/whls/' 'trl==0.21.0' 'optimum==1.27.0' 'auto-gptq==0.7.1' 'bitsandbytes==0.46.1' 'logits-processor-zoo==0.2.1' 'vllm==0.10.0'
!uv pip install --system --no-index --find-links='/kaggle/input/jigsaw-packages2/whls/' 'deepspeed==0.17.4' -q
!uv pip install --system --no-index --find-links='/kaggle/input/jigsaw-packages2/whls/' 'triton==3.2.0'
!uv pip install --system --no-index --find-links='/kaggle/input/jigsaw-packages2/whls/' 'clean-text'
!uv pip install --system --no-index -U --no-deps --find-links='/kaggle/input/jigsaw-packages2/whls/' 'peft' 'accelerate' 'datasets'
```

# Pseudo training

```python
%%writefile constants.py

seed = 0

base_model_path = "/kaggle/input/jigsaw-pretrain-public/pytorch/llama-3.2-3b-instruct/1"
pretrain_lora_path = None
lora_path = "/kaggle/working/pseudo_lora"
use_gptq = "gptq" in base_model_path

positive = "Yes"
negative = "No"
judge_words = "Violation:"
system_prompt = '''You are given a comment from reddit and a rule. 
Your task is to classify whether the comment violates the rule. 
Only respond Yes/No.'''

frac = 0.05
use_train = True

import kagglehub

deterministic = kagglehub.package_import('wasupandceacar/deterministic').deterministic
deterministic.init_all(seed)
```

```python
%%writefile utils.py

import numpy as np
import pandas as pd
from datasets import Dataset
from constants import *

def build_prompt(row):
    return f"""{system_prompt}
Subreddit: r/{row["subreddit"]}
Rule: {row["rule"]}
Examples:
1) {row["positive_example"]}
{judge_words} Yes
2) {row["negative_example"]}
{judge_words} No
Comment: {row["body"]}
{judge_words}"""

def get_df():
    merge = list()
    if use_train:
        train_dataset = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
        train_df = train_dataset[["body", "rule", "subreddit", "rule_violation",
                                "positive_example_1", "positive_example_2", 
                                "negative_example_1", "negative_example_2"]].copy()
        train_df["positive_example"] = np.where(np.random.rand(len(train_df)) < 0.5, train_df["positive_example_1"], train_df["positive_example_2"])
        train_df["negative_example"] = np.where(np.random.rand(len(train_df)) < 0.5, train_df["negative_example_1"], train_df["negative_example_2"])
        train_df.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"], inplace=True)
        merge.append(train_df)
    test_dataset = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    test_dataset = test_dataset.groupby('rule', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=seed)).reset_index(drop=True)
    print(f"Select {len(test_dataset)} test data")
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[["rule", "subreddit", "positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]].copy()
            body_col = f"{violation_type}_example_{i}"
            other_positive_col = f"{violation_type}_example_{3-i}"
            sub_dataset["body"] = sub_dataset[body_col]
            sub_dataset[f"{violation_type}_example"] = sub_dataset[other_positive_col]
            anti_violation_type = "negative" if violation_type == "positive" else "positive"
            sub_dataset[f"{anti_violation_type}_example"] = np.where(np.random.rand(len(sub_dataset)) < 0.5, sub_dataset[f"{anti_violation_type}_example_1"], sub_dataset[f"{anti_violation_type}_example_2"])
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"], inplace=True)
            merge.append(sub_dataset)
    return pd.concat(merge, axis=0).drop_duplicates(ignore_index=True)

def build_dataset(df):
    df["prompt"] = df.apply(build_prompt, axis=1)
    columns = ["prompt"]
    if "rule_violation" in df:
        df["completion"] = df["rule_violation"].map({
            1: positive,
            0: negative,})
        columns.append("completion")
    dataset = Dataset.from_pandas(df[columns])
    return dataset
```

```python
%%writefile train.py

import torch
import pandas as pd
from trl import SFTTrainer, SFTConfig
from peft import PeftModel, LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_torch_bf16_gpu_available

from utils import *
from constants import *

def main():
    train_dataset = build_dataset(get_df())
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    training_args = SFTConfig(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=is_torch_bf16_gpu_available(),
        fp16=not is_torch_bf16_gpu_available(),
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
        packing=False,
        remove_unused_columns=False,
    )

    if use_gptq:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="balanced_low_0",
            trust_remote_code=True,
            use_cache=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,     
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="balanced_low_0",
            trust_remote_code=True,
            use_cache=False,
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if pretrain_lora_path:
        model = PeftModel.from_pretrained(model, pretrain_lora_path)
        model = model.merge_and_unload()

    if len(train_dataset) > 0:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
        )
        trainer.train()
        trainer.save_model(lora_path)
    else:
        peft_model = get_peft_model(model, lora_config)
        peft_model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)

if __name__ == "__main__":
    main()
```

```python
%%writefile accelerate_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0
  train_micro_batch_size_per_gpu: 4
  
  zero_stage: 2
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  
  stage3_gather_16bit_weights_on_model_save: false
  stage3_max_live_parameters: 1e8
  stage3_max_reuse_distance: 1e8
  stage3_prefetch_bucket_size: 5e7
  stage3_param_persistence_threshold: 1e5
  
  zero_allow_untested_optimizer: true
  zero_force_ds_cpu_optimizer: false
  
  # fp16:
  #   enabled: true
  #   loss_scale: 0
  #   initial_scale_power: 16
  #   loss_scale_window: 1000
  #   hysteresis: 2
  #   min_loss_scale: 1
  bf16:
    enabled: true
  
distributed_type: DEEPSPEED
downcast_bf16: 'yes'
dynamo_config:
  dynamo_backend: INDUCTOR
  dynamo_use_fullgraph: false
  dynamo_use_dynamic: false
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

```python
%%writefile inference.py

import os
os.environ["VLLM_USE_V1"] = "0"

import random
import vllm
import torch
import numpy as np
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from vllm.lora.request import LoRARequest
from utils import build_dataset
from constants import *
import multiprocessing as mp

def run_inference_on_device(df_slice):
    llm = vllm.LLM(
        base_model_path,
        quantization="gptq" if use_gptq else None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=2048,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=64,
    )
    tokenizer = llm.get_tokenizer()
    outputs = llm.generate(
        build_dataset(df_slice)["prompt"],
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=[positive, negative])],
            logprobs=2,
        ),
        use_tqdm=True,
        lora_request=LoRARequest("lora1", 1, lora_path)
    )
    log_probs = [{lp.decoded_token: np.exp(lp.logprob) for lp in out.outputs[0].logprobs[0].values()} for out in outputs]
    predictions = pd.DataFrame(log_probs)[[positive, negative]]
    predictions["row_id"] = df_slice["row_id"].values
    return predictions

def worker(device_id, df_slice, return_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"[Worker {device_id}] Running on GPU {device_id}, data size={len(df_slice)}")
    preds = run_inference_on_device(df_slice)
    return_dict[device_id] = preds

def main():
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    test_df["positive_example"] = test_df.apply(lambda row: random.choice([row["positive_example_1"], row["positive_example_2"]]), axis=1)
    test_df["negative_example"] = test_df.apply(lambda row: random.choice([row["negative_example_1"], row["negative_example_2"]]), axis=1)
    test_df = test_df.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"], errors="ignore")

    mid = len(test_df) // 2
    df0 = test_df.iloc[:mid].reset_index(drop=True)
    df1 = test_df.iloc[mid:].reset_index(drop=True)

    manager = mp.Manager()
    return_dict = manager.dict()
    p0 = mp.Process(target=worker, args=(0, df0, return_dict))
    p1 = mp.Process(target=worker, args=(1, df1, return_dict))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    predictions = pd.concat([return_dict[0], return_dict[1]], ignore_index=True)
    submission = predictions[["row_id", positive]].rename(columns={positive: "rule_violation"})
    submission.to_csv("/kaggle/working/submission.csv", index=False)

if __name__ == "__main__":
    main()
```

```python
!accelerate launch --config_file accelerate_config.yaml train.py
!python inference.py

import pandas as pd
pd.read_csv('/kaggle/working/submission.csv')
```
# Starter NB: Fine Tune MedGemma 1.5 for Detection

- **Author:** Harpreet Sahota
- **Votes:** 45
- **Ref:** harpdeci/starter-nb-fine-tune-medgemma-1-5-for-detection
- **URL:** https://www.kaggle.com/code/harpdeci/starter-nb-fine-tune-medgemma-1-5-for-detection
- **Last run:** 2026-01-19 22:07:07.900000

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harpreetsahota204/medgemma_kaggle_competition/blob/main/medgemma_impact_starter.ipynb)

# Winning the MedGemma Impact Challenge with FiftyOne

**The difference between a demo and a winning submission is understanding where your model breaks—and why.**

This notebook shows you how to use [FiftyOne](https://docs.voxel51.com/) as your workbench for the MedGemma Impact Challenge. 
We'll go beyond running inference and printing metrics. You'll learn to:

1. **Explore your data** before modeling
2. **Visualize embeddings** to diagnose learnability  
3. **Run MedGemma inference** and store predictions alongside ground truth
4. **Analyze failures** systematically—not just count them
5. **Fine-tune for localization** using FiftyOne's PyTorch integration

We'll use the [SLAKE dataset](https://huggingface.co/datasets/Voxel51/SLAKE), a medical VQA benchmark 
with images from multiple modalities (CT, MRI, X-ray), rich annotations including bounding boxes and 
segmentation masks, and questions spanning anatomy, abnormalities, and more.

---

## Setup & Installation

```python
!pip install -U fiftyone huggingface_hub accelerate sentencepiece protobuf torch torchvision umap-learn
```

### Authenticate with Hugging Face

Both MedGemma and MedSigLIP are gated models. You'll need to:
1. Request access on [MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) and [MedSigLIP](https://huggingface.co/google/medsiglip-448)
2. Set your HF token

```python
import os
# os.environ["HF_TOKEN"] = "your_token_here"

# Or login via CLI: hf auth login

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #update for your setup
```

---
## 1. Load the SLAKE Dataset

The SLAKE dataset is already in [FiftyOne format](https://docs.voxel51.com/user_guide/using_datasets.html) on Hugging Face. 
One line to load it using the [`load_from_hub()`](https://docs.voxel51.com/api/fiftyone.utils.huggingface.html#fiftyone.utils.huggingface.load_from_hub) function.

```python
from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub(
    "Voxel51/SLAKE",
    name="SLAKE",
    overwrite=True,
    max_samples=50 #taking a small subset of the dataset for this example
)
```

### Understanding FiftyOne Datasets

A FiftyOne [Dataset](https://docs.voxel51.com/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset) 
is comprised of [Samples](https://docs.voxel51.com/api/fiftyone.core.sample.html#fiftyone.core.sample.Sample).

**Samples** store all information associated with a particular piece of data in a dataset, including:
- Basic metadata about the data
- One or more sets of labels
- Additional features associated with subsets of the data and/or label sets

The attributes of a Sample are called [Fields](https://docs.voxel51.com/api/fiftyone.core.fields.html#fiftyone.core.fields.Field), 
which store information about the Sample. When a new Field is assigned to a Sample in a Dataset, 
it is automatically added to the dataset's schema and thus accessible on all other samples in the dataset.

Let's look at the schema to understand what we're working with:

```python
dataset
```

To see the contents of a single Sample and its Fields, you can use the 
[`first()` method](https://docs.voxel51.com/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.first):

```python
dataset.first()
```

### Understanding the SLAKE Schema

This dataset is **image-centric**: each of the 642 samples represents one medical image,
with multiple Q&A pairs attached to it. Let's break down the key fields:

**Metadata fields** (stored as `Classification` objects—access via `.label`):
- `modality`: Imaging modality (CT, MRI, X-Ray) 
- `location`: Anatomical region (Lung, Brain, Abdomen, etc.)
- `answer_type`: Question type (OPEN or CLOSED)
- `base_type`: Task type (vqa)

**Multiple Q&A pairs** (up to 20 per image):
- `question_0`, `question_1`, ... `question_19`: Question strings
- `answer_0`, `answer_1`, ... `answer_19`: Answer as `Classification` objects

**Annotations** (where available):
- `detections`: Bounding boxes with labels (e.g., "Cardiomegaly")
- `segmentation`: Segmentation masks with `mask_path`

### Accessing Classification Fields

Many fields in this dataset are stored as FiftyOne 
[`Classification`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Classification) 
objects. To get the actual value, access the `.label` attribute:

```python
sample = dataset.first()

# These are Classification objects - access .label to get the string value
print(f"Modality: {sample.modality.label}")
print(f"Location: {sample.location.label}")
print(f"Answer Type: {sample.answer_type.label}")

# Questions are stored as strings
print(f"\nQuestion 0: {sample.question_0}")

# Answers are Classification objects
print(f"Answer 0: {sample.answer_0.label}")
```

### Slicing Field Values with `ViewField`

**Key Concept:** Methods like `count_values("modality.label")` work because they accept 
**field paths as strings** (using dot notation). However, **slicing/indexing requires 
`ViewField` expressions**.

**String field paths** (dot notation) work for:
- `count_values("modality.label")`
- `distinct("modality.label")`
- `sort_by("modality.label")`

**`ViewField` expressions** are required for:
- Array indexing: `F("bounding_box")[2]`
- Array slicing: `F("detections")[1:3]`
- String slicing: `F("text_field")[:10]`

```python
from fiftyone import ViewField as F

# ❌ This won't work (can't slice string paths)
dataset.count_values("predictions.detections[0].label")

# ✅ Use ViewField for slicing
expr = F("predictions.detections")[0].label
dataset.count_values(expr)

# ✅ Other examples
bbox_width = F("bounding_box")[2]
first_three = F("detections")[:3]
```

**Summary:**
- **Dot notation strings** = simple field paths
- **`F(...)` expressions** = when you need indexing/slicing operations on field values

### Exploring Q&A Pairs

Each image has multiple question-answer pairs. Let's look at a single sample:

```python
sample = dataset.first()

# Print Q&A pairs for this sample
print(f"Sample Q&A pairs:\n")
for i in range(7):  # First 7 questions (most samples have ~7)
    q = getattr(sample, f"question_{i}")
    a = getattr(sample, f"answer_{i}")
    if q is not None:
        print(f"Q{i}: {q}")
        print(f"A{i}: {a.label if a else 'None'}")
        print()
```

---
## 2. Explore Your Data (Before You Model)

Don't rush to inference. Understanding your data distribution is how you catch problems early.

FiftyOne provides powerful functionality to compute statistics about your dataset using 
[built-in Aggregation methods](https://docs.voxel51.com/user_guide/using_aggregations.html).

### What modalities do we have?

Use the [`count_values()` aggregation](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.count_values) 
to compute the occurrences of field values in a collection.

**Important:** Since `modality` is a Classification field, we need to access 
the `.label` attribute using dot notation in the field path:

```python
dataset.count_values("modality.label")
```

### What anatomical locations are covered?

The `location` field tells us what body part/organ the image focuses on:

```python
dataset.count_values("location.label")
```

### What types of questions?

The `answer_type` field indicates whether questions are OPEN (free-form) or CLOSED (yes/no, multiple choice):

```python
dataset.count_values("answer_type.label")
```

### What detection labels exist?

The `detections` field contains bounding boxes with labels (e.g., anatomical structures, 
abnormalities). Use [`count_values()`](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.count_values) 
on nested fields:

```python
dataset.count_values("detections.detections.label")
```

### Launch the App to explore visually

The most powerful part of FiftyOne is [the FiftyOne App](https://docs.voxel51.com/user_guide/app.html#using-the-fiftyone-app), 
which runs locally on your machine. Filter, sort, and browse your data interactively.

```python
import fiftyone as fo
session = fo.launch_app(dataset)
```

# ![Explore MedGemma](https://raw.githubusercontent.com/harpreetsahota204/medgemma_kaggle_competition/main/gifs/explore_med_gemma.gif)

**Try these in the App:**
- In sidebar of the app, under the Labels section, click the dropdown for `modality` and click the check box for CT to filter the samples in the panel to only CT scans
- Try the same for the to `location` label, for example filter to "Lung"` to see lung images
- Look at samples with detections (bounding boxes) vs without
- Explore the Q&A pairs in the sample panel

You'll start to notice patterns: certain anatomical locations have more images, 
certain modalities are over/under-represented, etc.

### Create useful Dataset Views

[Dataset Views](https://docs.voxel51.com/user_guide/using_views.html) let you filter, sort, and 
slice your data without modifying the underlying dataset. Views are powerful because they:
- Chain multiple operations together
- Are lazily evaluated for efficiency
- Can be saved and reloaded

You can use [`ViewField`](https://docs.voxel51.com/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewField) 
and [`ViewExpression`](https://docs.voxel51.com/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewExpression) 
classes to define expressions using native Python operators. Simply wrap the target field in a 
`ViewField` and apply comparison, logic, arithmetic or array operations to it.

Learn more about [creating Views](https://docs.voxel51.com/cheat_sheets/views_cheat_sheet.html) 
and [filtering](https://docs.voxel51.com/cheat_sheets/filtering_cheat_sheet.html) in the cheat sheets.

```python
from fiftyone import ViewField as F
# CLOSED answer type only (yes/no questions - easier to evaluate)
# Note: Use "answer_type.label" to filter on the Classification's label
closed_questions = dataset.match(F("answer_type.label") == "CLOSED")
dataset.save_view("closed_questions", closed_questions)
print(f"Images with CLOSED questions: {len(closed_questions)}")

# Images with detection annotations (bounding boxes)
has_detections = dataset.match(F("detections.detections").length() > 0)
dataset.save_view("has_detections", has_detections)
print(f"Images with detections: {len(has_detections)}")

# X-Ray images only
xray_only = dataset.match(F("modality.label") == "X-Ray")
dataset.save_view("xray_only", xray_only)
print(f"X-Ray images: {len(xray_only)}")

# CT images only
ct_only = dataset.match(F("modality.label") == "CT")
dataset.save_view("ct_only", ct_only)
print(f"CT images: {len(ct_only)}")

# Lung images
lung_images = dataset.match(F("location.label") == "Lung")
dataset.save_view("lung_images", lung_images)
print(f"Lung images: {len(lung_images)}")
```

For those familiar with `pandas`, check out this 
[pandas vs FiftyOne cheat sheet](https://docs.voxel51.com/cheat_sheets/pandas_vs_fiftyone.html) 
to learn how to translate common pandas operations into FiftyOne syntax.

---
## 3. Compute Embeddings with MedSigLIP

Before running VQA inference, let's see if the embedding space even separates our classes.
If MedSigLIP embeddings don't cluster by modality or body part, that's diagnostic information.

You can visualize [image embeddings](https://docs.voxel51.com/brain.html#visualizing-embeddings) 
using models from the [FiftyOne Model Zoo](https://docs.voxel51.com/model_zoo/overview.html), 
or custom models which you can integrate as a [Remote Zoo Model](https://docs.voxel51.com/model_zoo/remote.html#remotely-sourced-zoo-models).

### Register and load MedSigLIP

MedSigLIP is available as a Remote Zoo Model. First, register the model source:

```python
import fiftyone.zoo as foz
# Register the model source (one time)
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/medsiglip",
    overwrite=True
)

# Download the model (one time)
foz.download_zoo_model(
    "https://github.com/harpreetsahota204/medsiglip",
    model_name="google/medsiglip-448",
)
```

```python
# Load the model
medsiglip = foz.load_zoo_model("google/medsiglip-448")
```

### Compute embeddings

Use the [`compute_embeddings()` method](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.compute_embeddings) 
to compute embeddings for all samples in your dataset:

```python
dataset.compute_embeddings(
    model=medsiglip,
    embeddings_field="medsiglip_embeddings",
)
```

### Visualize in 2D

Use the [`compute_visualization()` method](https://docs.voxel51.com/api/fiftyone.brain.html#fiftyone.brain.compute_visualization) 
to generate low-dimensional representations of the samples in your Dataset. 
This projects high-dimensional embeddings to 2D/3D for visualization.

```python
import fiftyone.brain as fob

results = fob.compute_visualization(
    dataset,
    embeddings="medsiglip_embeddings",
    method="umap",
    brain_key="medsiglip_viz",
    num_dims=2,
)
```

### Build a similarity index for later

Use the [`compute_similarity()` method](https://docs.voxel51.com/api/fiftyone.brain.html#fiftyone.brain.compute_similarity) 
to build a similarity index over the images in your dataset. This allows you to 
[sort by similarity](https://docs.voxel51.com/brain.html#sorting-by-similarity) or 
[search with natural language](https://docs.voxel51.com/brain.html#text-similarity) (for models that support this, such as CLIP, SigLIP, or MedSigLIP).

```python
sim_index = fob.compute_similarity(
    dataset,
    model="google/medsiglip-448",
    brain_key="medsiglip_similarity",
    embeddings="medsiglip_embeddings"
)
```

With embeddings computed, you can perform non-trivial analysis like computing scores for 
[uniqueness](https://docs.voxel51.com/brain.html#image-uniqueness), 
[representativeness](https://docs.voxel51.com/brain.html#image-representativeness), 
and [identifying near duplicates](https://docs.voxel51.com/brain.html#near-duplicates) 
with simple function calls.

- Near-duplicates: Redundant images that inflate dataset size without adding value

- Uniqueness: How distinct each sample is from others (low = redundant, high = informative)

- Representativeness: How well a sample represents the overall distribution (high = typical, low = outlier)

As an example, let's compute uniqueness.

In a nutshell, uniqueness measures how far a sample is from its nearest neighbors in embedding space, with higher values indicating the sample is more isolated/distinct from other samples in the dataset.

It's computed by finding each sample's K nearest neighbors (K=3), calculating a weighted average of the distances to those neighbors, and normalizing the result to a 0-1 scale.

```python
# Compute uniqueness scores
fob.compute_uniqueness(
    dataset,
    embeddings="radio_embeddings",
    similarity_index=sim_index
    )
```

**In the App:**
- Open the [Embeddings panel](https://docs.voxel51.com/user_guide/app.html#embeddings-panel)
- Color by `modality` — do CT, MRI, X-ray form distinct clusters?
- Color by `body_part` — do anatomical regions separate?
- Color by `content_type` — do question types cluster?

**What you're looking for:**
- Clear separation = model has a chance
- Everything mixed together = fundamental representation problem

```python
# Relaunch app to see embeddings panel
import fiftyone as fo
session = fo.launch_app(dataset)
```

![Explore MedGemma Embeddings](https://raw.githubusercontent.com/harpreetsahota204/medgemma_kaggle_competition/main/gifs/explore_medgemma_embeddings.gif)

---
## 4. Run MedGemma Inference

Now let's run MedGemma 1.5 on the VQA task and store predictions.

FiftyOne is open-source and hackable, with a robust framework for 
[building Plugins](https://docs.voxel51.com/plugins/developing_plugins.html) that extend 
and customize the tool. Browse this [curated collection of plugins](https://docs.voxel51.com/plugins/) 
to see integrations with various computer vision models and AI tools.

### Register and load MedGemma

```python
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/medgemma_1_5",
    overwrite=True
)

foz.download_zoo_model(
    "https://github.com/harpreetsahota204/medgemma_1_5",
    model_name="google/medgemma-1.5-4b-it",
)
```

```python
medgemma = foz.load_zoo_model("google/medgemma-1.5-4b-it")
```

### Configure for VQA

```python
# Set operation mode
medgemma.operation = "classify"

# Set a custom system prompt
medgemma.system_prompt = """You are an expert radiologist, histopathologist, ophthalmologist, and dermatologist.

Your expert opinion is needed for answering questions about medical images.

Report your answer as JSON array in this format: 

```json
{
    "classifications": [
        {
            "label": "descriptive medical condition or relevant label"
            ...,
        }
    ]
}
```

Always return your response as valid JSON wrapped in ```json blocks.  You must produce only a single word answer. Do not report your confidence.
"""
```

### Running Inference on Multi-Question Samples

Since each image has multiple Q&A pairs (`question_0`/`answer_0` through `question_19`/`answer_19`),
we have a few options for running inference:

1. **Pick one question per image** (simplest) - use `prompt_field="question_0"`
2. **Run on all questions** - loop through question fields
3. **Flatten the dataset** - create a new sample per Q&A pair

Let's start simple by running on the first question of each image:

```python
dataset.apply_model(
    medgemma,
    label_field="pred_answer_0",
    prompt_field="question_0",  # Use the first question for each image
    batch_size=32,
    num_workers=4,
)
```

### Inspect predictions

```python
dataset.first()['pred_answer_0']
```

```python
# Look at a few samples
for sample in dataset.take(5):
    print(f"Image: {sample.filepath.split('/')[-1]}")
    print(f"Modality: {sample.modality.label}")
    print(f"Q: {sample.question_0}")
    print(f"GT: {sample.answer_0.label if sample.answer_0 else 'None'}")
    print(f"Pred: {sample.pred_answer_0.classifications[0].label if sample.pred_answer_0 else 'None'}")
    print("-" * 50)
```

---
## 5. Evaluate Performance

Let's compute accuracy—but more importantly, let's slice it to find patterns.

FiftyOne provides [evaluation methods](https://docs.voxel51.com/user_guide/evaluation.html) 
for various task types including [detection](https://docs.voxel51.com/user_guide/evaluation.html#detections), [classification](https://docs.voxel51.com/user_guide/evaluation.html#classifications), and [segmentation](https://docs.voxel51.com/user_guide/evaluation.html#semantic-segmentations).

##### We need to make a conversion from Classifications → ⁠Classification


The implementation of MedGemma outputs a FiftyOne *Classifications* object (notice it's plural), but to run the evaluation for classification we need a FiftyOne *Classification* (singluar)

FiftyOne's `evaluate_classifications()` only works with **single-label** classification fields (`Classification`), not multilabel containers (`Classifications`).

**What you need to do:**

1. Choose one label per sample (e.g., first label, highest confidence)
2. Store it as a `Classification` field
3. Pass that field to `evaluate_classifications()`

**Read more in the docs:**

- [Classification evaluation overview](https://docs.voxel51.com/user_guide/evaluation.html#classifications)
- [Simple evaluation example](https://docs.voxel51.com/user_guide/evaluation.html#id4)
- [Binary evaluation example](https://docs.voxel51.com/user_guide/evaluation.html#binary-evaluation)
- [Classification evaluation tutorial](https://docs.voxel51.com/tutorials/evaluate_classifications.html#Evaluating-model-with-FiftyOne)

```python
import fiftyone as fo

# assume dataset has a multilabel field "multi" of type fo.Classifications
# and we want a single-label field "single" of type fo.Classification

for sample in dataset:
    cls_list = sample["pred_answer_0"].classifications if sample["pred_answer_0"] is not None else []

    if cls_list:
        # choose one classification; here we take the first
        chosen = cls_list[0]
        sample["pred_answer_0_as_cls"] = fo.Classification(
            label=chosen.label,
        )
    else:
        sample["pred_answer_0_as_cls"] = None

    sample.save()
```

```python
# Evaluate the predictions in the `predictions` field with respect to the
# labels in the `ground_truth` field
classify_results = dataset.evaluate_classifications(
    "pred_answer_0_as_cls",
    gt_field="answer_0",
    eval_key="eval_ans_0",
)
```

```python
# Print a classification report
classify_results.print_report()
```

You can also open the evaluation panel in the app for a more interactive evaluation experience.

You can use [Scenario Analysis](https://docs.voxel51.com/user_guide/app.html#scenario-analysis) 
for a deep dive into model behavior across different scenarios. This helps uncover edge cases, 
identify annotation errors, and understand performance variations in different contexts.

![Eval MedGemma Classifications](https://raw.githubusercontent.com/harpreetsahota204/medgemma_kaggle_competition/main/gifs/medgemma_eval.gif)


### Visual Question Answering

You can also use MedGemma for visual question answering to get a more open-ended answer:

```python
medgemma.operation="vqa" #change operation

medgemma.system_prompt = None #reset system prompt, use default system prompt for vqa

print(medgemma.system_prompt) #print the default vqa system prompt
```

```python
dataset.apply_model(
    medgemma,
    label_field="free_text_answer_0",
    prompt_field="question_0",  # Use the first question for each image
    batch_size=32,
    num_workers=4,
)
```

```python
dataset.first()['free_text_answer_0']
```

### Running on All Questions (Optional)

If you want to evaluate on all Q&A pairs, you can loop through the question fields.
This stores predictions for each question in separate fields:

```python
# # Run inference on all questions (takes longer)
# for i in range(20):  # Up to 20 questions per image
#     q_field = f"question_{i}"
#     pred_field = f"free_text_answer_{i}"
#     
#     # Only run if this question exists in any sample
#     if dataset.count(q_field) > 0:
#         print(f"Running inference on {q_field}...")
#         dataset.apply_model(
#             medgemma,
#             label_field=pred_field,
#             prompt_field=q_field,
#             batch_size=32,
#             num_workers=4,
#         )
```

### Add correctness field


Since MedGemma produces verbose answers in VQA mode, we use Gemma 3 270m as a semantic judge to determine if the predicted answer is correct rather than relying on exact string matching.

Use [`values()`](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.values) 
to efficiently extract field values across all samples, and 
[`set_values()`](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.set_values) 
to add computed fields back to the dataset.

```python
from transformers import pipeline
from tqdm import tqdm
import torch

# Load LLM judge
judge = pipeline(
    "text-generation",
    model="google/gemma-3-270m-it",
    device="cuda",
    dtype="auto"
)

# Get data
gt_values = dataset.values("free_text_answer_0")
pred_values = dataset.values("pred_answer_0")
questions = dataset.values("question_0")

SYSTEM_PROMPT = """You are an expert medical evaluator. Your task is to determine if a predicted answer correctly answers a question, given the ground truth answer. The predicted answer may be more verbose or phrased differently, but should be semantically equivalent to the ground truth.

Respond with ONLY "CORRECT" or "INCORRECT" - no other text."""

def is_correct(question, gt, pred):
    if not pred or not gt:
        return False
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": f"""Question: {question}
Ground Truth Answer: {gt}
Predicted Answer: {pred}

Is the Predicted Answer CORRECT or INCORRECT?"""}]}
    ]
    
    output = judge(messages, max_new_tokens=16, do_sample=False)
    return "CORRECT" in output[0]["generated_text"][-1]["content"].upper()

# Evaluate and save
results = [is_correct(q, gt, p) for q, gt, p in tqdm(zip(questions, gt_values, pred_values), total=len(questions))]
dataset.set_values("is_correct_0", results)
dataset.save()

print(f"{sum(results)}/{len(results)} answers judged as correct")
```

### Overall accuracy (on question_0, LLM-judged)

```python
# Clean up judge pipeline to free GPU memory for subsequent operations
from fiftyone import ViewField as F
del judge
import gc
gc.collect()
torch.cuda.empty_cache()

# Calculate overall accuracy using LLM-judged correctness
correct = dataset.match(F("is_correct_0") == True)
total = len(dataset)
accuracy = len(correct) / total

print(f"Overall Accuracy (Q0, LLM-judged): {accuracy:.2%} ({len(correct)}/{total})")
```

### Accuracy by answer type

CLOSED questions (yes/no) should be easier than OPEN (free-form) ones.

```python
print("Accuracy by Answer Type:")
for atype in dataset.distinct("answer_type.label"):
    view = dataset.match(F("answer_type.label") == atype)
    correct_view = view.match(F("is_correct_0") == True)
    acc = len(correct_view) / len(view) if len(view) > 0 else 0
    print(f"  {atype}: {acc:.2%} ({len(correct_view)}/{len(view)})")
```

### Accuracy by modality

Does MedGemma perform differently on CT vs MRI vs X-Ray?

```python
print("\nAccuracy by Modality:")
for modality in dataset.distinct("modality.label"):
    view = dataset.match(F("modality.label") == modality)
    correct_view = view.match(F("is_correct_0") == True)
    acc = len(correct_view) / len(view) if len(view) > 0 else 0
    print(f"  {modality}: {acc:.2%} ({len(correct_view)}/{len(view)})")
```

### Accuracy by anatomical location

```python
print("\nAccuracy by Location:")
results = []
for location in dataset.distinct("location.label"):
    view = dataset.match(F("location.label") == location)
    correct_view = view.match(F("is_correct_0") == True)
    acc = len(correct_view) / len(view) if len(view) > 0 else 0
    results.append((location, acc, len(view)))

# Sort by accuracy
for location, acc, n in sorted(results, key=lambda x: x[1]):
    print(f"  {location}: {acc:.2%} (n={n})")
```

**This is where it gets interesting.** 

You might find things like:
- "MedGemma struggles on Brain MRI images"  
- "Abnormality detection is worse on Abdomen CT than Lung X-Ray"
- "OPEN questions have much lower accuracy than CLOSED questions"

These are *actionable insights*, not just numbers.

## Detection with MedGemma

You can use MedGemma to localize anatomical structures and pathologies in medical images. The model outputs bounding boxes in FiftyOne's Detections format.

```python
# Set detection mode
medgemma.operation = "detect"

# Get labels to detect (e.g., from ground truth)
labels = dataset.distinct("detections.detections.label")
labels_str = ", ".join(labels)

# Prompt for localization
medgemma.prompt = f"""Locate the following in this scan: {labels_str}. 
Output the final answer in the format "Final Answer: X" where X is a JSON list of objects. 
The object needs a "box_2d" and "label" key. 
If the object is not present in the scan, skip it and don't output anything for that object.
Answer:"""

# Apply detection
dataset.apply_model(
    medgemma,
    label_field="pred_detection",
    batch_size=32,
    num_workers=4,
)
```

We can then use [FiftyOne's evaluation API](https://docs.voxel51.com/user_guide/evaluation.html) to see how well the initial results. You can [`evaluate_detections()` method](https://docs.voxel51.com/user_guide/evaluation.html#detections) to evaluate the predictions of an object detection model stored in a [`Detections`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Detections), [`Polylines`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Polylines), or [`Keypoints`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Keypoints) field of your dataset or of a temporal detection model stored in a [`TemporalDetections`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.TemporalDetection) field of your dataset.

```python
results = dataset.evaluate_detections(
    "pred_detection",        
    gt_field="detections",  
    eval_key="initial_detection_eval",
    tolerance=2
)
```

The `evaluate_detections()` method returns a [`DetectionResults` instance](https://docs.voxel51.com/api/fiftyone.utils.eval.detection.html#fiftyone.utils.eval.detection.DetectionResults) that provides a variety of methods for generating various aggregate evaluation reports about your model.

In addition, when you specify an `eval_key` parameter, a number of helpful fields will be populated on each sample and its predicted/ground truth objects that you can leverage via the FiftyOne App to interactively explore the strengths and weaknesses of your model on individual samples.

You can print the report to get a high-level picture of the model performance:

```python
results.print_report()
```

You can inspect the quality of the detections also use the model evaluation panel in the app:


![Eval MedGemma Classifications](https://raw.githubusercontent.com/harpreetsahota204/medgemma_kaggle_competition/main/gifs/medgemma_eval_detections.gif)



The results look...not great.

But, this means we have a starting point. Now that we know the model can predict bounding boxes we can fine-tune it on our dataset!

If you're running this notebook end to end, then it's a good idea to clear up some GPU memory:

```python
del medgemma
del medsiglip

import gc
gc.collect()
torch.cuda.empty_cache()
```

---
## 7. Fine-Tuning MedGemma for Localization

You've explored the data, identified failure patterns, and have hypotheses about what to fix.
Now let's fine-tune MedGemma to output bounding box coordinates for localization tasks.

This section demonstrates converting datasets to PyTorch format for training.

We'll follow these steps:
1. Define a [`GetItem`](https://docs.voxel51.com/api/fiftyone.utils.torch.html#fiftyone.utils.torch.GetItem) subclass to extract and transform data from FiftyOne
2. Create train/val splits and flatten detections using [`to_patches()`](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.to_patches)
3. Convert to PyTorch datasets using [`to_torch()`](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.to_torch)
4. Set up QLoRA fine-tuning with the TRL library's `SFTTrainer`

### Install fine-tuning dependencies

```python
!pip install --upgrade --quiet bitsandbytes peft trl
```

### Step 1: Define the GetItem subclass

FiftyOne's [`GetItem`](https://docs.voxel51.com/api/fiftyone.utils.torch.html#fiftyone.utils.torch.GetItem) 
class is the bridge between FiftyOne and PyTorch. It tells FiftyOne:

1. **What fields to extract** from each sample (via `required_keys`)
2. **How to transform them** into your desired format (via `__call__`)

The `field_mapping` parameter is important when working with patches. In a patches view,
the detection data lives in the original field name (e.g., "detections"), but we want 
to access it with a generic name in our code.

`field_mapping={"detection": "detections"}` means:
- In our code, we write `d["detection"]`
- FiftyOne knows to pull from the "detections" field

This makes our `GetItem` reusable across datasets with different field names.

```python
from typing import Any
from PIL import Image
from fiftyone.utils.torch import GetItem

# System prompt for localization task
LOCALIZATION_SYSTEM_PROMPT = """Instructions:
The following user query will require outputting bounding boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. You MUST output a single parseable json list of objects enclosed into ```json...``` brackets, for instance ```json[{"box_2d": [800, 3, 840, 471], "label": "car"}, {"box_2d": [400, 22, 600, 73], "label": "dog"}]``` is a valid output. Now answer to the user query.

Remember "left" refers to the patient's left side where the heart is and sometimes underneath an L in the upper right corner of the image."""


class LocalizationGetItem(GetItem):
    """
    Extracts and transforms detection data for MedGemma localization fine-tuning.
    
    Each patch sample (after to_patches()) contains:
    - filepath: path to the full image
    - detection: the Detection object (bbox, label, etc.)
    - metadata: image dimensions
    
    We transform this into MedGemma's expected message format with:
    - System prompt explaining the bbox output format
    - User message with the localization query
    - Assistant message with the target bbox in JSON format
    """
    
    def __init__(self, field_mapping=None):
        # Must call super().__init__() with field_mapping - this sets up
        # the internal mapping that FiftyOne uses to pull the right fields
        super().__init__(field_mapping=field_mapping)
    
    @property
    def required_keys(self):
        # These are the keys we'll access in __call__.
        # 'detection' is a virtual key that gets mapped to the real field
        # via field_mapping. 'filepath' and 'metadata' are standard fields
        # that exist on all FiftyOne samples.
        return ["filepath", "detection", "metadata"]
    
    def __call__(self, d):
        """
        Transform a FiftyOne sample dict into MedGemma fine-tuning format.
        
        This is where the FiftyOne → MedGemma conversion happens:
        - FiftyOne bbox format [x, y, w, h] in [0,1] 
        - MedGemma format [y0, x0, y1, x1] normalized to [0, 1000]
        """
        filepath = d["filepath"]
        detection = d["detection"]
        
        # Get the label from the detection
        label = detection.label
        
        # --- Bounding Box Conversion ---
        # FiftyOne stores bboxes as [x, y, width, height] with values in [0, 1]
        # MedGemma expects [y0, x0, y1, x1] normalized to [0, 1000]
        rx, ry, rw, rh = detection.bounding_box
        
        # Convert to [y0, x0, y1, x1] format, scaled to [0, 1000]
        x0 = int(rx * 1000)
        y0 = int(ry * 1000)
        x1 = int((rx + rw) * 1000)
        y1 = int((ry + rh) * 1000)
        
        # Format as [y0, x0, y1, x1] per the prompt instructions
        bbox_normalized = [y0, x0, y1, x1]
        
        # --- Construct Messages ---
        # Format the target response as JSON
        target_json = f'```json[{{"box_2d": {bbox_normalized}, "label": "{label}"}}]```'
        
        # Build the message payload in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{LOCALIZATION_SYSTEM_PROMPT}\n\nLocate the {label} in this medical image."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_json},
                ],
            },
        ]
        
        return {
            "filepath": filepath,
            "image": Image.open(filepath).convert("RGB"),
            "messages": messages,
            "label": label,
        }
```

### Step 2: Create train/val split and flatten detections

Since our dataset doesn't have existing train/val [tags](https://docs.voxel51.com/user_guide/basics.html#tags), 
we'll create them using [`random_split()`](https://docs.voxel51.com/api/fiftyone.utils.random.html#fiftyone.utils.random.random_split).

Then we use [`to_patches()`](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.to_patches) 
to flatten the dataset so each detection becomes its own sample.

**Key insight:** `to_patches(field)` creates a view where each detection in that field becomes 
its own sample. If you have 100 images with 5 detections each, `to_patches` gives you 500 patch samples. 
This is perfect for instance-level training.

```python
import fiftyone.utils.random as four
from fiftyone import ViewField as F

# Filter to samples that have detections
has_detections_view = dataset.match(F("detections") != None)
print(f"Samples with detections: {len(has_detections_view)}")

# Create train/val split (80/20)
four.random_split(has_detections_view, {"train": 0.8, "val": 0.2})
```

```python
# Filter by split tags using match_tags()
# https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.match_tags
train_view = has_detections_view.match_tags("train")
val_view = has_detections_view.match_tags("val")

print(f"Samples - train: {len(train_view)}, val: {len(val_view)}")
```

```python
# Flatten using to_patches() - each detection becomes its own sample
train_patches = train_view.to_patches("detections")
val_patches = val_view.to_patches("detections")

print(f"Patches - train: {len(train_patches)}, val: {len(val_patches)}")
```

### Step 3: Convert to PyTorch datasets

Use [`to_torch()`](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.to_torch) 
with our `GetItem` class to create PyTorch-compatible datasets.

In the patches view, each sample's detection data lives in the original field (e.g., "detections"). 
The `field_mapping` lets us access it with a generic name in our `GetItem` code, making the class 
reusable across different datasets.

```python
# Set up field mapping - in patches view, each sample's detection data 
# lives in the original field "detections"
field_mapping = {"detection": "detections"}

# Create GetItem instances
train_getter = LocalizationGetItem(field_mapping=field_mapping)
val_getter = LocalizationGetItem(field_mapping=field_mapping)

# Convert to PyTorch datasets
train_dataset = train_patches.to_torch(train_getter)
val_dataset = val_patches.to_torch(val_getter)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
```

```python
# Verify the data format
sample = train_dataset[0]
print("Sample keys:", sample.keys())
print("Label:", sample["label"])
print("Messages structure:")
for msg in sample["messages"]:
    print(f"  Role: {msg['role']}")
```

### Step 4: Load MedGemma with QLoRA configuration

We use 4-bit quantization (QLoRA) to reduce memory requirements while maintaining
fine-tuning capability. This allows fine-tuning on consumer GPUs.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

model_id = "google/medgemma-1.5-4b-it"

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    ),
)

model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "right"
```

### Step 5: Configure LoRA

LoRA (Low-Rank Adaptation) allows efficient fine-tuning by only training 
small adapter matrices instead of all model weights.

```python
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)
```

### Step 6: Define the collate function

The collate function processes batches by:
1. Applying the chat template to format messages
2. Processing images and text together
3. Creating labels with proper masking for padding and image tokens

```python
def collate_fn(examples: list[dict[str, Any]]):
    texts = []
    images = []
    
    for example in examples:
        # Convert image to RGB and wrap in list (processor expects list of images per sample)
        images.append([example["image"].convert("RGB")])
        
        # Apply chat template to format the conversation
        texts.append(processor.apply_chat_template(
            example["messages"], 
            add_generation_prompt=False, 
            tokenize=False
        ).strip())
    
    # Tokenize texts and process images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    # Create labels from input_ids
    # We mask padding tokens and image tokens so they don't contribute to loss
    labels = batch["input_ids"].clone()
    
    # Get the image token ID to mask it
    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.special_tokens_map["boi_token"]
    )
    
    # Mask tokens that should not be used in loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100  # Additional image token ID
    
    batch["labels"] = labels
    return batch
```

### Step 7: Configure training

We use TRL's `SFTConfig` and `SFTTrainer` for a clean training setup with
all the best practices built in.

```python
from trl import SFTConfig, SFTTrainer

num_train_epochs = 1  # Adjust based on your needs
learning_rate = 2e-4

training_args = SFTConfig(
    output_dir="medgemma-localization-lora",         # Directory to save the model
    num_train_epochs=num_train_epochs,               # Number of training epochs
    per_device_train_batch_size=4,                   # Batch size per device during training
    per_device_eval_batch_size=4,                    # Batch size per device during evaluation
    gradient_accumulation_steps=4,                   # Number of steps before performing a backward/update pass
    gradient_checkpointing=True,                     # Enable gradient checkpointing to reduce memory usage
    optim="adamw_torch_fused",                       # Use fused AdamW optimizer for better performance
    logging_steps=50,                                # Number of steps between logs
    save_strategy="epoch",                           # Save checkpoint every epoch
    eval_strategy="steps",                           # Evaluate every `eval_steps`
    eval_steps=50,                                   # Number of steps between evaluations
    learning_rate=learning_rate,                     # Learning rate
    bf16=True,                                       # Use bfloat16 precision
    max_grad_norm=0.3,                               # Max gradient norm
    warmup_steps=5,                               # Warmup steps
    lr_scheduler_type="linear",                      # Use linear learning rate scheduler
    push_to_hub=False,                               # Set to True to push model to Hub
    report_to="tensorboard",                         # Report metrics to tensorboard
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_kwargs={"skip_prepare_dataset": True},   # We preprocess manually
    remove_unused_columns=False,                     # Keep columns for data collator
    label_names=["labels"],                          # Input keys that correspond to labels
)
```

### Step 8: Create trainer and train!

```python
# Workaround for MedGemma 1.5's SiglipVisionTransformer
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer
SiglipVisionTransformer.get_input_embeddings = lambda self: None

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)
```

```python
# Start training
trainer.train()
```

```python
# Save the fine-tuned model
trainer.save_model()

# Optional: Push to Hugging Face Hub
# trainer.push_to_hub()
```

### Clean up GPU memory

```python
del model
del trainer

import gc
gc.collect()
torch.cuda.empty_cache()
```

## Evaluating Your Fine-Tuned Model

Of course, the above is just a blueprint for what to do. For the best results, you need to figure out the right data to train on as well as the training recipe.

Once you've fine-tuned MedGemma for localization, go back through the earlier 
sections of this notebook to evaluate how well your model performs:

1. **Load your fine-tuned model** and run inference on the validation set. To do this, you will need to [fork my implementation of MedGemma 1.5](https://github.com/harpreetsahota204/medgemma_1_5) as a remote zoo model and update the [model maifest](https://github.com/harpreetsahota204/medgemma_1_5/blob/main/manifest.json) to download your weights. You may also need to make changes to the [zoo.py](https://github.com/harpreetsahota204/medgemma_1_5/blob/main/zoo.py) to merge your LORA with the original model. This is an exercise left to you.
2. **Store predictions** in FiftyOne alongside the ground truth
3. **Use the evaluation techniques** from Sections 5 and 6:
   - Compute accuracy by modality, body part, and content type
   - Analyze errors using the App and similarity search
   - Tag patterns in failures

You can use FiftyOne's [`evaluate_detections()` method](https://docs.voxel51.com/user_guide/evaluation.html#detections) 
to evaluate object detection predictions, computing metrics like mAP and per-class performance.

This iterative workflow—explore, model, evaluate, fine-tune—is how you systematically
improve your model's performance on specific failure modes.

---
## Bringing It All Together

Here's what you've learned to do:

| Step | What You Did | Why It Matters |
|------|-------------|----------------|
| Load & Explore | Understood data distribution before modeling | Caught potential issues early |
| Embeddings | Visualized MedSigLIP clusters | Diagnosed whether classes are separable |
| Inference | Ran MedGemma, stored predictions with data | Everything in one place for analysis |
| Evaluation | Sliced accuracy by modality, location, etc. | Found *where* the model fails |
| Error Analysis | Visualized failures, tagged patterns | Understood *why* it fails |
| Fine-Tuning | Used GetItem + SFTTrainer for localization | Improved model on specific failure modes |

**The workflow you built here works for any dataset, any model, any challenge.**

Whether you're doing:
- Chest X-ray report generation
- Dermatology classification  
- CT severity assessment
- Histopathology analysis

The pattern is the same:
1. Organize your data in FiftyOne
2. Understand it before modeling
3. Run inference, store predictions
4. Slice, visualize, debug
5. Fine-tune and iterate

**Now go win that challenge.** 🏆

---
## Resources

### FiftyOne Documentation
- [FiftyOne Documentation](https://docs.voxel51.com/)
- [FiftyOne Datasets](https://docs.voxel51.com/user_guide/using_datasets.html)
- [FiftyOne Views Cheat Sheet](https://docs.voxel51.com/cheat_sheets/views_cheat_sheet.html)
- [FiftyOne Filtering Cheat Sheet](https://docs.voxel51.com/cheat_sheets/filtering_cheat_sheet.html)
- [FiftyOne PyTorch Integration](https://docs.voxel51.com/integrations/pytorch.html)
- [FiftyOne Brain](https://docs.voxel51.com/brain.html) (embeddings, similarity, visualization)
- [FiftyOne Evaluation](https://docs.voxel51.com/user_guide/evaluation.html)
- [FiftyOne Model Zoo](https://docs.voxel51.com/model_zoo/overview.html)
- [FiftyOne Plugins](https://docs.voxel51.com/plugins/)

### Dataset & Models
- [SLAKE Dataset on HuggingFace](https://huggingface.co/datasets/Voxel51/SLAKE)
- [MedGemma Model Card](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedSigLIP Model Card](https://huggingface.co/google/medsiglip-448)

### Fine-Tuning
- [TRL SFTTrainer Documentation](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT LoRA Documentation](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

### Competition
- [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
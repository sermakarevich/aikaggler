# MABe Nearest Neighbors: The Original ⭐️⭐️⭐️⭐️⭐️

- **Author:** AmbrosM
- **Votes:** 381
- **Ref:** ambrosm/mabe-nearest-neighbors-the-original
- **URL:** https://www.kaggle.com/code/ambrosm/mabe-nearest-neighbors-the-original
- **Last run:** 2025-09-27 12:17:36.957000

---

# MABe Challenge - Social Action Recognition in Mice: Nearest neighbors

This is the original notebook for social action recognition with nearest neighbors. I've tried to explain what the code does—feel free to ask questions.

The notebook shows how to overcome the five challenges of this competition:
1. Modeling for variable-size sets of mice
2. Multiclass prediction with missing labels
3. Transforming coordinates to an invariant representation
4. A dataset that doesn't fit into memory
5. Modeling for variable sets of body parts

The title of the notebook mentions *Nearest Neighbors* because in earlier versions I used nearest neighbors classification, an algorithm which doesn't need a lot of tuning. The current version uses LightGBM, and maybe I'll ensemble the two later.

References
- Competition: [MABe Challenge - Social Action Recognition in Mice](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection)
- [MABe EDA which makes sense ⭐️⭐️⭐️⭐️⭐️](https://www.kaggle.com/code/ambrosm/mabe-eda-which-makes-sense)
- [MABe Validated baseline without machine learning](https://www.kaggle.com/code/ambrosm/mabe-validated-baseline-without-machine-learning)

This notebook can be run in validate or submission mode. If you look at other saved versions of this notebook, you'll see both modes. You can switch between the modes by setting the variable `validate_or_submit`:

```python
validate_or_submit = 'validate' # 'validate' or 'submit' or 'stresstest'
verbose = True
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import itertools
import warnings
import json
import os
import lightgbm

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
```

```python
class TrainOnSubsetClassifier(ClassifierMixin, BaseEstimator):
    """Fit estimator to a subset of the training data."""
    def __init__(self, estimator, n_samples):
        self.estimator = estimator
        self.n_samples = n_samples

    def fit(self, X, y):
        downsample = len(X) // self.n_samples
        downsample = max(downsample, 1)
        self.estimator.fit(np.array(X, copy=False)[::downsample],
                           np.array(y, copy=False)[::downsample])
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        if len(self.classes_) == 1:
            return np.full((len(X), 1), 1.0)
        probs = self.estimator.predict_proba(np.array(X))
        return probs
        
    def predict(self, X):
        return self.estimator.predict(np.array(X))
```

```python
"""F Beta customized for the data format of the MABe challenge."""

import json

from collections import defaultdict

import pandas as pd
import polars as pl


class HostVisibleError(Exception):
    pass


def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set) # key is video/agent/target/action from solution
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set) # key is video/agent/target/action from submission

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()  # ty: ignore
        active_labels: set[str] = set(json.loads(active_labels)) # set of agent,target,action from solution
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set) # key is agent,target from submission

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts(): # every submission row is converted to a dict
            # Since the labels are sparse, we can't evaluate prediction keys not in the active labels.
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                # print(f'ignoring {video}', ','.join([str(row['agent_id']), str(row['target_id']), row['action']]), active_labels)
                continue # these submission rows are ignored
           
            new_frames = set(range(row['start_frame'], row['stop_frame']))
            # Ignore truly redundant predictions.
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                # A single agent can have multiple targets per frame (ex: evading all other mice) but only one action per target per frame.
                raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int) # key is action
    fns = defaultdict(int) # key is action
    fps = defaultdict(int) # key is action
    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    action_f1s = []
    for action in distinct_actions:
        # print(f"{tps[action]:8} {fns[action]:8} {fps[action]:8}")
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
    return sum(action_f1s) / len(action_f1s)


def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    """
    Doctests:
    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10},
    ... ])
    >>> mouse_fbeta(solution, submission)
    1.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 0, 'stop_frame': 10}, # Wrong action
    ... ])
    >>> mouse_fbeta(solution, submission)
    0.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9},
    ... ])
    >>> "%.12f" % mouse_fbeta(solution, submission)
    '0.500000000000'

    >>> solution = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 345, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 2, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 345, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 2, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9},
    ... ])
    >>> "%.12f" % mouse_fbeta(solution, submission)
    '0.250000000000'

    >>> # Overlapping solution events, one prediction matching both.
    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 10, 'stop_frame': 20, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 20},
    ... ])
    >>> mouse_fbeta(solution, submission)
    1.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 30, 'stop_frame': 40, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 40},
    ... ])
    >>> mouse_fbeta(solution, submission)
    0.6666666666666666
    """
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']

    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution is missing column {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission is missing column {col}')

    solution: pl.DataFrame = pl.DataFrame(solution)
    submission: pl.DataFrame = pl.DataFrame(submission)
    assert (solution['start_frame'] <= solution['stop_frame']).all()
    assert (submission['start_frame'] <= submission['stop_frame']).all()
    solution_videos = set(solution['video_id'].unique())
    # Need to align based on video IDs as we can't rely on the row IDs for handling public/private splits.
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    solution = solution.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission = submission.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    """
    F1 score for the MABe Challenge
    """
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)
```

We start by reading the training metadata from train.csv.

```python
train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
train_without_mabe22 = train.query("~ lab_id.str.startswith('MABe22_')")

test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')

# labs = list(np.unique(train.lab_id))

body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

# behaviors = list(train.behaviors_labeled.drop_duplicates().dropna())
# behaviors = sorted(list({b.replace("'", "") for bb in behaviors for b in json.loads(bb)}))
# behaviors = [b.split(',') for b in behaviors]
# behaviors = pd.DataFrame(behaviors, columns=['agent', 'target', 'action'])
```

We collect all annotation files and bring the true labels into the format required by the competition scoring function.

```python
def create_solution_df(dataset):
    """Create the solution dataframe for validating out-of-fold predictions.

    From https://www.kaggle.com/code/ambrosm/mabe-validated-baseline-without-machine-learning/
    
    Parameters:
    dataset: (a subset of) the train dataframe
    
    Return values:
    solution: solution dataframe in the correct format for the score() function
    """
    solution = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
    
        # Load annotation file
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'): continue
        video_id = row['video_id']
        path = f"/kaggle/input/MABe-mouse-behavior-detection/train_annotation/{lab_id}/{video_id}.parquet"
        try:
            annot = pd.read_parquet(path)
        except FileNotFoundError:
            # MABe22 and one more training file lack annotations.
            if verbose: print(f"No annotations for {path}")
            continue
    
        # Add all annotations to the solution
        annot['lab_id'] = lab_id
        annot['video_id'] = video_id
        annot['behaviors_labeled'] = row['behaviors_labeled']
        annot['target_id'] = np.where(annot.target_id != annot.agent_id, annot['target_id'].apply(lambda s: f"mouse{s}"), 'self')
        annot['agent_id'] = annot['agent_id'].apply(lambda s: f"mouse{s}")
        solution.append(annot)
    
    solution = pd.concat(solution)
    return solution

if validate_or_submit == 'validate':
    solution = create_solution_df(train_without_mabe22)
```

# Stress testing with unusual inputs

After submission, this notebook will see a test set that it has never seen before. If the notebook crashes, debugging will be hard. It's better to stress-test the notebook before the submission by giving it some unusual inputs. The following hidden cell generate synthetic data with missing values, excessively long videos and so on.

```python
if validate_or_submit == 'stresstest':
    n_videos_per_lab = 2
    
    try:
        os.mkdir(f"stresstest_tracking")
    except FileExistsError:
        pass
    
    rng = np.random.default_rng()
    stresstest = pd.concat(
        [train.query("video_id == 1459695188")] # long video from BoisterousParrot
        + [df.sample(min(n_videos_per_lab, len(df)), random_state=1) for (_, df) in train.groupby('lab_id')])
    for _, row in tqdm(stresstest.iterrows(), total=len(stresstest)):
        lab_id = row['lab_id']
        video_id = row['video_id']
        
        # Load video
        path = f"/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
    
        if video_id == 1459695188: # long video from BoisterousParrot
            vid = pd.concat([vid] * 3) # provoke out of memory (5 is too much)
            vid['video_frame'] = np.arange(len(vid))
    
        # Drop some complete frames
        dropped_frames = list(rng.choice(np.unique(vid.video_frame), size=100, replace=False))
        vid = vid.query("~ video_frame.isin(@dropped_frames)")
        
        # Drop a complete bodypart
        if rng.uniform() < 0.2:
            dropped_bodypart = rng.choice(np.unique(vid.bodypart), size=1, replace=False)[0]
            vid = vid.query("bodypart != @dropped_bodypart")
        
        # Drop a mouse
        if rng.uniform() < 0.1:
            vid = vid.query("mouse_id != 1")
        
        # Drop random bodyparts from random frames
        if rng.uniform() < 0.7:
            mask = np.ones(len(vid), dtype=bool)
            mask[:int(0.4 * len(mask))] = False
            rng.shuffle(mask)
            vid = vid[mask]
    
        # Set random coordinates of bodyparts to nan
        if rng.uniform() < 0.7:
            mask = np.ones(len(vid), dtype=bool)
            mask[:int(0.2 * len(mask))] = False
            rng.shuffle(mask)
            vid.loc[:, 'x'] = np.where(mask, np.nan, vid.loc[:, 'x'])
            rng.shuffle(mask)
            vid.loc[:, 'y'] = np.where(mask, np.nan, vid.loc[:, 'y'])
    
        # Save the video
        try:
            os.mkdir(f"stresstest_tracking/{lab_id}")
        except FileExistsError:
            pass
        new_path = f"stresstest_tracking/{lab_id}/{video_id}.parquet"
        vid.to_parquet(new_path)
```

# Challenge 1: Modeling for variable-sized sets of mice

The first challenge we're going to solve is the fact that we have a variable number of mice (2, 3 or 4), and that the labeled behaviors apply either to one mouse or a pair of mice.

The following function, `generate_mouse_data()`, solves this challenge. It transforms the dataset into batches. There are single-mouse batches and mouse-pair batches. Every single-mouse batch has data of only one mouse, every mouse-pair batch has data of exactly two mice. A single video frame can end up in several batches. If the frame has two visible mice, it can be part of four batches:
- a single-mouse batch for individual behavior of mouse 1
- a single-mouse batch for individual behavior of mouse 2
- a mouse-pair batch for actions of mouse 1 with mouse 2 as target
- a mouse-pair batch for actions of mouse 2 with mouse 1 as target

The features (`data`) will consist of coordinates of body parts; the metadata (`meta`) will specify which mouse is / which mice are involved.

```python
drop_body_parts =  ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
                    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
                    'spine_1', 'spine_2',
                    'tail_middle_1', 'tail_middle_2', 'tail_midpoint']

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    """Generate batches of data in coordinate representation.

    The batches have variable length, and every batch can have other columns
    for the labels, depending on what behaviors
    were labeled for the batch.

    Every video can produce zero, one or two batches.
    
    Parameters
    ----------
    dataset: (subset of) train.csv or test.csv dataframe
    traintest: either 'train' or 'test'

    Yields
    ------
    switch: either 'single' or 'pair'
    data: dataframe containing coordinates of the body parts of a single mouse or of a pair of mice
    meta: dataframe with columns ['video_id', 'agent_id', 'target_id', 'video_frame']
    label: dataframe with labels (0, 1), one column per action, only if traintest == 'train'
    actions: list of actions to be predicted for this batch, only if traintest == 'test'
    """
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    for _, row in dataset.iterrows():
        
        # Load the video and pivot it sn that one frame = one row
        lab_id = row.lab_id
        if lab_id.startswith('MABe22'): continue
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            # We cannot use videos without labeled behaviors
            print('No labeled behaviors:', lab_id, video_id)
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        if pvid.isna().any().any():
            if verbose and traintest == 'test': print('video with missing values', video_id, traintest, len(vid), 'frames')
        else:
            if verbose and traintest == 'test': print('video with all values', video_id, traintest, len(vid), 'frames')
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T # mouse_id, body_part, xy
        pvid /= row.pix_per_cm_approx # convert to cm

        # Determine the behaviors of this video
        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        
        # Load the annotations
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                # MABe22 and one more training file lack annotations.
                # We simply drop these videos.
                continue

        # Create the single_mouse dataframes: single_mouse, single_mouse_label and single_mouse_meta
        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'") # single-mouse behaviors of this video
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    assert len(single_mouse) == len(pvid)
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                    else:
                        if verbose: print('- test single', video_id, mouse_id)
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                except KeyError:
                    pass # If there is no data for the selected agent mouse, we skip the mouse.

        # Create the mouse_pair dataframes: mouse_pair, mouse_label and mouse_meta
        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2): # int8
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    assert len(mouse_pair) == len(pvid)
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        if verbose: print('- test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions
```

# Challenge 2: Multiclass prediction with missing labels

This competition is a multi-class classification task. For every video_id/video_frame/agent/target combination, we may predict at most one of several actions. Every action is a class, and 'no-action' is an additional class.

We cannot use a standard multi-class estimator from scikit-learn because many values in the labels of our dataset are missing. For this reason, we train a binary classifier for every action, omitting the samples for which the target is unknown. Every binary classificator predicts a probability, and for the multiclass prediction we predict the class with the highest binary probability, if this probability is above a threshold; otherwise, we predict no action.

```python
# Make the multi-class prediction
def predict_multiclass(pred, meta):
    """Derive multiclass predictions from a set of binary predictions.
    
    Parameters
    pred: dataframe of predicted binary probabilities, shape (n_samples, n_actions), index doesn't matter
    meta: dataframe with columns ['video_id', 'agent_id', 'target_id', 'video_frame'], index doesn't matter
    """
    # Find the most probable class, but keep it only if its probability is above the threshold
    ama = np.argmax(pred, axis=1)
    ama = np.where(pred.max(axis=1) >= threshold, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)
    # Keep only start and stop frames
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    # mask selects the start frames
    mask = ama_changes.values >= 0 # start of action
    mask[-1] = False
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
    assert (submission_part.stop_frame > submission_part.start_frame).all(), 'stop <= start'
    if verbose: print('  actions found:', len(submission_part))
    return submission_part
```

# Challenge 3: Transforming coordinates to an invariant representation

The body part of the mice are given in cartesian coordinates. If the mice show some behavior at varying positions and with varying spatial orientation, cartesian coordinates are an inadequate representation. Our feature engineering transforms the coordinates to distances between body parts. Distances are invariant under translation and rotation.

For a single mouse, the distances indicate whether and how much it turns its head, shoulders, hip and tail left or right. For a pair of mice, the distances indicate how far the head of the first mouse is near what part of the second one, and what body parts either mouse turns towards or away from the other one.

```python
def transform_single(single_mouse, body_parts_tracked):
    """Transform from cartesian coordinates to distance representation.

    Parameters:
    single_mouse: dataframe with coordinates of the body parts of one mouse
                  shape (n_samples, n_body_parts * 2)
                  two-level MultiIndex on columns
    body_parts_tracked: list of body parts
    """
    available_body_parts = single_mouse.columns.get_level_values(0)
    X = pd.DataFrame({
            f"{part1}+{part2}": np.square(single_mouse[part1] - single_mouse[part2]).sum(axis=1, skipna=False)
            for part1, part2 in itertools.combinations(body_parts_tracked, 2) if part1 in available_body_parts and part2 in available_body_parts
        })
    X = X.reindex(columns=[f"{part1}+{part2}" for part1, part2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    if 'ear_left' in single_mouse.columns and 'ear_right' in single_mouse.columns and 'tail_base' in single_mouse.columns:
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(10)
        X = pd.concat([
            X, 
            pd.DataFrame({
                'speed_left': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
                'speed_right': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
                'speed_left2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
                'speed_right2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
            })
        ], axis=1)
    return X

def transform_pair(mouse_pair, body_parts_tracked):
    """Transform from cartesian coordinates to distance representation.

    Parameters:
    mouse_pair: dataframe with coordinates of the body parts of two mice
                  shape (n_samples, 2 * n_body_parts * 2)
                  three-level MultiIndex on columns
    body_parts_tracked: list of body parts
    """
    # drop_body_parts =  ['ear_left', 'ear_right',
    #                     'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    #                     'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
    #                     'tail_midpoint']
    # if len(body_parts_tracked) > 5:
    #     body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    available_body_parts_A = mouse_pair['A'].columns.get_level_values(0)
    available_body_parts_B = mouse_pair['B'].columns.get_level_values(0)
    X = pd.DataFrame({
            f"12+{part1}+{part2}": np.square(mouse_pair['A'][part1] - mouse_pair['B'][part2]).sum(axis=1, skipna=False)
            for part1, part2 in itertools.product(body_parts_tracked, repeat=2) if part1 in available_body_parts_A and part2 in available_body_parts_B
        })
    X = X.reindex(columns=[f"12+{part1}+{part2}" for part1, part2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        shifted_A = mouse_pair['A']['ear_left'].shift(10)
        shifted_B = mouse_pair['B']['ear_left'].shift(10)
        X = pd.concat([
            X,
            pd.DataFrame({
                'speed_left_A': np.square(mouse_pair['A']['ear_left'] - shifted_A).sum(axis=1, skipna=False),
                'speed_left_AB': np.square(mouse_pair['A']['ear_left'] - shifted_B).sum(axis=1, skipna=False),
                'speed_left_B': np.square(mouse_pair['B']['ear_left'] - shifted_B).sum(axis=1, skipna=False),
            })
        ], axis=1)
    return X
```

# Cross-validation

We're now almost ready to cross-validate our models. 

The following function gets as input
- a binary classification model
- a 2d array of features (i.e., distances between body parts); after we have dealt with variable-sized mouse sets (challenge 1) and variable-sized bodyparts sets (challenge 5), this array is rectangular.
- a 2d array of binary labels, some elements of which may be missing
- a 2d array of metadata so that we can match the predictions with the original video_id, agent, target and video_frame

It first computes out-of-fold predictions with a set of binary classifiers and then transforms these binary predictions into a multiclass prediction (see above).

```python
threshold = 0.27

def cross_validate_classifier(binary_classifier, X, label, meta):
    """Cross-validate a binary classifier per action and a multi-class classifier over all actions.

    Parameters
    ----------
    binary_classifier: classifier with predict_proba
    X: 2d array-like (distance representation) of shape (n_samples, n_features)
    label: dataframe with binary targets (one column per action, may have missing values), index doesn't matter
    meta: dataframe with columns ['video_id', 'agent_id', 'target_id', 'video_frame'], index doesn't matter

    Output
    ------
    appends to f1_list (binary) and submission_list (multi-class)
    
    """
    # Cross-validate a binary classifier for every action
    oof = pd.DataFrame(index=meta.video_frame) # will get a column per action
    for action in label.columns:
        # Filter for samples (video frames) with a defined target (i.e., target is not nan)
        action_mask = ~ label[action].isna().values
        X_action = X[action_mask]
        y_action = label[action][action_mask].values.astype(int)
        p = y_action.mean()
        baseline_score = p / (1 + p)
        groups_action = meta.video_id[action_mask] # ensure validation has unseen videos
        if len(np.unique(groups_action)) < 5:
            continue # GroupKFold would fail with fewer than n_splits groups

        if not (y_action == 0).all():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                # Number of classes in training fold (1) does not match total number of classes (2)
                oof_action = cross_val_predict(binary_classifier, X_action, y_action, groups=groups_action, cv=GroupKFold(), method='predict_proba')
            oof_action = oof_action[:, 1]
        else:
            oof_action = np.zeros(len(y_action))
        f1 = f1_score(y_action, (oof_action >= threshold), zero_division=0)
        ch = '>' if f1 > baseline_score else '=' if f1 == baseline_score else '<'
        print(f"  F1: {f1:.3f} {ch} ({baseline_score:.3f}) {action}")
        f1_list.append((body_parts_tracked_str, action, f1))
        oof_column = np.zeros(len(label))
        oof_column[action_mask] = oof_action
        oof[action] = oof_column

    # Make the multi-class prediction
    submission_part = predict_multiclass(oof, meta)
    submission_list.append(submission_part)
```

# Challenge 4: A dataset that doesn't fit into memory

The competition dataset doesn't fit into memory as whole. The problem is exacerbated if we compute lots of distance in feature engineering. We tackle this challenge with the following measures:
- Training on a subset of the data: The training dataset is highly redundant. In videos taken with 30 frames per second, the difference from one frame to the next is small. We can well afford to subsample the training data.
- Processing the test data in batches: There is no need to have the full test dataset in memory at any time. (This decision has the drawback that the test data are read from disk several times.)
- It helps that we split all data by body_parts_tracked (see challenge 5 below). This way, we don't even need to have the full training dataset in memory.

```python
def submit(body_parts_tracked_str, switch_tr, binary_classifier, X_tr, label, meta):
    """Produce a submission file for the selected subset of the test data.

    Parameters
    ----------
    body_parts_tracked_str: subset of body parts for filtering the test set
    switch_tr: 'single' or 'pair'
    binary_classifier: classifier with predict_proba
    X_tr: training features as 2d array-like of shape (n_samples, n_features)
    label: dataframe with binary targets (one column per action, may have missing values), index doesn't matter
    meta: dataframe with columns ['video_id', 'agent_id', 'target_id', 'video_frame'], index doesn't matter

    Output
    ------
    appends to submission_list
    
    """
    # Fit a binary classifier for every action
    model_list = [] # will get a model per action
    for action in label.columns:
        # Filter for samples (video frames) with a defined target (i.e., target is not nan)
        action_mask = ~ label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)

        if not (y_action == 0).all():
            model = clone(binary_classifier)
            model.fit(X_tr[action_mask], y_action)
            assert len(model.classes_) == 2
            model_list.append((action, model))

    # Compute test predictions in batches
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    if validate_or_submit == 'submit':
        test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
        generator = generate_mouse_data(test_subset, 'test',
                                        generate_single=(switch_tr == 'single'), 
                                        generate_pair=(switch_tr == 'pair'))
    else:
        test_subset = stresstest.query("body_parts_tracked == @body_parts_tracked_str")
        generator = generate_mouse_data(test_subset, 'test',
                                        traintest_directory='stresstest_tracking',
                                        generate_single=(switch_tr == 'single'),
                                        generate_pair=(switch_tr == 'pair'))
    if verbose: print(f"n_videos: {len(test_subset)}")
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            # Transform from coordinate representation into distance representation
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked) # may raise KeyError
            else:
                X_te = transform_pair(data_te, body_parts_tracked) # may raise KeyError
            if verbose and len(X_te) == 0: print("ERROR: X_te is empty")
            del data_te
    
            # Compute binary predictions
            pred = pd.DataFrame(index=meta_te.video_frame) # will get a column per action
            for action, model in model_list:
                if action in actions_te:
                    pred[action] = model.predict_proba(X_te)[:, 1]
            del X_te
            # Compute multiclass predictions
            if pred.shape[1] != 0:
                submission_part = predict_multiclass(pred, meta_te)
                submission_list.append(submission_part)
            else: # this happens if there was no useful training data for the test actions
                if verbose: print(f"  ERROR: no useful training data")
        except KeyError:
            if verbose: print(f'  ERROR: KeyError because of missing bodypart ({switch_tr})')
            del data_te
```

# Challenge 5: Modeling for variable sets of body parts

Different labs have tracked different sets of body parts, but a machine learning model expects to see the same features for every sample. We solve this challenge by the principle of divide and conquer: For every set of body parts, we fit separate models.

```python
# %%time
f1_list = []
submission_list = []
for section in range(1, len(body_parts_tracked_list)): # skip index 0 (MABe22)
    body_parts_tracked_str = body_parts_tracked_list[section]
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"{section}. Processing videos with {body_parts_tracked}")
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
        # We read all training data which match the body parts tracked
        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
        single_mouse_list = []
        single_mouse_label_list = []
        single_mouse_meta_list = []
        mouse_pair_list = []
        mouse_pair_label_list = []
        mouse_pair_meta_list = []
    
        for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
            if switch == 'single':
                single_mouse_list.append(data)
                single_mouse_meta_list.append(meta)
                single_mouse_label_list.append(label)
            else:
                mouse_pair_list.append(data)
                mouse_pair_meta_list.append(meta)
                mouse_pair_label_list.append(label)
    
        # Construct a binary classifier
        binary_classifier = make_pipeline(
            SimpleImputer(),
            TrainOnSubsetClassifier(
                lightgbm.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.03,
                    min_child_samples=40,
                    # early_stopping_round=10, 
                    verbose=-1),
                100000)
        )
    
        # Predict single-mouse actions
        if len(single_mouse_list) > 0:
            # Concatenate all batches
            # The concatenation will generate label dataframes with missing values.
            single_mouse = pd.concat(single_mouse_list)
            single_mouse_label = pd.concat(single_mouse_label_list)
            single_mouse_meta = pd.concat(single_mouse_meta_list)
            del single_mouse_list, single_mouse_label_list, single_mouse_meta_list
            assert len(single_mouse) == len(single_mouse_label)
            assert len(single_mouse) == len(single_mouse_meta)
            
            # Transform the coordinate representation into a distance representation for single_mouse
            X_tr = transform_single(single_mouse, body_parts_tracked)
            del single_mouse
            print(f"{X_tr.shape=}")
    
            if validate_or_submit == 'validate':
                cross_validate_classifier(binary_classifier, X_tr, single_mouse_label, single_mouse_meta)
            else:
                submit(body_parts_tracked_str, 'single', binary_classifier, X_tr, single_mouse_label, single_mouse_meta)
            del X_tr
                
        # Predict mouse-pair actions
        if len(mouse_pair_list) > 0:
            # Concatenate all batches
            # The concatenation will generate label dataframes with missing values.
            mouse_pair = pd.concat(mouse_pair_list)
            mouse_pair_label = pd.concat(mouse_pair_label_list)
            mouse_pair_meta = pd.concat(mouse_pair_meta_list)
            del mouse_pair_list, mouse_pair_label_list, mouse_pair_meta_list
            assert len(mouse_pair) == len(mouse_pair_label)
            assert len(mouse_pair) == len(mouse_pair_meta)
        
            # Transform the coordinate representation into a distance representation for mouse_pair
            # Use a subset of body_parts_tracked to conserve memory
            X_tr = transform_pair(mouse_pair, body_parts_tracked)
            del mouse_pair
            print(f"{X_tr.shape=}")
    
            if validate_or_submit == 'validate':
                cross_validate_classifier(binary_classifier, X_tr, mouse_pair_label, mouse_pair_meta)
            else:
                submit(body_parts_tracked_str, 'pair', binary_classifier, X_tr, mouse_pair_label, mouse_pair_meta)
            del X_tr
                
    except Exception as e:
        print(f'***Exception*** {e}')
    print()
```

The code above probably contains bugs, but we don't want the submission to fail because of these bugs. The function `robustify` modifies the submission dataframe so that it conforms to the rules of the competition.

```python
def robustify(submission, dataset, traintest, traintest_directory=None):
    """Ensure that the submission conforms to the three rules"""
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    # Rule 1: Ensure that start_frame >= stop_frame
    old_submission = submission.copy()
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != len(old_submission):
        print("ERROR: Dropped frames with start >= stop")
    
    # Rule 2: Avoid multiple predictions for the same frame from one agent/target pair
    old_submission = submission.copy()
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop_frame:
                mask[i] = False
            else:
                last_stop_frame = row['stop_frame']
        group_list.append(group[mask])
    submission = pd.concat(group_list)
    if len(submission) != len(old_submission):
        print("ERROR: Dropped duplicate frames")

    # Rule 3: Submit something for every video
    # Fill missing videos as in https://www.kaggle.com/code/ambrosm/mabe-validated-baseline-without-machine-learning
    s_list = []
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):
            continue
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue
        if type(row.behaviors_labeled) != str:
            continue

        if verbose: print(f"Video {video_id} has no predictions.")
        
        # Load video
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
    
        # Determine the behaviors of this video
        vid_behaviors = json.loads(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
    
        # Determine start_frame and stop_frame
        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1
    
        # Predict all possible actions as often as possible
        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])
        print("ERROR: Filled empty videos")

    submission = submission.reset_index(drop=True)
    return submission
```

```python
if validate_or_submit == 'validate':
    # Score the oof predictions with the competition scoring function
    submission = pd.concat(submission_list)
    submission_robust = robustify(submission, train, 'train')
    print(f"# OOF score with competition metric: {score(solution, submission_robust, ''):.4f}")

    f1_df = pd.DataFrame(f1_list, columns=['body_parts_tracked_str', 'action', 'binary F1 score'])
    print(f"# Average of {len(f1_df)} binary F1 scores {f1_df['binary F1 score'].mean():.4f}")
    # with pd.option_context('display.max_rows', 500):
    #     display(f1_df)
```

```python
if validate_or_submit != 'validate':
    if len(submission_list) > 0:
        submission = pd.concat(submission_list)
    else:
        submission = pd.DataFrame(
            dict(
                video_id=438887472,
                agent_id='mouse1',
                target_id='self',
                action='rear',
                start_frame='278',
                stop_frame='500'
            ), index=[44])
    if validate_or_submit == 'submit':
        submission_robust = robustify(submission, test, 'test')
    else:
        submission_robust = robustify(submission, stresstest, 'stresstest', 'stresstest_tracking')
    submission_robust.index.name = 'row_id'
    submission_robust.to_csv('submission.csv')
    !head submission.csv
```
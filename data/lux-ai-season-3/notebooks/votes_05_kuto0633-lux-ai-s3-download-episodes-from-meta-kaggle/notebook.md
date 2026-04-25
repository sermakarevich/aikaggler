# Lux AI S3 Download Episodes from Meta Kaggle

- **Author:** kuto
- **Votes:** 126
- **Ref:** kuto0633/lux-ai-s3-download-episodes-from-meta-kaggle
- **URL:** https://www.kaggle.com/code/kuto0633/lux-ai-s3-download-episodes-from-meta-kaggle
- **Last run:** 2024-12-27 03:19:55.100000

---

```python
import pandas as pd
import numpy as np
from pathlib import Path
import os
import requests
import json
from tqdm.auto import tqdm
import datetime
import time
import glob
import collections
import polars as pl
```

```python
OUTPUT_DIR = Path("./")
META_DIR = Path("../input/meta-kaggle/")
BASE_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/"
GET_URL = BASE_URL + "GetEpisodeReplay"

LOWEST_SCORE_THRESH = 2000
EPISODE_LIMIT_SIZE = 1000  # Kaggle says don't do more than 3600 per day and 1 per second
```

## Get competition id
Originally, we intended to retrieve the CompetitionId from the Competitions.csv file. However, it appears that the CSV does not yet include lux-ai-season3.   
As a result, we need an alternative method to obtain the Id. In this case, we will use an arbitrary submission ID from the competition to extract the CompetitionId.

```python
# pl.read_csv(
#     "/kaggle/input/meta-kaggle/Competitions.csv",
#     schema_overrides={"UserRankMultiplier": pl.Float64}
# ).filter(pl.col("Slug").str.contains("lux"))
```

```python
# %%time

# def get_competition_id_by_submission(sample_submission_id: int) -> int:
#     agents_df = pl.scan_csv(META_DIR / "EpisodeAgents.csv", schema_overrides={'Reward':pl.Float32})
#     sample_episode_id = agents_df.filter(pl.col("SubmissionId")==sample_submission_id).collect()["EpisodeId"].unique()[0]
#     episodes_df = pl.scan_csv(META_DIR / "Episodes.csv")
#     competition_id = episodes_df.filter(pl.col("Id")==sample_episode_id).collect()["CompetitionId"].unique()[0]
#     print(f"{competition_id=}")
#     return competition_id


# sample_submission_id = 41952777
# _ = get_competition_id_by_submission(sample_submission_id)/
```

```python
COMPETITION_ID = 86411  # lux-ai-s3
```

## Extract Top Submission

```python
%%time

episodes_df = pl.scan_csv(META_DIR / "Episodes.csv")
episodes_df = (
    episodes_df
    .filter(pl.col('CompetitionId')==COMPETITION_ID)
    .with_columns(
        pl.col("CreateTime").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False),
        pl.col("EndTime").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False),
    )
    .sort("Id")
    .collect()
)
print(f'Episodes.csv: {len(episodes_df)} rows.')
```

```python
episodes_df
```

```python
%%time

agents_df = pl.scan_csv(
    META_DIR / "EpisodeAgents.csv", 
    schema_overrides={'Reward':pl.Float32, 'UpdatedConfidence': pl.Float32, 'UpdatedScore': pl.Float32}
)

agents_df = (
    agents_df
    .filter(pl.col("EpisodeId").is_in(episodes_df['Id'].to_list()))
    .with_columns([
        pl.when(pl.col("InitialConfidence") == "")
        .then(None)
        .otherwise(pl.col("InitialConfidence"))
        .cast(pl.Float64)
        .alias("InitialConfidence"),
        
        pl.when(pl.col("InitialScore") == "")
        .then(None)
        .otherwise(pl.col("InitialScore"))
        .cast(pl.Float64)
        .alias("InitialScore")])
    .collect()
)
print(f'EpisodeAgents.csv: {len(agents_df)} rows.')
```

```python
agents_df
```

```python
target_agents_df = (
    agents_df
    .sort('EpisodeId', descending=True)
    .group_by('SubmissionId')
    .head(1)
    .filter(pl.col("UpdatedScore")>LOWEST_SCORE_THRESH)
)

create_time_df = (
    agents_df
    .sort('EpisodeId', descending=False)
    .group_by('SubmissionId')
    .head(1)
    .join(episodes_df, left_on='EpisodeId', right_on='Id')
    .select(['SubmissionId', 'CreateTime'])
)

num_episodes_df = (
    agents_df
    .group_by('SubmissionId')
    .agg(pl.count().alias('NumEpisodes'))
)

target_agents_df = (
    target_agents_df
    .join(num_episodes_df, on='SubmissionId')
    .join(create_time_df, on='SubmissionId')
    .select(['SubmissionId', 'EpisodeId', 'UpdatedScore', 'NumEpisodes', 'CreateTime', "Index"])
)
```

```python
target_agents_df
```

```python
team_name_list = []
for row in tqdm(target_agents_df.iter_rows(named=True), total=len(target_agents_df)):
    ep_id = row['EpisodeId']
    team_idx = int(row['Index'])
    re = requests.post(GET_URL, json = {"episodeId": int(ep_id)})
    replay = re.json()
    team_name_list.append(replay['info']['TeamNames'][team_idx])
```

```python
target_agents_df = (
    target_agents_df
    .with_columns(pl.Series(team_name_list).alias('TeamName'))
    .drop(['EpisodeId', "Index"])
    .sort('UpdatedScore', descending=True)
)
```

```python
target_agents_df.head(10)
```

## Extract episode information

```python
TARGET_SUBMISSION_IDS = [41862933, 41863713]
target_episodes_df = agents_df.filter(pl.col("SubmissionId").is_in(TARGET_SUBMISSION_IDS))
target_episodes_df.write_csv("episodes.csv")
```

```python
def create_info_json(epid:int) -> dict:
    create_seconds = int(episodes_df.filter(pl.col('EpisodeId') == epid)['CreateTime'].item() / 1e9)
    end_seconds = int(episodes_df.filter(pl.col('EpisodeId') == epid)['CreateTime'].item() / 1e9)

    agents_df_filtered = agents_df.filter(pl.col('EpisodeId') == epid).sort('Index')

    agents = []
    for row in agents_df_filtered.iter_rows(named=True):
        agent = {
            "id": int(row["Id"]),
            "state": int(row["State"]),
            "submissionId": int(row['SubmissionId']),
            "reward": float(row['Reward']),
            "index": int(row['Index']),
            "initialScore": float(row['InitialScore']),
            "initialConfidence": float(row['InitialConfidence']),
            "updatedScore": float(row['UpdatedScore']),
            "updatedConfidence": float(row['UpdatedConfidence']),
            "teamId": int(99999)
        }
        agents.append(agent)

    info = {
        "id": int(epid),
        "competitionId": COMPETITION_ID,
        "createTime": {
            "seconds": create_seconds
        },
        "endTime": {
            "seconds": end_seconds
        },
        "agents": agents
    }

    return info
```

```python
def saveEpisode(epid:int, sub_id:int) -> None:
    # request
    re = requests.post(GET_URL, json = {"episodeId": int(epid)})
        
    # save replay
    replay = re.json()
    with open(OUTPUT_DIR / f'{sub_id}_{epid}.json', 'w') as f:
        json.dump(replay, f)
```

```python
start_time = datetime.datetime.now()
episode_count = 0
for _sub_id, df in target_episodes_df.group_by('SubmissionId'):
    sub_id = _sub_id[0]
    ep_ids = df['EpisodeId'].unique()
    for epid in ep_ids:
        saveEpisode(epid, sub_id); 
        episode_count+=1
        try:
            size = os.path.getsize(OUTPUT_DIR / f'{sub_id}_{epid}.json') / 1e6
            print(str(episode_count) + f': saved episode #{epid}')
        except:
            print(f'  file {sub_id}_{epid}.json did not seem to save')

        # process 1 episode/sec
        spend_seconds = (datetime.datetime.now() - start_time).seconds
        if episode_count > spend_seconds:
            time.sleep(episode_count - spend_seconds)
            
        if episode_count > EPISODE_LIMIT_SIZE:
            break 
        
    print(f'Episodes saved: {episode_count}')
```
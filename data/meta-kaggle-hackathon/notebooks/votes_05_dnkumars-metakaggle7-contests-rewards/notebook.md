# MetaKaggle7|Contests & Rewards

- **Author:** Dinesh Naveen Kumar Samudrala
- **Votes:** 28
- **Ref:** dnkumars/metakaggle7-contests-rewards
- **URL:** https://www.kaggle.com/code/dnkumars/metakaggle7-contests-rewards
- **Last run:** 2025-07-20 12:14:43.760000

---

<div style="
    font-family: 'Courier New', monospace; 
    font-size: 24px; 
    font-weight: bold; 
    text-align: center; 
    color: #FFD700; /* Gold text */
    background-color: #1E1E1E; /* Dark gray background */
    padding: 20px; 
    border: 2px solid #FFD700; 
    border-radius: 15px; 
    box-shadow: 0 4px 6px rgba(255, 215, 0, 0.3);">
    ⚡MetaKaggle7|Contests & Rewards⚡
</div>

<p style="font-family: 'Comic Sans MS', serif; font-size: 32px; font-weight: bold; text-align: center; color: #D4EBF8; background-color: #212529; padding: 20px; border: 2px solid #D4EBF8; border-radius: 15px;">Monitoring Kaggle| Analytical Intelligence | Generative Reasoning</p>

<a id="toc"></a>  
<div style="text-align:center; border-radius:30px 30px; padding:7px; color:white; font-size:110%; font-family:Arial; background-color:#5b81d4;">  
  <b>📘 Table of Contents</b>  
</div>  

<ul>  
  <li><a href="#overview">1. 🌐 Overview</a></li>  

  <li><a href="#setup">2. 🧾 Environment Setup, Importing Libraries and Loading Dataset</a>  
    <ul>  
      <li><a href="#setup1">2.1 Importing Required Libraries</a></li>  
      <li><a href="#setup2">2.2 Downloading and Loading Meta-Kaggle Datasets</a></li>  
    </ul>  
  </li>  

  <li><a href="#host-segment">3. 🏢 Competition Hosts Analysis</a>  
    <ul>  
      <li><a href="#host1">3.1 Unique Host Segments</a></li>  
      <li><a href="#host2">3.2 Yearly Competition Count by Host</a></li>  
    </ul>  
  </li>  

  <li><a href="#reward-type">4. 💰 Reward Types in Competitions</a>  
    <ul>  
      <li><a href="#reward1">4.1 Unique Reward Types and Frequency</a></li>  
      <li><a href="#reward2">4.2 Reward Types by Host Segment</a></li>  
    </ul>  
  </li>  

  <li><a href="#usd-trends">5. 📈 USD Reward Trends Over Years</a>  
    <ul>  
      <li><a href="#usd1">5.1 USD Competitions by Year and Host</a></li>  
      <li><a href="#usd2">5.2 Total USD Reward Amounts Over Time</a></li>  
    </ul>  
  </li>  

  <li><a href="#submission-trends">6. 📬 Submission Behavior Analysis</a>  
    <ul>  
      <li><a href="#sub1">6.1 Monthly Submission Trends</a></li>  
      <li><a href="#sub2">6.2 Pre- vs Post-Deadline Submission Analysis</a></li>  
    </ul>  
  </li>  

  <li><a href="#teams">7. 👥 Team Formation Insights</a>  
    <ul>  
      <li><a href="#teams1">7.1 Team Member Registrations by Year</a></li>  
      <li><a href="#teams2">7.2 Average Team Size Trends</a></li>  
    </ul>  
  </li>  
</ul>

<a id="overview"></a>  
# <div style="background-color:#ef476f; color:white; padding:10px; border-radius:10px;">1. 🌐 Overview</div>

> This analysis delves into Kaggle competition dynamics—examining host segments, reward types, USD trends, team behavior, and submission patterns. Using Meta-Kaggle data, it uncovers how competitions are structured, funded, and participated in over time, providing insights into organizational preferences, participant engagement, and evolving collaboration patterns in competitive data science.

<a id="setup"></a>
# <div style="background-color:#cc263b; color:white; padding:10px; border-radius:10px;">2. 🧰 Environment Setup, Importing Libraries and Loading Dataset </div>

<a id="setup1"></a>
## <div style="background-color:#cc263b; color:white; padding:10px; border-radius:10px;">2.1 Importing Required Libraries</div>

```python
import os
import glob
import json
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from IPython.display import IFrame
import kagglehub
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

<a id="setup2"></a>
## <div style="background-color:#cc263b; color:white; padding:10px; border-radius:10px;">2.2 Downloading and Loading Meta-Kaggle Datasets</div>

```python
MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")
print("✅ Downloaded Meta-Kaggle data.")
print("📂 MK_PATH =", MK_PATH)
print("📂 MKC_PATH =", MKC_PATH)
```

<a id="host-segment"></a>
# <div style="background-color:#118ab2; color:white; padding:10px; border-radius:10px;">3. 🏢 Competition Hosts Analysis</div>

```python
competitions = pl.read_csv(f"{MK_PATH}/Competitions.csv")
print("Competitions.csv Columns:", competitions.columns)
print(competitions.shape)
competitions.head()
```

<a id="host1"></a>
## <div style="background-color:#118ab2; color:white; padding:10px; border-radius:10px;">3.1 Unique Host Segments</div>

```python
hosts = (
    competitions
    .filter(pl.col("HostSegmentTitle").is_not_null() & (pl.col("HostSegmentTitle").str.strip_chars() != ""))
    .select("HostSegmentTitle")
    .unique()
    .sort("HostSegmentTitle")
)
host_list = hosts["HostSegmentTitle"].to_list()
print(f"🌍 Total Unique Hosts: {len(host_list)}\n")
for host in host_list:
    print(host)
```

```python
host_counts = (
    competitions
    .filter(
        pl.col("HostSegmentTitle").is_not_null() & 
        (pl.col("HostSegmentTitle").str.strip_chars() != "")
    )
    .group_by("HostSegmentTitle")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)
print("📊 HostSegmentTitle Value Counts:\n")
print(host_counts)
```

```python
competitions = pl.read_csv("/kaggle/input/meta-kaggle/Competitions.csv", try_parse_dates=True)
```

```python
competitions = competitions.with_columns([
    pl.col("EnabledDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("EnabledDateParsed")
])
competitions = competitions.with_columns([
    pl.col("EnabledDateParsed").dt.year().alias("EnabledYear")
])
```

```python
filtered = competitions.filter(
    pl.col("HostSegmentTitle").is_not_null() &
    (pl.col("HostSegmentTitle").str.strip_chars() != "")
)
```

```python
grouped = (
    filtered
    .group_by(["EnabledYear", "HostSegmentTitle"])
    .agg(pl.len().alias("count"))
    .sort(["EnabledYear", "HostSegmentTitle"])
)
grouped
```

```python
df_plot = grouped.to_pandas()
```

<a id="host2"></a>
## <div style="background-color:#118ab2; color:white; padding:10px; border-radius:10px;">3.2 Yearly Competition Count by Host</div>

```python
fig_comparison = go.Figure()
for host in df_plot["HostSegmentTitle"].unique():
    data = df_plot[df_plot["HostSegmentTitle"] == host]
    fig_comparison.add_trace(go.Scatter(
        x=data["EnabledYear"],
        y=data["count"],
        mode="lines+markers",
        name=host
    ))

fig_comparison.update_layout(
    title="Competitions per Host Segment per Year",
    xaxis_title="Year",
    yaxis_title="Count",
    template="plotly_white"
)
fig_comparison.write_html("Competitions_year.html")
IFrame("Competitions_year.html", width=1200, height=700)
```

<a id="reward-type"></a>
# <div style="background-color:#ffd166; color:black; padding:10px; border-radius:10px;">4. 💰 Reward Types in Competitions</div>

<a id="reward1"></a>
## <div style="background-color:#ffd166; color:black; padding:10px; border-radius:10px;">4.1 Unique Reward Types and Frequency</div>

```python
RewardType = (
    competitions
    .filter(pl.col("RewardType").is_not_null() & (pl.col("RewardType").str.strip_chars() != ""))
    .select("RewardType")
    .unique()
    .sort("RewardType")
)
RewardType_list = RewardType["RewardType"].to_list()
print(f"🌍 Total Unique RewardType: {len(RewardType_list)}\n")
for res in RewardType_list:
    print(res)
```

```python
RewardType_counts = (
    competitions
    .filter(pl.col("RewardType").is_not_null() & (pl.col("RewardType").str.strip_chars() != ""))
    .group_by("RewardType")
    .len()
    .sort("len", descending=True)
)
RewardType_counts_list = RewardType_counts.to_numpy().tolist()
print(f"🌍 Total Unique RewardType: {len(RewardType_counts_list)}\n")
for reward_type, count in RewardType_counts_list:
    print(f"{reward_type}: {count}")
```

<a id="reward2"></a>
## <div style="background-color:#ffd166; color:black; padding:10px; border-radius:10px;">4.2 Reward Types by Host Segment</div>

```python
filtered = competitions.filter(
    pl.col("RewardType").is_not_null() & 
    (pl.col("RewardType").str.strip_chars() != "")
)
host_reward_counts = (
    filtered
    .group_by(["HostSegmentTitle", "RewardType"])
    .len()
    .sort(["HostSegmentTitle", "len"], descending=[False, True])
)
print(host_reward_counts)
```

<a id="usd-trends"></a>
# <div style="background-color:#06d6a0; color:white; padding:10px; border-radius:10px;">5. 📈 USD Reward Trends Over Years</div>

<a id="usd1"></a>
## <div style="background-color:#06d6a0; color:white; padding:10px; border-radius:10px;">5.1 USD Competitions by Year and Host</div>

```python
competitions = competitions.with_columns(
    pl.col("EnabledDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("EnabledDate_parsed")
).with_columns(
    pl.col("EnabledDate_parsed").dt.year().alias("Year")
)
```

```python
usd_competitions = competitions.filter(
    (pl.col("RewardType") == "USD") &
    pl.col("Year").is_not_null() &
    pl.col("HostSegmentTitle").is_not_null()
)
```

```python
usd_yearly_counts = (
    usd_competitions
    .group_by(["Year", "HostSegmentTitle"])
    .len()
    .sort(["Year", "HostSegmentTitle"])
)
```

```python
pivoted = (
    usd_yearly_counts
    .pivot(values="len", index="Year", on="HostSegmentTitle")
    .fill_null(0)
    .sort("Year")
)
```

```python
pdf = pivoted.to_pandas()
df_melted = pdf.melt(id_vars="Year", var_name="HostSegmentTitle", value_name="Count")
fig_comparison = px.line(
    df_melted,
    x="Year",
    y="Count",
    color="HostSegmentTitle",
    markers=True,
    title="📈 USD Competitions Over Years by HostSegmentTitle",
    labels={"Count": "Number of Competitions", "Year": "Year"},
    width=1100,
    height=600
)
fig_comparison.write_html("all_compi_comparison.html")
IFrame("all_compi_comparison.html", width=1200, height=700)
```

<a id="usd2"></a>
## <div style="background-color:#06d6a0; color:white; padding:10px; border-radius:10px;">5.2 Total USD Reward Amounts Over Time</div>

```python
competitions = competitions.with_columns(
    pl.col("EnabledDate")
    .str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S", strict=False)
    .alias("EnabledDate_parsed")
).with_columns(
    pl.col("EnabledDate_parsed").dt.year().alias("Year")
)
```

```python
usd_rewards = competitions.filter(
    (pl.col("RewardType") == "USD") &
    (pl.col("RewardQuantity").str.strip_chars().is_not_null()) &
    (pl.col("RewardQuantity").str.strip_chars() != "") &
    pl.col("Year").is_not_null()
).with_columns(
    pl.col("RewardQuantity").str.strip_chars().cast(pl.Float64)
)
```

```python
usd_by_year = (
    usd_rewards
    .group_by("Year")
    .agg(pl.sum("RewardQuantity").alias("TotalUSDReward"))
    .sort("Year")
)
```

```python
df_usd = usd_by_year.to_pandas()

fig_usd = px.line(
    df_usd,
    x="Year",
    y="TotalUSDReward",
    title="💰 Total USD Rewards on Kaggle Per Year",
    labels={"Year": "Year", "TotalUSDReward": "Total USD Reward"},
    markers=True,
    width=1000,
    height=500
)
fig_usd.write_html("usd_reward_by_year.html")
IFrame("usd_reward_by_year.html", width=1200, height=600)
```

<a id="submission-trends"></a>
# <div style="background-color:#073b4c; color:white; padding:10px; border-radius:10px;">6. 📬 Submission Behavior Analysis</div>

```python
Submissions = pl.read_csv("/kaggle/input/meta-kaggle/Submissions.csv")
print(Submissions.columns)
print(Submissions.shape)
Submissions.head()
```

<a id="sub1"></a>
## <div style="background-color:#073b4c; color:white; padding:10px; border-radius:10px;">6.1 Monthly Submission Trends</div>

```python
Submissions = Submissions.with_columns(
    pl.col("SubmissionDate").str.to_datetime("%m/%d/%Y").alias("SubmissionDateDT")
)
submission_trend = Submissions.group_by(pl.col("SubmissionDateDT").dt.truncate("1mo").alias("Month")).agg(
    SubmissionCount=pl.col("Id").count()
).sort("Month")
plt.figure(figsize=(10, 6))
plt.plot(submission_trend["Month"], submission_trend["SubmissionCount"], label="Submission Count", color="blue")
plt.title("Submission Frequency Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Submissions")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<a id="sub2"></a>
## <div style="background-color:#073b4c; color:white; padding:10px; border-radius:10px;">6.2 Pre- vs Post-Deadline Submission Analysis</div>

```python
Submissions = pl.read_csv("/kaggle/input/meta-kaggle/Submissions.csv")
Submissions = Submissions.with_columns(pl.col("SubmissionDate").str.to_date("%m/%d/%Y"))
deadline_trends = Submissions.group_by([
    pl.col("SubmissionDate").dt.truncate("1w").alias("Week"),
    "IsAfterDeadline"
]).agg(
    pl.col("Id").count().alias("SubmissionCount")
).sort("Week")
deadline_pivot = deadline_trends.pivot(
    values="SubmissionCount",
    index="Week",
    on="IsAfterDeadline",
    aggregate_function="sum"
).fill_null(0)
deadline_pivot = deadline_pivot.rename({
    "true": "PostDeadline",
    "false": "PreDeadline"
})
fig = make_subplots()
fig.add_trace(
    go.Scatter(
        x=deadline_pivot["Week"],
        y=deadline_pivot["PreDeadline"],
        name="Pre-Deadline Submissions",
        line=dict(color="blue")
    )
)
fig.add_trace(
    go.Scatter(
        x=deadline_pivot["Week"],
        y=deadline_pivot["PostDeadline"],
        name="Post-Deadline Submissions",
        line=dict(color="red")
    )
)
fig.update_layout(
    title="Submission Trends: Pre- vs. Post-Deadline",
    xaxis_title="Week",
    yaxis_title="Number of Submissions",
    template="plotly",
    showlegend=True
)
pio.write_html(fig, file="deadline_behavior_plot.html", auto_open=False, include_plotlyjs="cdn")
display(IFrame("deadline_behavior_plot.html", width=1200, height=700))
```

```python
import gc
gc.collect()
```

```python
import sys
for name, size in sorted(((name, sys.getsizeof(obj)) for name, obj in globals().items()), key=lambda x: -x[1])[:10]:
    print(f"{name}: {size/1e6:.2f} MB")
```

```python
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
import gc
gc.collect()
```

<a id="teams"></a>
# <div style="background-color:#8338ec; color:white; padding:10px; border-radius:10px;">7. 👥 Team Formation Insights</div>

```python
import polars as pl
from IPython.display import IFrame
import kagglehub
import os
import plotly.express as px
import polars as pl
from pathlib import Path
import json
import glob
import os
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

```python
Teams = pl.read_csv("/kaggle/input/meta-kaggle/Teams.csv")
print(Teams.columns)
print(Teams.shape)
Teams.head()
```

```python
TeamMemberships = pl.read_csv("/kaggle/input/meta-kaggle/TeamMemberships.csv")
print(TeamMemberships.columns)
print(TeamMemberships.shape)
TeamMemberships.head()
```

```python
Teams = pd.read_csv("/kaggle/input/meta-kaggle/Teams.csv")
TeamMemberships = pd.read_csv("/kaggle/input/meta-kaggle/TeamMemberships.csv")
```

```python
def clean_dates(df, date_cols):
    for col in date_cols:
        df[col] = df[col].replace('', pd.NaT)
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df
```

```python
TeamMemberships = clean_dates(TeamMemberships, ['RequestDate'])
TeamMemberships['RequestYear'] = TeamMemberships['RequestDate'].dt.year
```

<a id="teams1"></a>
## <div style="background-color:#8338ec; color:white; padding:10px; border-radius:10px;">7.1 Team Member Registrations by Year</div>

```python
plt.figure(figsize=(12, 8))
team_formation = TeamMemberships.groupby('RequestYear').size().reset_index()
team_formation.columns = ['Year', 'Members']
team_formation = team_formation.dropna()

plt.plot(team_formation['Year'], team_formation['Members'], marker='s', linewidth=3, markersize=8, color='orange')
plt.title('Team Member Registrations by Year', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('New Team Members', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

<a id="teams2"></a>
## <div style="background-color:#8338ec; color:white; padding:10px; border-radius:10px;">7.2 Average Team Size Trends</div>

```python
plt.figure(figsize=(12, 8))
team_sizes = TeamMemberships.groupby(['TeamId', 'RequestYear']).size().reset_index()
team_sizes.columns = ['TeamId', 'Year', 'Size']
team_size_yearly = team_sizes.groupby('Year')['Size'].mean().reset_index()

plt.plot(team_size_yearly['Year'], team_size_yearly['Size'], marker='^', linewidth=3, markersize=8, color='purple')
plt.title('Average Team Size Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Team Size', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

# My MetaKaggle's Notebook Series

Explore a series of notebooks by [dnkumars](https://www.kaggle.com/dnkumars) that dive deep into the MetaKaggle and MetaKaggle Code datasets. Each notebook uncovers different facets of the Kaggle ecosystem, from user demographics to code patterns.

---
### [User Demographics Forecast](https://www.kaggle.com/code/dnkumars/metakaggle-user-demographics-forecast) 📊  
_Delve into trends and forecasts related to the Kaggle user base._

### [Decrypting Datasets](https://www.kaggle.com/code/dnkumars/metakaggle2-decrypting-datasets) 📁  
_Analyze the landscape and metadata of datasets available on Kaggle._

### [Kernels' Crux](https://www.kaggle.com/code/dnkumars/metakaggle3-kernal-s-crux) 🧠  
_Examine kernel (notebook) best practices, popularity, and patterns._

### [Enigmatic Episodes](https://www.kaggle.com/code/dnkumars/metakaggle4-enigmatic-episodes) 📅  
_Explore significant events which have episodes in competitions that have shaped Kaggle (reinforcement learning usage) ._

### [Labels of Recognition](https://www.kaggle.com/code/dnkumars/metakaggle5-labels-of-recogniton) 🏷️  
_Investigate Tags mechanisms and mechanics within Kaggle._

### [Demystifying Code](https://www.kaggle.com/code/dnkumars/metakagglecode-demystifying-code) 🧑‍💻  
_Understand trends, favorite libraries, and patterns in Kaggle code notebooks._

### [Contests & Rewards](https://www.kaggle.com/code/dnkumars/metakaggle7-contests-rewards) 🏆  
_Break down reward structures, competition formats, and highlights from Kaggle contests._

## Citation
\[@meta-kaggle-hackathon]
www.kaggle.com/competitions/meta-kaggle-hackathon/overview/citation

**Citation**: Paul Mooney, Meg Risdal, Maria Cruz, and Addison Howard. *Meta Kaggle Hackathon*. Kaggle. [Kaggle](https://kaggle.com/competitions/meta-kaggle-hackathon)

<div align="center" style="background-color: #C4E1F6; padding: 20px; border-radius: 10px;">
  <h1 style="color: blue;">Thank You 🙇‍♂️ for Visiting My Notebook!</h1>

  <p style="font-size: 18px; color: black;">
    <br>Your feedback is greatly appreciated and motivates me to continue developing more valuable and informative notebooks
  </p>
</div>
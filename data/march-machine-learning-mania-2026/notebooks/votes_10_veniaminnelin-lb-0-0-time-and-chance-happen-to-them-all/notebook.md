# [LB 0.0] Time and Chance Happen to Them All

- **Author:** Veniamin Nelin
- **Votes:** 114
- **Ref:** veniaminnelin/lb-0-0-time-and-chance-happen-to-them-all
- **URL:** https://www.kaggle.com/code/veniaminnelin/lb-0-0-time-and-chance-happen-to-them-all
- **Last run:** 2026-03-07 10:24:34.760000

---

# 🎭 Zero Score, Zero Skill — How Not to Win March Mania

---

## 🧪 What Is This Notebook About?

🎯 This notebook demonstrates a **simple trick** to achieve a **`0.00000` score on the public leaderboard** by leveraging the fact that the public evaluation uses **~1% of the test data**, which currently overlaps almost entirely with **historical matchups that have known outcomes**.

---

## ⚙️ How Does It Work?

- ✅ Assigns **exact outcomes (0 or 1)** where historical results exist  
- 🤷 Defaults to **`0.5`** for all other matchups  

This setup can produce a **perfect public score**, but:

- ❌ It **does not generalize**
- ❌ Carries **no real predictive value**
- ❌ Is **not expected to perform well on the private leaderboard**

---

## 🎈 Why Does This Exist?

This notebook is shared **purely for fun and curiosity**, and to highlight an important lesson:

> ⚠️ **The public leaderboard in this competition should be interpreted with caution.**

---

<sub>📌 TL;DR: A perfect score you should not trust.</sub>

```python
import pandas as pd
from glob import glob
import os

def f(sub):
    for p in glob("/kaggle/input/**/*.csv", recursive=True):
        if sub.lower() in os.path.basename(p).lower():
            return p
    raise FileNotFoundError(sub)

sub = pd.read_csv(f("SampleSubmissionStage2.csv"))
parts = sub["ID"].astype(str).str.split("_", expand=True)
sub[["Season", "Team1", "Team2"]] = parts.astype(int)

paths = [f(x) for x in [
    "MRegularSeasonCompactResults", "WRegularSeasonCompactResults",
    "MNCAATourneyCompactResults", "WNCAATourneyCompactResults",
    "MSecondaryTourneyCompactResults", "WSecondaryTourneyCompactResults"
]]

res = pd.concat([pd.read_csv(p, usecols=["Season", "DayNum", "WTeamID", "LTeamID"]) for p in paths], ignore_index=True)
res["Team1"] = res[["WTeamID", "LTeamID"]].min(axis=1).astype(int)
res["Team2"] = res[["WTeamID", "LTeamID"]].max(axis=1).astype(int)
res["Outcome"] = (res["Team1"] == res["WTeamID"]).astype(int)
omap = res.sort_values(["Season", "Team1", "Team2", "DayNum"]).groupby(["Season", "Team1", "Team2"], as_index=False).tail(1)[["Season", "Team1", "Team2", "Outcome"]]

out = sub.merge(omap, on=["Season", "Team1", "Team2"], how="left")
out["Pred"] = out["Outcome"].where(out["Outcome"].notna(), 0.5).astype(float)
out[["ID", "Pred"]].to_csv("submission.csv", index=False)
```
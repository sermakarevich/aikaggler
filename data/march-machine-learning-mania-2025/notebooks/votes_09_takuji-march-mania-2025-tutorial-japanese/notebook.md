# March Mania 2025 / Tutorial [Japanese]

- **Author:** esprit
- **Votes:** 156
- **Ref:** takuji/march-mania-2025-tutorial-japanese
- **URL:** https://www.kaggle.com/code/takuji/march-mania-2025-tutorial-japanese
- **Last run:** 2025-02-20 17:24:07.943000

---

# はじめに
このコンペは、**これから行われるバスケの試合の勝敗を予測するコンペ**です。

個人的にバスケの分析についてもっと知りたいと思い、自分の勉強も兼ねてチュートリアルを書くことにしました。バスケコンペだけでなく、他のいろいろなテーブルコンペに参加するときにも参考になりましたら幸いです。

# Part 1. コンペデータを見る
まずはコンペデータをざっと眺めてみます。詳しい内容や使い方はあとのパートで調べるので、ここではざっと眺めるだけにしておきます。

```python
import os
import warnings
import pandas as pd

warnings.simplefilter('ignore')
pd.set_option('display.max_columns', None)

for dirname, _, filenames in os.walk('/kaggle/input'):
    if dirname != "/kaggle/input/march-machine-learning-mania-2025":
        continue
    filenames = sorted(filenames)
    print("コンペデータのCSVの数は", len(filenames))
    cnt = 0
    for filename in filenames:
        cnt += 1
        print("*" * 50)
        try:
            df = pd.read_csv(os.path.join(dirname, filename))
        except:
            df = pd.read_csv(os.path.join(dirname, filename), encoding='cp1252')
        print(f"No.{cnt}", filename, df.shape)
        display(df.head(3))
```

結構ありますね。ファイル名が"M"から始まるものが男子のデータ、"W"から始まるものが女子のデータのようです。どのCSVをどう使えばよいかは、あとのパートで調べることにします。

続いて、コンペに必要なドメイン知識を押さえましょう。

# Part 2. ドメイン知識を知る
March Maniaはアメリカで毎年行われる大学バスケットボールの大会です。ドメイン知識があると解法が分かりやすくなるので、ざっと紹介します。

## カンファレンス
地域リーグのようなもの。原則1つのチームが所属するカンファレンスは1つだけ。カンファレンスごとのチーム数は8や16、18など様々。

## レギュラーシーズン
同一カンファレンス内で対戦が組まれる「カンファレンス戦」と、そうでない「非カンファレンス戦」の2つに分かれる。
1. **非カンファレンス戦**  
   シーズンの最初（11月〜12月）に行われる。
   
2. **カンファレンス戦**  
   シーズン後半（1月〜3月初旬）に行われる。ホーム＆アウェイ形式が基本。**カンファレンス戦の成績が、カンファレンストーナメントのシード順に影響する。**

## トーナメント
主にカンファレンストーナメント、NCAAトーナメント、セカンダリートーナメントの3種類がある。
1. **カンファレンストーナメント**  
   カンファレンス（地域リーグ）ごとに開催されるトーナメント。３月上旬に行われる。**優勝チームは自動的にNCAAトーナメントに出場できる。**

2. **★重要★ NCAAトーナメント**  
   通称「**March Madness**」。**このコンペで予測するのは、このNCAAトーナメントの勝敗。**
   
   March Madnessには**男女それぞれ68チームずつが出場し、負けたら即敗退**。トーナメントは大きく4つの地域（East、West、South、Midwest）に分かれており、各地域からファイナル4（準決勝）進出チームが決まる。
   
   March Madnessへの出場枠は次の2種類で決まる。
   1. **オートマチック・ビッド（自動出場枠）**  
      カンファレンストーナメントの優勝チーム（31チーム）
      
   2. **アットラージ・ビッド（選考出場枠）**  
      レギュラーシーズンの結果から、選考委員会が決定する37チーム

   March Madnessの最初の4試合を特に**First Four**と呼ぶ。これは、全68チーム中最もシードの低い8チームが4試合を戦い、勝ち上がった4チームが正式に64チームによる本トーナメントに進出するというもの。
  
4. **セカンダリートーナメント**  
   March Madnessに出場できなかったチームのための大会。「NIT」、「CBI」、「The Basketball Classsic」などの大会がある。出場チーム数は大会によって様々。

**コンペデータはコンペ締め切り日まで随時更新されます。**（レギュラーシーズンもカンファレンストーナメントもコンペ期間中に終了するため。）

# Part 3. March Mania 2023の1位解法を読む
March Mania 2025のコンペ内容は、**March Mania 2023とかなり似ています**。2025のコンペデータは、2023と比べてテーブル数がいくつか増えています（どのテーブルかはあとのパートで見ることにします）。他のテーブルについては同じ構造です。評価指標も同じです。したがって、March Mania 2023の1位解法を読めば大筋が掴めるはずです。
- リファレンス：[March Mania 2023 1位解法のノートブック](https://www.kaggle.com/code/rustyb/paris-madness-2023)

では読んでみましょう。

## コンペデータの読み込み
たくさんあったコンペデータのうち、どのCSVを使っているかに注目します。

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm
import matplotlib.pyplot as plt

DATA_PATH = '/kaggle/input/march-machine-learning-mania-2023/'

# 使うCSVは6つ。
# マーチマッドネスの試合結果。これは過去のデータしかないはず
tourney_results = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneyDetailedResults.csv"),
    pd.read_csv(DATA_PATH + "WNCAATourneyDetailedResults.csv"),
], ignore_index=True)

# マーチマッドネスのシード情報。これはコンペ期間中にその年の情報が手に入っていたはず
seeds = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv"),
    pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv"),
], ignore_index=True)

# シーズンの試合結果。これもコンペ期間中にその年の情報が手に入っていたはず
regular_results = pd.concat([
    pd.read_csv(DATA_PATH + "MRegularSeasonDetailedResults.csv"),
    pd.read_csv(DATA_PATH + "WRegularSeasonDetailedResults.csv"),
], ignore_index=True)
```

使っているCSVは6つでした。コンペデータを全部使わなくても戦えるようです。

## 特徴量の作成
続いて、どのような特徴量が作られていたかや、特徴量の数はどれくらいだったかなどを見てみましょう。

```python
def prepare_data(df):
    # 勝者と敗者をスワップしたものを用意。
    # 内容は同じでもデータ量は2倍に
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 
    'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 
    'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]

    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    df.columns.values[6] = 'location'
    dfswap.columns.values[6] = 'location'    
      
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]

    # 元データとスワップしたものを結合
    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output.loc[output.location=='N','location'] = '0'
    output.loc[output.location=='H','location'] = '1'
    output.loc[output.location=='A','location'] = '-1'
    output.location = output.location.astype(int)
    
    output['PointDiff'] = output['T1_Score'] - output['T2_Score']
    
    return output

regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)

# 特徴量作成を開始。
# まずはシーズンのゲームスタッツ系の特徴量
boxscore_cols = [
        'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 
        'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',  
        'PointDiff']

funcs = [np.mean]

season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(funcs).reset_index()

season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]

season_statistics_T1 = season_statistics.copy()
season_statistics_T2 = season_statistics.copy()

season_statistics_T1.columns = ["T1_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T1.columns)]
season_statistics_T2.columns = ["T2_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T2.columns)]
season_statistics_T1.columns.values[0] = "Season"
season_statistics_T2.columns.values[0] = "Season"

# マーチマッドネスデータの右に、作った特徴量をくっつける。
# マーチマッドネスのスタッツ情報は残念ながら削除。その年の情報を得ることができないため、使い道がない
tourney_data = tourney_data[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID' ,'T2_Score']]

tourney_data = pd.merge(tourney_data, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')

# 直近14日間の勝率
last14days_stats_T1 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff']>0,1,0)
last14days_stats_T1 = last14days_stats_T1.groupby(['Season','T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

last14days_stats_T2 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff']<0,1,0)
last14days_stats_T2 = last14days_stats_T2.groupby(['Season','T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')

# これもマーチマッドネスデータの右にくっつける
tourney_data = pd.merge(tourney_data, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')

# チームクオリティという特徴量を作る
regular_season_effects = regular_data[['Season','T1_TeamID','T2_TeamID','PointDiff']].copy()
regular_season_effects['T1_TeamID'] = regular_season_effects['T1_TeamID'].astype(str)
regular_season_effects['T2_TeamID'] = regular_season_effects['T2_TeamID'].astype(str)
regular_season_effects['win'] = np.where(regular_season_effects['PointDiff']>0,1,0)
march_madness = pd.merge(seeds[['Season','TeamID']],seeds[['Season','TeamID']],on='Season')
march_madness.columns = ['Season', 'T1_TeamID', 'T2_TeamID']
march_madness.T1_TeamID = march_madness.T1_TeamID.astype(str)
march_madness.T2_TeamID = march_madness.T2_TeamID.astype(str)
regular_season_effects = pd.merge(regular_season_effects, march_madness, on = ['Season','T1_TeamID','T2_TeamID'])

def team_quality(season):
    formula = 'win~-1+T1_TeamID+T2_TeamID'
    glm = sm.GLM.from_formula(formula=formula, 
                              data=regular_season_effects.loc[regular_season_effects.Season==season,:], 
                              family=sm.families.Binomial()).fit()
    
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID','quality']
    quality['Season'] = season
    #quality['quality'] = np.exp(quality['quality'])
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
    return quality

formula = 'win~-1+T1_TeamID+T2_TeamID'
glm = sm.GLM.from_formula(formula=formula, 
                          data=regular_season_effects.loc[regular_season_effects.Season==2010,:], 
                          family=sm.families.Binomial()).fit()

quality = pd.DataFrame(glm.params).reset_index()

glm_quality = pd.concat([team_quality(2010),
                         team_quality(2011),
                         team_quality(2012),
                         team_quality(2013),
                         team_quality(2014),
                         team_quality(2015),
                         team_quality(2016),
                         team_quality(2017),
                         team_quality(2018),
                         team_quality(2019),
                         ##team_quality(2020),
                         team_quality(2021),
                         team_quality(2022),
                         team_quality(2023)
                         ]).reset_index(drop=True)

# マーチマッドネスの右に
glm_quality_T1 = glm_quality.copy()
glm_quality_T2 = glm_quality.copy()
glm_quality_T1.columns = ['T1_TeamID','T1_quality','Season']
glm_quality_T2.columns = ['T2_TeamID','T2_quality','Season']

tourney_data = pd.merge(tourney_data, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')

# シードナンバーも特徴量
seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))

# マーチマッドネスの右に
seeds_T1 = seeds[['Season','TeamID','seed']].copy()
seeds_T2 = seeds[['Season','TeamID','seed']].copy()
seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
seeds_T2.columns = ['Season','T2_TeamID','T2_seed']

tourney_data = pd.merge(tourney_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]

# できあがった特徴量の数をチェック
features = list(season_statistics_T1.columns[2:999]) + \
    list(season_statistics_T2.columns[2:999]) + \
    list(seeds_T1.columns[2:999]) + \
    list(seeds_T2.columns[2:999]) + \
    list(last14days_stats_T1.columns[2:999]) + \
    list(last14days_stats_T2.columns[2:999]) + \
    ["Seed_diff"] + ["T1_quality","T2_quality"]

print("特徴量の数は", len(features))

# できあがった特徴量はこんな感じ
display(tourney_data[features].head(3))
print("\nfeatures shape:", tourney_data[features].shape)
```

1位解法では45個の特徴量を作っていたようです。案外少ない気もしますが、厳選された特徴量と考えることもできます。次はモデルのトレーニングです。

## XGBoost Modelのトレーニング
予測モデルの構築をします。1位解法では、**得点差を予測するXGBoost Model**と、**得点差を勝つ確率に変換するSpline Model**の2つに分けていたみたいです。まずXGBoost Modelから見ていきます。

```python
# 勝敗を直接予測するのでなく、得点差を予測しにいくようだ。
# つまり、得点差を勝敗に変換するためのモデルもあとで作ることになる
y = tourney_data['T1_Score'] - tourney_data['T2_Score']

X = tourney_data[features].values
dtrain = xgb.DMatrix(X, label = y)

# なにこれ？
def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000 
    x =  preds-labels    
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess

param = {} 
#param['objective'] = 'reg:linear'
param['eval_metric'] =  'mae'
param['booster'] = 'gbtree'
param['eta'] = 0.05 #change to ~0.02 for final run
param['subsample'] = 0.35
param['colsample_bytree'] = 0.7
param['num_parallel_tree'] = 3 #recommend 10
param['min_child_weight'] = 40
param['gamma'] = 10
param['max_depth'] =  3
param['silent'] = 1

# repeat_cvは推論に使うモデルの数。
# 各モデルで適正なイテレーション数を調べることをしているみたい
xgb_cv = []
repeat_cv = 3 # recommend 10

for i in range(repeat_cv): 
    print(f"Fold repeater {i}")
    xgb_cv.append(
        xgb.cv(
          params = param,
          dtrain = dtrain,
          obj = cauchyobj,
          num_boost_round = 3000,
          folds = KFold(n_splits = 5, shuffle = True, random_state = i),
          early_stopping_rounds = 25,
          verbose_eval = 50
        )
    )

iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]

# バリデーションデータの予測値は取っておく。
# 後続の変換モデルを構築する際に使います
oof_preds = []
for i in range(repeat_cv):
    preds = y.copy()
    kfold = KFold(n_splits = 5, shuffle = True, random_state = i)    
    for train_index, val_index in kfold.split(X,y):
        dtrain_i = xgb.DMatrix(X[train_index], label = y[train_index])
        dval_i = xgb.DMatrix(X[val_index], label = y[val_index])  
        model = xgb.train(
              params = param,
              dtrain = dtrain_i,
              num_boost_round = iteration_counts[i],
              verbose_eval = 50
        )
        preds[val_index] = model.predict(dval_i)
    oof_preds.append(np.clip(preds,-30,30))
```

これでXGBoost Modelが3つ完成。

## Spline Modelの構築
スプラインは、得点差（予測値）を横軸、勝つ確率（グランドトゥルース）を縦軸に取って近似曲線を引くようなイメージ。

```python
# UnivariateSplineでモデル化。
# XGBoost Modelsの数と同じだけ作っている
spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))

# スプラインを可視化するとこんな感じ
plot_df = pd.DataFrame({"pred":oof_preds[0], "label":np.where(y>0,1,0), "spline":spline_model[0](oof_preds[0])})
plot_df["pred_int"] = (plot_df["pred"]).astype(int)
plot_df = plot_df.groupby('pred_int').mean().reset_index()

plt.figure(figsize=[5.3,3.0])
plt.plot(plot_df.pred_int,plot_df.spline)
plt.plot(plot_df.pred_int,plot_df.label)
```

スプライン曲線の両端が少し反転してしまっているのがちょっと気になりますね。モデルが全て用意できたので、最後に推論です。

## 推論
テストデータに対しても同様に特徴量を作り、モデルにインプットするという流れになります。

```python
# テストデータの準備。
# トレーニングデータは〜2022、テストデータは2023のものです
sub = pd.read_csv(DATA_PATH + "SampleSubmission2023.csv")
sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))
sub["T1_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[1]))
sub["T2_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[2]))

# 特徴量を結合。
# 2023の特徴量も作成済みなので、右にくっつければOK
sub = pd.merge(sub, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub = pd.merge(sub, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub = pd.merge(sub, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub = pd.merge(sub, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub["Seed_diff"] = sub["T1_seed"] - sub["T2_seed"]

# データセットの形にする
Xsub = sub[features].values
dtest = xgb.DMatrix(Xsub)

# バリデーションなしの全トレーニングデータでトレーニングした方がより精度が出るはずなので、再度トレーニングし直している
sub_models = []
for i in range(repeat_cv):
    sub_models.append(
        xgb.train(
          params = param,
          dtrain = dtrain,
          num_boost_round = int(iteration_counts[i] * 1.05),
          verbose_eval = 50
        )
    )

# テストデータで推論。
# XGBoostで得点差を予測し、Splineで勝つ確率に変換
sub_preds = []
for i in range(repeat_cv):
    sub_preds.append(np.clip(spline_model[i](np.clip(sub_models[i].predict(dtest),-30,30)),0.025,0.975))
    
sub["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)
sub[['ID','Pred']].to_csv("submission.csv", index = None)

display(sub[['ID','Pred']].head(3))
```

以上で提出ファイル完成です。どうやら最後に出てきたPredの値が[March Mania 2023 1位解法のノートブック](https://www.kaggle.com/code/rustyb/paris-madness-2023)と違うのですが、これはxgboostのバージョン差異によるものと思われます（xgboost==0.81 -> 2.0.3）。同様に**提出後のスコアも変わるので注意が必要です**。ライブラリのバージョンは、ノートブックのedit画面右側のタブの**ENVIRONMENT**の設定によって変わります。

また、1位解法だけでなく他の上位解法についてもDiscussionなどに公開されているので参考にするとよいでしょう。

# Part 4. テーブルの意味を知る
March Mania 2025では、March Mania 2023にはなかった新しいテーブルが追加されています。ただ、March Mania 2025でこれらのテーブルが活かせるかどうかは今のところちょっとよく分かりません。（SeedBenchmarkStage2.csvについて情報が欲しいところです。）
- SeedBenchmarkStage1.csv
- WConferenceTourneyGames.csv
- WSecondaryTourneyCompactResults.csv
- WSecondaryTourneyTeams.csv

以下、コンペデータの中から知っておくとよさそうな内容をピックアップしてみます。

## M/WSeasons.csv
- DayZero列: 各年度の基準日。いくつかのテーブルに出てくるDayNum列は、このDayZeroを0としての日数。
- RegionW/X/Y/Z列: ファイナル4（準決勝）での対戦を決めるためのもの。つまりファイナル4ではRegionW vs RegionX、RegionY vs RegionZ、という対戦が組まれる。

## ****Slots.csv
- Slot列: トーナメント上の各対戦に**スロット**というのが割り当てられている。例えば「R1W1」というスロットは、「1回戦 West regionの第1シード」と読む。また「X16」のような3桁のスロットは、NCAAトーナメントの一番最初に行われるFirst Four用のスロット。

## ****CompactResults.csv
- W: 勝ちチーム
- L: 負けチーム
- Loc: 試合会場のホーム/アウェイ/中立
- NumOT: 延長戦のピリオド数

## ****DetailedResults.csv
- FGM: Field-Goal-Made
- FGA: Field-Goal-Attempt
- （数が多いので中略）
- PF: Penalty-Foul

## M/WMasseyOrdinals.csv
- Massey Ordinals: 数十種類のランキングシステムを統合した総合ランキング。NCAA選考委員会がチームの実力を判断する際の参考指標の一つとされている。アットラージ・ビッドの選考やシード順位決定の際に考慮される可能性がある。

チュートリアルはこれでおしまいです。楽しみましょう！
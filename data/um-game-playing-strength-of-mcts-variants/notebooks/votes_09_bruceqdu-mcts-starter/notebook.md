# MCTS Starter

- **Author:** Bruce
- **Votes:** 201
- **Ref:** bruceqdu/mcts-starter
- **URL:** https://www.kaggle.com/code/bruceqdu/mcts-starter
- **Last run:** 2024-11-03 14:13:03.560000

---

## Created by <a href="https://github.com/yunsuxiaozi/">yunsuxiaozi </a>  2024/10/18

#### 1.<a href="https://www.kaggle.com/konstantinboyko">Konstantin Boiko </a>,Thank you for finding an error in my notebook.<a href="https://www.kaggle.com/code/konstantinboyko/mcts-starter-lb-0-431-or-not">MCTS Starter LB=0.431 OR NOT?</a>

#### 2.<a href="https://www.kaggle.com/code/litsea/mcts-baseline-fe-lgbm">MCTS baseline fe lgbm</a>,I have referred to some of his Feature Engineer here.

- v1:CV:0.4245,LB:0.454

- v2:sample+folds:CV:0.4218,LB:0.455

- v3:fusion model:CV:0.4182,LB:0.441

- v4:cross+noise:CV:0.4186,LB:0.440

- v5:without xgb:CV:0.4167,LB:0.438

- v6:StratifiedGroupKFold+skew feature:CV:0.4232,LB:0.440

- v7:deal with outliers+clip:CV:0.4274,LB:0.447

- v8:area feature+cat_folds15+lgb_folds10:CV:0.4191,LB:0.435

- v9:drop area:CV:0.4184,LB:0.435

- v10:drop_row_equal_col:CV:0.4181,LB:0.431

- v11,v12:errors

- v13:fix bug:CV:0.42145,LB:0.438

- v14:2 catboost+new feature+drop  agent1,agent2 tfidf:CV:0.4108,LB:0.429

- v15:5num_folds\*2catboost + agent positive feature + update drop_cols + Extract rules semantic information + minimize RMSE(w\* oof+b,target) :

- v16:english comment just quick save:

# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">Import Libraries</h1></span>

```python
#necessary
import polars as pl#similar to pandas, but with better performance when dealing with large datasets.
import pandas as pd#read csv,parquet
import numpy as np#for scientific computation of matrices
#kfold
from sklearn.model_selection import StratifiedGroupKFold
#models(lgb,xgb,cat)
from  lightgbm import LGBMRegressor,log_evaluation,early_stopping
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import dill#serialize and deserialize objects (such as saving and loading tree models)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer#word2vec feature
import re#python's built-in regular expressions.
import gc#rubbish collection
import warnings#avoid some negligible errors
#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')

import random#provide some function to generate random_seed.
#set random seed,to make sure model can be recurrented.
def seed_everything(seed):
    np.random.seed(seed)#numpy's random seed
    random.seed(seed)#python built-in random seed
seed_everything(seed=2024)
```

# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">Read Data</h1></span>

```python
train=pl.read_csv("/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv")
train=train.to_pandas()
print(f"len(train):{len(train)}")
test=pl.read_csv("/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv")
test=test.to_pandas()
print(f"len(test):{len(test)}")
test.head()
```

# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">Preprocessor</h1></span>

### Feature engineering, model training, and inference are all written in this class.

####  1.Based on the description on the data introduction page, the features of agent1 and agent2 were constructed.

>Agent String Descriptions
All agent string descriptions in training and test data are in the following format: MCTS-<SELECTION>-<EXPLORATION_CONST>-<PLAYOUT>-<SCORE_BOUNDS>,where:
<SELECTION> is one of: UCB1, UCB1GRAVE, ProgressiveHistory, UCB1Tuned. These are different strategies that may be used within the Selection phase of the MCTS algorithm.   
EXPLORATION_CONST is one of: 0.1, 0.6, 1.41421356237. These are three different values that we have tested for the "exploration constant" (a numeric hyperparameter shared among all of the tested Selection strategies).   
PLAYOUT is one of: Random200, MAST, NST. These are different strategies that may be used within the Play-out phase of the MCTS algorithm.   
SCORE_BOUNDS is one of: true or false, indicating whether or not a "Score-Bounded" version of MCTS (a version of the algorithm that can prove when certain nodes in its search tree are wins/losses/draws under perfect play, and adapt the search accordingly).   
For example, an MCTS agent that uses the UCB1GRAVE selection strategy, an exploration constant of 0.1, the NST play-out strategy, and Score Bounds, will be described as MCTS-UCB1GRAVE-0.1-NST-true.
       
#### 2.onehot and tfidf is simple.To prevent data leakage, separate tfidf was performed for each training and validation set in cross validation.
    
#### 3.Removed some useless features from manual selection and model selection.

```python
class Preprocessor():
    def __init__(self,seed=2024,target='utility_agent1',train=None,num_folds=10,CV_LB_path="/kaggle/input/mcts-eda-about-cv-and-lb/1018CV_LB.csv"):
        self.seed=seed
        self.target=target
        self.train=train
        self.model_paths=[('cat1', 0),
                          ('cat1', 1),
                          ('cat1', 2),
                          ('cat1', 3),
                          ('cat1', 4),
                          ('cat2', 0),
                          ('cat2', 1),
                          ('cat2', 2),
                          ('cat2', 3),
                          ('cat2', 4)]#train and inference model
        self.tfidf_paths=[('cat1', 0, 'EnglishRules'),
                          ('cat1', 0, 'LudRules'),
                          ('cat1', 1, 'EnglishRules'),
                          ('cat1', 1, 'LudRules'),
                          ('cat1', 2, 'EnglishRules'),
                          ('cat1', 2, 'LudRules'),
                          ('cat1', 3, 'EnglishRules'),
                          ('cat1', 3, 'LudRules'),
                          ('cat1', 4, 'EnglishRules'),
                          ('cat1', 4, 'LudRules'),
                          ('cat2', 0, 'EnglishRules'),
                          ('cat2', 0, 'LudRules'),
                          ('cat2', 1, 'EnglishRules'),
                          ('cat2', 1, 'LudRules'),
                          ('cat2', 2, 'EnglishRules'),
                          ('cat2', 2, 'LudRules'),
                          ('cat2', 3, 'EnglishRules'),
                          ('cat2', 3, 'LudRules'),
                          ('cat2', 4, 'EnglishRules'),('cat2', 4, 'LudRules')] #tfidf models paths
        self.num_folds=num_folds
        #check CV and LB 
        #self.check=pd.read_csv(CV_LB_path)
        
    #clean str columns
    def clean(self,df,col):
        #fillna with 'nan'
        df[col]=df[col].fillna("nan")
        #string lower
        df[col]=df[col].apply(lambda x:x.lower())
        # think about ‘MCTS-UCB1-0.6-NST-false‘
        ps='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        for p in ps:
            df[col]=df[col].apply(lambda x:x.replace(p,' '))
        return df
    
    def ARI(self,txt):
        characters=len(txt)
        words=len(re.split(' |\\n|\\.|\\?|\\!|\,',txt))
        sentence=len(re.split('\\.|\\?|\\!',txt))
        ari_score=4.71*(characters/words)+0.5*(words/sentence)-21.43
        return ari_score
    """
    http://www.supermagnus.com/mac/Word_Counter/index.html
    McAlpine EFLAW© Test
         (W + SW) / S
    McAlpine EFLAW© Readability
         Scale:
         1-20: Easy
         21-25: Quite Easy
         26-29: Mildly Difficult
         ≥ 30: Very Confusing
         S:total sentences
         W:total words
    """
    def McAlpine_EFLAW(self,txt):
        W=len(re.split(' |\\n|\\.|\\?|\\!|\,',txt))
        S=len(re.split('\\.|\\?|\\!',txt))
        mcalpine_eflaw_score=(W+S*W)/S
        return mcalpine_eflaw_score
    """
    https://readable.com/readability/coleman-liau-readability-index/

    =0.0588*L-0.296*S-15.8
    """
    def CLRI(self,txt):
        characters=len(txt)
        words=len(re.split(' |\\n|\\.|\\?|\\!|\,',txt))
        sentence=len(re.split('\\.|\\?|\\!',txt))
        L=100*characters/words
        S=100*sentence/words
        clri_score=0.0588*L-0.296*S-15.8
        return clri_score
        
    #save models after training
    def pickle_dump(self,obj, path):
        #open path,binary write
        with open(path, mode="wb") as f:
            dill.dump(obj, f, protocol=4)
    #load models when inference
    def pickle_load(self,path):
        #open path,binary read
        with open(path, mode="rb") as f:
            data = dill.load(f)
            return data
    
    #reduce df memory
    def reduce_mem_usage(self,df, float16_as32=True):
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object and str(col_type)!='category':
                c_min,c_max = df[col].min(),df[col].max()
                if str(col_type)[:3] == 'int':
                    
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:#float
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        if float16_as32:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float16)  
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        #calculate memory usage after optimization
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df
        
    def FE(self,df,mode='train'):
        print(f"FE:{mode}")

        print("agent position feature")
        #agent is positive or negative
        total_agent=['MCTS-ProgressiveHistory-0.1-MAST-false', 'MCTS-ProgressiveHistory-0.1-MAST-true', 'MCTS-ProgressiveHistory-0.1-NST-false', 'MCTS-ProgressiveHistory-0.1-NST-true', 'MCTS-ProgressiveHistory-0.1-Random200-false', 'MCTS-ProgressiveHistory-0.1-Random200-true', 'MCTS-ProgressiveHistory-0.6-MAST-false', 'MCTS-ProgressiveHistory-0.6-MAST-true', 'MCTS-ProgressiveHistory-0.6-NST-false', 'MCTS-ProgressiveHistory-0.6-NST-true', 'MCTS-ProgressiveHistory-0.6-Random200-false', 'MCTS-ProgressiveHistory-0.6-Random200-true', 'MCTS-ProgressiveHistory-1.41421356237-MAST-false', 'MCTS-ProgressiveHistory-1.41421356237-MAST-true', 'MCTS-ProgressiveHistory-1.41421356237-NST-false', 'MCTS-ProgressiveHistory-1.41421356237-NST-true', 'MCTS-ProgressiveHistory-1.41421356237-Random200-false', 'MCTS-ProgressiveHistory-1.41421356237-Random200-true', 'MCTS-UCB1-0.1-MAST-false', 'MCTS-UCB1-0.1-MAST-true', 'MCTS-UCB1-0.1-NST-false', 'MCTS-UCB1-0.1-NST-true', 'MCTS-UCB1-0.1-Random200-false', 'MCTS-UCB1-0.1-Random200-true', 'MCTS-UCB1-0.6-MAST-false', 'MCTS-UCB1-0.6-MAST-true', 'MCTS-UCB1-0.6-NST-false', 'MCTS-UCB1-0.6-NST-true', 'MCTS-UCB1-0.6-Random200-false', 'MCTS-UCB1-0.6-Random200-true', 'MCTS-UCB1-1.41421356237-MAST-false', 'MCTS-UCB1-1.41421356237-MAST-true', 'MCTS-UCB1-1.41421356237-NST-false', 'MCTS-UCB1-1.41421356237-NST-true', 'MCTS-UCB1-1.41421356237-Random200-false', 'MCTS-UCB1-1.41421356237-Random200-true', 'MCTS-UCB1GRAVE-0.1-MAST-false', 'MCTS-UCB1GRAVE-0.1-MAST-true', 'MCTS-UCB1GRAVE-0.1-NST-false', 'MCTS-UCB1GRAVE-0.1-NST-true', 'MCTS-UCB1GRAVE-0.1-Random200-false', 'MCTS-UCB1GRAVE-0.1-Random200-true', 'MCTS-UCB1GRAVE-0.6-MAST-false', 'MCTS-UCB1GRAVE-0.6-MAST-true', 'MCTS-UCB1GRAVE-0.6-NST-false', 'MCTS-UCB1GRAVE-0.6-NST-true', 'MCTS-UCB1GRAVE-0.6-Random200-false', 'MCTS-UCB1GRAVE-0.6-Random200-true', 'MCTS-UCB1GRAVE-1.41421356237-MAST-false', 'MCTS-UCB1GRAVE-1.41421356237-MAST-true', 'MCTS-UCB1GRAVE-1.41421356237-NST-false', 'MCTS-UCB1GRAVE-1.41421356237-NST-true', 'MCTS-UCB1GRAVE-1.41421356237-Random200-false', 'MCTS-UCB1GRAVE-1.41421356237-Random200-true', 'MCTS-UCB1Tuned-0.1-MAST-false', 'MCTS-UCB1Tuned-0.1-MAST-true', 'MCTS-UCB1Tuned-0.1-NST-false', 'MCTS-UCB1Tuned-0.1-NST-true', 'MCTS-UCB1Tuned-0.1-Random200-false', 'MCTS-UCB1Tuned-0.1-Random200-true', 'MCTS-UCB1Tuned-0.6-MAST-false', 'MCTS-UCB1Tuned-0.6-MAST-true', 'MCTS-UCB1Tuned-0.6-NST-false', 'MCTS-UCB1Tuned-0.6-NST-true', 'MCTS-UCB1Tuned-0.6-Random200-false', 'MCTS-UCB1Tuned-0.6-Random200-true', 'MCTS-UCB1Tuned-1.41421356237-MAST-false', 'MCTS-UCB1Tuned-1.41421356237-MAST-true', 'MCTS-UCB1Tuned-1.41421356237-NST-false', 'MCTS-UCB1Tuned-1.41421356237-NST-true', 'MCTS-UCB1Tuned-1.41421356237-Random200-false', 'MCTS-UCB1Tuned-1.41421356237-Random200-true']
        agent1,agent2=df['agent1'].values,df['agent2'].values
        for i in range(len(total_agent)):
            value=np.zeros(len(df))
            for j in range(len(df)):
                if agent1[j]==total_agent[i]:
                    value[j]+=1
                elif agent2[j]==total_agent[i]:
                    value[j]-=1
            df[f'agent_{total_agent[i]}']=value

        
        df['area']=df['NumRows']*df['NumColumns']
        df['row_equal_col']=(df['NumColumns']==df['NumRows']).astype(np.int8)
        df['Playouts/Moves'] = df['PlayoutsPerSecond'] / (df['MovesPerSecond'] + 1e-15)
        df['EfficiencyPerPlayout'] = df['MovesPerSecond'] / (df['PlayoutsPerSecond'] + 1e-15)
        df['TurnsDurationEfficiency'] = df['DurationActions'] / (df['DurationTurnsStdDev'] + 1e-15)
        df['AdvantageBalanceRatio'] = df['AdvantageP1'] / (df['Balance'] + 1e-15)
        df['ActionTimeEfficiency'] = df['DurationActions'] / (df['MovesPerSecond'] + 1e-15)
        df['StandardizedTurnsEfficiency'] = df['DurationTurnsStdDev'] / (df['DurationActions'] + 1e-15)
        df['AdvantageTimeImpact'] = df['AdvantageP1'] / (df['DurationActions'] + 1e-15)
        df['DurationToComplexityRatio'] = df['DurationActions'] / (df['StateTreeComplexity'] + 1e-15)
        df['NormalizedGameTreeComplexity'] =  df['GameTreeComplexity'] /  (df['StateTreeComplexity'] + 1e-15)
        df['ComplexityBalanceInteraction'] =  df['Balance'] *  df['GameTreeComplexity']
        df['OverallComplexity'] =  df['StateTreeComplexity'] +  df['GameTreeComplexity']
        df['ComplexityPerPlayout'] =  df['GameTreeComplexity'] /  (df['PlayoutsPerSecond'] + 1e-15)
        df['TurnsNotTimeouts/Moves'] = df['DurationTurnsNotTimeouts'] / (df['MovesPerSecond'] + 1e-15)
        df['Timeouts/DurationActions'] = df['Timeouts'] / (df['DurationActions'] + 1e-15)
        df['OutcomeUniformity/AdvantageP1'] = df['OutcomeUniformity'] / (df['AdvantageP1'] + 1e-15)
        df['ComplexDecisionRatio'] = df['StepDecisionToEnemy'] + df['SlideDecisionToEnemy'] + df['HopDecisionMoreThanOne']
        df['AggressiveActionsRatio'] = df['StepDecisionToEnemy'] + df['HopDecisionEnemyToEnemy'] + df['HopDecisionFriendToEnemy'] + df['SlideDecisionToEnemy']
        
        print("deal with outliers")
        df['PlayoutsPerSecond']=df['PlayoutsPerSecond'].clip(0,25000)
        df['MovesPerSecond']=df['MovesPerSecond'].clip(0,1000000)
        
        print("agent1 agent2 feature")
        cols=['selection','exploration_const','playout','score_bounds']
        for i in range(len(cols)):
            for j in range(2):
                df[f'{cols[i]}{j+1}']=df[f'agent{j+1}'].apply(lambda x:x.split('-')[i+1])
        

        print(f"one_hot_encoder")
        #train set nunique is (2,10)
        onehot_cols=[['NumOffDiagonalDirections', [0.0, 4.82, 2.0, 5.18, 3.08, 0.06]], ['NumLayers', [1, 0, 4, 5]], ['NumPhasesBoard', [3, 2, 1, 5, 4]], ['NumContainers', [1, 4, 3, 2]], ['NumDice', [0, 2, 1, 4, 6, 3, 5, 7]], ['ProposeDecisionFrequency', [0.0, 0.05, 0.01]], ['PromotionDecisionFrequency', [0.0, 0.01, 0.03, 0.02, 0.11, 0.05, 0.04]], ['SlideDecisionToFriendFrequency', [0.0, 0.19, 0.06]], ['LeapDecisionToEnemyFrequency', [0.0, 0.04, 0.01, 0.02, 0.07, 0.03, 0.14, 0.08]], ['HopDecisionFriendToFriendFrequency', [0.0, 0.13, 0.09]], ['HopDecisionEnemyToEnemyFrequency', [0.0, 0.01, 0.2, 0.03]], ['HopDecisionFriendToEnemyFrequency', [0.0, 0.01, 0.09, 0.25, 0.02]], ['FromToDecisionFrequency', [0.0, 0.38, 1.0, 0.31, 0.94, 0.67]], ['ProposeEffectFrequency', [0.0, 0.01, 0.03]], ['PushEffectFrequency', [0.0, 0.5, 0.96, 0.25]], ['FlipFrequency', [0.0, 0.87, 1.0, 0.96]], ['SetCountFrequency', [0.0, 0.62, 0.54, 0.02]], ['DirectionCaptureFrequency', [0.0, 0.55, 0.54]], ['EncloseCaptureFrequency', [0.0, 0.08, 0.1, 0.07, 0.12, 0.02, 0.09]], ['InterveneCaptureFrequency', [0.0, 0.01, 0.14, 0.04]], ['SurroundCaptureFrequency', [0.0, 0.01, 0.03, 0.02]], ['NumPlayPhase', [1, 2, 3, 4, 5, 6, 7, 8]], ['LineLossFrequency', [0.0, 0.96, 0.87, 0.46, 0.26, 0.88, 0.94]], ['ConnectionEndFrequency', [0.0, 0.19, 1.0, 0.23, 0.94, 0.35, 0.97]], ['ConnectionLossFrequency', [0.0, 0.54, 0.78]], ['GroupEndFrequency', [0.0, 1.0, 0.11, 0.79]], ['GroupWinFrequency', [0.0, 0.11, 1.0]], ['LoopEndFrequency', [0.0, 0.14, 0.66]], ['LoopWinFrequency', [0.0, 0.14, 0.66]], ['PatternEndFrequency', [0.0, 0.63, 0.35]], ['PatternWinFrequency', [0.0, 0.63, 0.35]], ['NoTargetPieceWinFrequency', [0.0, 0.72, 0.77, 0.95, 0.32, 1.0]], ['EliminatePiecesLossFrequency', [0.0, 0.85, 0.96, 0.68]], ['EliminatePiecesDrawFrequency', [0.0, 0.03, 0.91, 1.0, 0.36, 0.86]], ['NoOwnPiecesLossFrequency', [0.0, 1.0, 0.68]], ['FillEndFrequency', [0.0, 1.0, 0.04, 0.01, 0.99, 0.72]], ['FillWinFrequency', [0.0, 1.0, 0.04, 0.01, 0.99]], ['ReachDrawFrequency', [0.0, 0.9, 0.98]], ['ScoringLossFrequency', [0.0, 0.6, 0.62]], ['NoMovesLossFrequency', [0.0, 1.0, 0.13, 0.06]], ['NoMovesDrawFrequency', [0.0, 0.01, 0.04, 0.03, 0.22]], ['BoardSitesOccupiedChangeNumTimes', [0.0, 0.06, 0.42, 0.12, 0.14, 0.94]], ['BranchingFactorChangeNumTimesn', [0.0, 0.3, 0.02, 0.07, 0.04, 0.13, 0.01, 0.21, 0.03]], ['PieceNumberChangeNumTimes', [0.0, 0.06, 0.42, 0.12, 0.14, 1.0]], ['selection1', ['ProgressiveHistory', 'UCB1', 'UCB1GRAVE', 'UCB1Tuned']], ['selection2', ['ProgressiveHistory', 'UCB1GRAVE', 'UCB1', 'UCB1Tuned']], ['exploration_const1', ['0.1', '0.6', '1.41421356237']], ['exploration_const2', ['0.6', '0.1', '1.41421356237']], ['playout1', ['MAST', 'NST', 'Random200']], ['playout2', ['Random200', 'NST', 'MAST']]]
        for col,unique in onehot_cols:
            for u in unique:
                df[f'{col}_{u}']=(df[col]==u).astype(np.int8)
                
                
        print("deal with LudRules") 
        print("1:drop game")
        def drop_gamename(rule):
            rule=rule[len('(game "'):]
            for i in range(len(rule)):
                if rule[i]=='"':
                    return rule[i+1:]
        df['LudRules']=df['LudRules'].apply(lambda x:drop_gamename(x))

        print("2:player")
        def get_player(rule):
            player=''
            stack=[]#stack match () {}.
            for i in range(len(rule)):
                player+=rule[i]
                if rule[i] in ['(','{']:
                    stack.append(rule[i])
                elif rule[i] in [')','}']:
                    stack=stack[:-1]
                    if len(stack)==0:
                        return player
        df['player']=df['LudRules'].apply(lambda rule:get_player(rule))
        df=self.clean(df,'player')
        df['player_len']=df['player'].apply(len)
        df['LudRules']=[rule[len(player):] for player,rule in zip(df['player'],df['LudRules'])]
        df.drop(['player'],axis=1,inplace=True)
          
        print("Rules readable")
        for rule in ['EnglishRules', 'LudRules']:
            df[rule+"_ARI"]=df[rule].apply(lambda x:self.ARI(x))
            df[rule+"CLRI"]=df[rule].apply(lambda x:self.CLRI(x))
            df[rule+"McAlpine_EFLAW"]=df[rule].apply(lambda x:self.McAlpine_EFLAW(x))
                
        df['PlayoutsPerSecond/MovesPerSecond']=df['PlayoutsPerSecond']/df['MovesPerSecond']
        
        #model selection useless featurte
        drop_cols=['Cooperation', 'Team', 'TriangleShape', 'DiamondShape', 'SpiralShape', 'StarShape', 'SquarePyramidalShape', 'SemiRegularTiling', 'CircleTiling', 'SpiralTiling', 'MancalaThreeRows', 'MancalaSixRows', 'MancalaCircular', 'AlquerqueBoardWithOneTriangle', 'AlquerqueBoardWithTwoTriangles', 'AlquerqueBoardWithFourTriangles', 'AlquerqueBoardWithEightTriangles', 'ThreeMensMorrisBoard', 'ThreeMensMorrisBoardWithTwoTriangles', 'NineMensMorrisBoard', 'StarBoard', 'PachisiBoard', 'Boardless', 'NumColumns', 'NumCorners', 'NumOffDiagonalDirections', 'NumLayers', 'NumCentreSites', 'NumConvexCorners', 'NumPhasesBoard', 'NumContainers', 'Piece', 'PieceValue', 'PieceRotation', 'PieceDirection', 'LargePiece', 'Tile', 'NumComponentsType', 'NumDice', 'OpeningContract', 'SwapOption', 'Repetition', 'TurnKo', 'PositionalSuperko', 'AutoMove', 'InitialRandomPlacement', 'InitialScore', 'InitialCost', 'Moves', 'VoteDecision', 'SwapPlayersDecision', 'SwapPlayersDecisionFrequency', 'ProposeDecision', 'ProposeDecisionFrequency', 'PromotionDecisionFrequency', 'RotationDecision', 'RotationDecisionFrequency', 'StepDecisionToFriend', 'StepDecisionToFriendFrequency', 'StepDecisionToEnemy', 'SlideDecisionToEnemy', 'SlideDecisionToEnemyFrequency', 'SlideDecisionToFriend', 'SlideDecisionToFriendFrequency', 'LeapDecision', 'LeapDecisionFrequency', 'LeapDecisionToEmpty', 'LeapDecisionToEmptyFrequency', 'LeapDecisionToEnemy', 'LeapDecisionToEnemyFrequency', 'HopDecisionFriendToEmpty', 'HopDecisionFriendToEmptyFrequency', 'HopDecisionFriendToFriendFrequency', 'HopDecisionEnemyToEnemy', 'HopDecisionEnemyToEnemyFrequency', 'HopDecisionFriendToEnemy', 'HopDecisionFriendToEnemyFrequency', 'FromToDecisionFrequency', 'FromToDecisionEnemy', 'FromToDecisionEnemyFrequency', 'FromToDecisionFriend', 'SwapPiecesDecision', 'SwapPiecesDecisionFrequency', 'ShootDecision', 'ShootDecisionFrequency', 'VoteEffect', 'SwapPlayersEffect', 'PassEffect', 'ProposeEffect', 'ProposeEffectFrequency', 'AddEffectFrequency', 'SowFrequency', 'SowCapture', 'SowCaptureFrequency', 'SowRemove', 'SowBacktracking', 'SowBacktrackingFrequency', 'SowProperties', 'SowOriginFirst', 'SowCCW', 'PromotionEffectFrequency', 'PushEffect', 'PushEffectFrequency', 'Flip', 'FlipFrequency', 'SetNextPlayer', 'SetValue', 'SetValueFrequency', 'SetCount', 'SetCountFrequency', 'SetRotation', 'SetRotationFrequency', 'StepEffect', 'SlideEffect', 'LeapEffect', 'ByDieMove', 'MaxDistance', 'ReplacementCaptureFrequency', 'HopCaptureMoreThanOne', 'DirectionCapture', 'DirectionCaptureFrequency', 'EncloseCaptureFrequency', 'CustodialCapture', 'CustodialCaptureFrequency', 'InterveneCapture', 'InterveneCaptureFrequency', 'SurroundCapture', 'SurroundCaptureFrequency', 'CaptureSequence', 'CaptureSequenceFrequency', 'Group', 'Loop', 'Pattern', 'PathExtent', 'Territory', 'Fill', 'CanNotMove', 'Threat', 'CountPiecesMoverComparison', 'ProgressCheck', 'RotationalDirection', 'SameLayerDirection', 'ForwardDirection', 'BackwardDirection', 'BackwardsDirection', 'LeftwardDirection', 'RightwardsDirection', 'LeftwardsDirection', 'ForwardLeftDirection', 'ForwardRightDirection', 'BackwardLeftDirection', 'BackwardRightDirection', 'SameDirection', 'OppositeDirection', 'NumPlayPhase', 'LineLoss', 'LineLossFrequency', 'LineDraw', 'ConnectionEnd', 'ConnectionEndFrequency', 'ConnectionWinFrequency', 'ConnectionLoss', 'ConnectionLossFrequency', 'GroupEnd', 'GroupEndFrequency', 'GroupWin', 'GroupWinFrequency', 'GroupLoss', 'GroupDraw', 'LoopEnd', 'LoopEndFrequency', 'LoopWin', 'LoopWinFrequency', 'LoopLoss', 'PatternEnd', 'PatternEndFrequency', 'PatternWin', 'PatternWinFrequency', 'PathExtentEnd', 'PathExtentWin', 'PathExtentLoss', 'TerritoryEnd', 'TerritoryWin', 'TerritoryWinFrequency', 'Checkmate', 'CheckmateWin', 'NoTargetPieceEndFrequency', 'NoTargetPieceWin', 'NoTargetPieceWinFrequency', 'EliminatePiecesLoss', 'EliminatePiecesLossFrequency', 'EliminatePiecesDraw', 'EliminatePiecesDrawFrequency', 'NoOwnPiecesEnd', 'NoOwnPiecesWin', 'NoOwnPiecesLoss', 'NoOwnPiecesLossFrequency', 'FillEnd', 'FillEndFrequency', 'FillWin', 'FillWinFrequency', 'ReachWin', 'ReachLoss', 'ReachLossFrequency', 'ReachDraw', 'ReachDrawFrequency', 'ScoringLoss', 'ScoringLossFrequency', 'ScoringDraw', 'NoMovesLoss', 'NoMovesDrawFrequency', 'NoProgressEnd', 'NoProgressEndFrequency', 'NoProgressDraw', 'NoProgressDrawFrequency', 'BoardCoverageFull', 'BoardSitesOccupiedChangeNumTimes', 'BranchingFactorChangeLineBestFit', 'BranchingFactorChangeNumTimesn', 'DecisionFactorChangeNumTimes', 'MoveDistanceChangeSign', 'MoveDistanceChangeLineBestFit', 'PieceNumberChangeNumTimes', 'PieceNumberMaxIncrease', 'ScoreDifferenceMedian', 'ScoreDifferenceVariance', 'ScoreDifferenceChangeAverage', 'ScoreDifferenceChangeSign', 'ScoreDifferenceChangeLineBestFit', 'Math', 'Division', 'Modulo', 'Absolute', 'Exponentiation', 'Minimum', 'Maximum', 'Even', 'Odd', 'Visual', 'GraphStyle', 'MancalaStyle', 'PenAndPaperStyle', 'ShibumiStyle', 'BackgammonStyle', 'JanggiStyle', 'XiangqiStyle', 'ShogiStyle', 'TableStyle', 'SurakartaStyle', 'NoBoard', 'ChessComponent', 'KingComponent', 'QueenComponent', 'KnightComponent', 'RookComponent', 'BishopComponent', 'PawnComponent', 'FairyChessComponent', 'PloyComponent', 'ShogiComponent', 'XiangqiComponent', 'StrategoComponent', 'JanggiComponent', 'TaflComponent', 'StackType', 'Stack', 'ShowPieceValue', 'ShowPieceState', 'Implementation', 'StateType', 'StackState', 'VisitedSites', 'InternalCounter', 'SetInternalCounter', 'Efficiency', 'NumOffDiagonalDirections_0.0', 'NumOffDiagonalDirections_4.82', 'NumOffDiagonalDirections_2.0', 'NumOffDiagonalDirections_5.18', 'NumOffDiagonalDirections_3.08', 'NumOffDiagonalDirections_0.06', 'NumLayers_1', 'NumLayers_0', 'NumLayers_4', 'NumLayers_5', 'NumPhasesBoard_1', 'NumPhasesBoard_5', 'NumDice_0', 'NumDice_2', 'NumDice_6', 'NumDice_3', 'NumDice_5', 'NumDice_7', 'ProposeDecisionFrequency_0.0', 'ProposeDecisionFrequency_0.05', 'ProposeDecisionFrequency_0.01', 'PromotionDecisionFrequency_0.0', 'PromotionDecisionFrequency_0.01', 'PromotionDecisionFrequency_0.03', 'PromotionDecisionFrequency_0.02', 'PromotionDecisionFrequency_0.11', 'PromotionDecisionFrequency_0.05', 'PromotionDecisionFrequency_0.04', 'SlideDecisionToFriendFrequency_0.0', 'SlideDecisionToFriendFrequency_0.19', 'SlideDecisionToFriendFrequency_0.06', 'LeapDecisionToEnemyFrequency_0.0', 'LeapDecisionToEnemyFrequency_0.04', 'LeapDecisionToEnemyFrequency_0.01', 'LeapDecisionToEnemyFrequency_0.02', 'LeapDecisionToEnemyFrequency_0.07', 'LeapDecisionToEnemyFrequency_0.03', 'LeapDecisionToEnemyFrequency_0.14', 'LeapDecisionToEnemyFrequency_0.08', 'HopDecisionFriendToFriendFrequency_0.0', 'HopDecisionFriendToFriendFrequency_0.13', 'HopDecisionFriendToFriendFrequency_0.09', 'HopDecisionEnemyToEnemyFrequency_0.0', 'HopDecisionEnemyToEnemyFrequency_0.01', 'HopDecisionEnemyToEnemyFrequency_0.2', 'HopDecisionEnemyToEnemyFrequency_0.03', 'HopDecisionFriendToEnemyFrequency_0.0', 'HopDecisionFriendToEnemyFrequency_0.01', 'HopDecisionFriendToEnemyFrequency_0.09', 'HopDecisionFriendToEnemyFrequency_0.25', 'HopDecisionFriendToEnemyFrequency_0.02', 'FromToDecisionFrequency_0.0', 'FromToDecisionFrequency_0.38', 'FromToDecisionFrequency_1.0', 'FromToDecisionFrequency_0.31', 'FromToDecisionFrequency_0.94', 'FromToDecisionFrequency_0.67', 'ProposeEffectFrequency_0.0', 'ProposeEffectFrequency_0.01', 'ProposeEffectFrequency_0.03', 'PushEffectFrequency_0.0', 'PushEffectFrequency_0.5', 'PushEffectFrequency_0.96', 'PushEffectFrequency_0.25', 'FlipFrequency_0.0', 'FlipFrequency_0.87', 'FlipFrequency_1.0', 'FlipFrequency_0.96', 'SetCountFrequency_0.0', 'SetCountFrequency_0.62', 'SetCountFrequency_0.54', 'SetCountFrequency_0.02', 'DirectionCaptureFrequency_0.0', 'DirectionCaptureFrequency_0.55', 'DirectionCaptureFrequency_0.54', 'EncloseCaptureFrequency_0.0', 'EncloseCaptureFrequency_0.08', 'EncloseCaptureFrequency_0.1', 'EncloseCaptureFrequency_0.07', 'EncloseCaptureFrequency_0.12', 'EncloseCaptureFrequency_0.02', 'EncloseCaptureFrequency_0.09', 'InterveneCaptureFrequency_0.0', 'InterveneCaptureFrequency_0.01', 'InterveneCaptureFrequency_0.14', 'InterveneCaptureFrequency_0.04', 'SurroundCaptureFrequency_0.0', 'SurroundCaptureFrequency_0.01', 'SurroundCaptureFrequency_0.03', 'SurroundCaptureFrequency_0.02', 'NumPlayPhase_3', 'NumPlayPhase_4', 'NumPlayPhase_5', 'NumPlayPhase_6', 'NumPlayPhase_7', 'NumPlayPhase_8', 'LineLossFrequency_0.0', 'LineLossFrequency_0.96', 'LineLossFrequency_0.87', 'LineLossFrequency_0.46', 'LineLossFrequency_0.26', 'LineLossFrequency_0.88', 'LineLossFrequency_0.94', 'ConnectionEndFrequency_0.0', 'ConnectionEndFrequency_0.19', 'ConnectionEndFrequency_1.0', 'ConnectionEndFrequency_0.23', 'ConnectionEndFrequency_0.94', 'ConnectionEndFrequency_0.35', 'ConnectionEndFrequency_0.97', 'ConnectionLossFrequency_0.0', 'ConnectionLossFrequency_0.54', 'ConnectionLossFrequency_0.78', 'GroupEndFrequency_0.0', 'GroupEndFrequency_1.0', 'GroupEndFrequency_0.11', 'GroupEndFrequency_0.79', 'GroupWinFrequency_0.0', 'GroupWinFrequency_0.11', 'GroupWinFrequency_1.0', 'LoopEndFrequency_0.0', 'LoopEndFrequency_0.14', 'LoopEndFrequency_0.66', 'LoopWinFrequency_0.0', 'LoopWinFrequency_0.14', 'LoopWinFrequency_0.66', 'PatternEndFrequency_0.0', 'PatternEndFrequency_0.63', 'PatternEndFrequency_0.35', 'PatternWinFrequency_0.0', 'PatternWinFrequency_0.63', 'PatternWinFrequency_0.35', 'NoTargetPieceWinFrequency_0.0', 'NoTargetPieceWinFrequency_0.72', 'NoTargetPieceWinFrequency_0.77', 'NoTargetPieceWinFrequency_0.95', 'NoTargetPieceWinFrequency_0.32', 'NoTargetPieceWinFrequency_1.0', 'EliminatePiecesLossFrequency_0.0', 'EliminatePiecesLossFrequency_0.85', 'EliminatePiecesLossFrequency_0.96', 'EliminatePiecesLossFrequency_0.68', 'EliminatePiecesDrawFrequency_0.0', 'EliminatePiecesDrawFrequency_0.03', 'EliminatePiecesDrawFrequency_0.91', 'EliminatePiecesDrawFrequency_1.0', 'EliminatePiecesDrawFrequency_0.36', 'EliminatePiecesDrawFrequency_0.86', 'NoOwnPiecesLossFrequency_0.0', 'NoOwnPiecesLossFrequency_1.0', 'NoOwnPiecesLossFrequency_0.68', 'FillEndFrequency_0.0', 'FillEndFrequency_1.0', 'FillEndFrequency_0.04', 'FillEndFrequency_0.01', 'FillEndFrequency_0.99', 'FillEndFrequency_0.72', 'FillWinFrequency_0.0', 'FillWinFrequency_1.0', 'FillWinFrequency_0.04', 'FillWinFrequency_0.01', 'FillWinFrequency_0.99', 'ReachDrawFrequency_0.0', 'ReachDrawFrequency_0.9', 'ReachDrawFrequency_0.98', 'ScoringLossFrequency_0.0', 'ScoringLossFrequency_0.6', 'ScoringLossFrequency_0.62', 'NoMovesLossFrequency_0.0', 'NoMovesLossFrequency_1.0', 'NoMovesLossFrequency_0.13', 'NoMovesLossFrequency_0.06', 'NoMovesDrawFrequency_0.0', 'NoMovesDrawFrequency_0.01', 'NoMovesDrawFrequency_0.04', 'NoMovesDrawFrequency_0.03', 'NoMovesDrawFrequency_0.22', 'BoardSitesOccupiedChangeNumTimes_0.0', 'BoardSitesOccupiedChangeNumTimes_0.06', 'BoardSitesOccupiedChangeNumTimes_0.42', 'BoardSitesOccupiedChangeNumTimes_0.12', 'BoardSitesOccupiedChangeNumTimes_0.14', 'BoardSitesOccupiedChangeNumTimes_0.94', 'BranchingFactorChangeNumTimesn_0.0', 'BranchingFactorChangeNumTimesn_0.3', 'BranchingFactorChangeNumTimesn_0.02', 'BranchingFactorChangeNumTimesn_0.07', 'BranchingFactorChangeNumTimesn_0.04', 'BranchingFactorChangeNumTimesn_0.13', 'BranchingFactorChangeNumTimesn_0.01', 'BranchingFactorChangeNumTimesn_0.21', 'BranchingFactorChangeNumTimesn_0.03', 'PieceNumberChangeNumTimes_0.0', 'PieceNumberChangeNumTimes_0.06', 'PieceNumberChangeNumTimes_0.42', 'PieceNumberChangeNumTimes_0.12', 'PieceNumberChangeNumTimes_0.14', 'PieceNumberChangeNumTimes_1.0', 'KintsBoard', 'FortyStonesWithFourGapsBoard', 'Roll', 'SumDice', 'CheckmateFrequency', 'NumDice_4']
        df.drop(['Id',#meaningless
         #train nunique=1
         'Properties', 'Format', 'Time', 'Discrete', 'Realtime', 'Turns', 'Alternating', 'Simultaneous', 'HiddenInformation', 'Match', 'AsymmetricRules', 'AsymmetricPlayRules', 'AsymmetricEndRules', 'AsymmetricSetup', 'Players', 'NumPlayers', 'Simulation', 'Solitaire', 'TwoPlayer', 'Multiplayer', 'Coalition', 'Puzzle', 'DeductionPuzzle', 'PlanningPuzzle', 'Equipment', 'Container', 'Board', 'PrismShape', 'ParallelogramShape', 'RectanglePyramidalShape', 'TargetShape', 'BrickTiling', 'CelticTiling', 'QuadHexTiling', 'Hints', 'PlayableSites', 'Component', 'DiceD3', 'BiasedDice', 'Card', 'Domino', 'Rules', 'SituationalTurnKo', 'SituationalSuperko', 'InitialAmount', 'InitialPot', 'Play', 'BetDecision', 'BetDecisionFrequency', 'VoteDecisionFrequency', 'ChooseTrumpSuitDecision', 'ChooseTrumpSuitDecisionFrequency', 'LeapDecisionToFriend', 'LeapDecisionToFriendFrequency', 'HopDecisionEnemyToFriend', 'HopDecisionEnemyToFriendFrequency', 'HopDecisionFriendToFriend', 'FromToDecisionWithinBoard', 'FromToDecisionBetweenContainers', 'BetEffect', 'BetEffectFrequency', 'VoteEffectFrequency', 'SwapPlayersEffectFrequency', 'TakeControl', 'TakeControlFrequency', 'PassEffectFrequency', 'SetCost', 'SetCostFrequency', 'SetPhase', 'SetPhaseFrequency', 'SetTrumpSuit', 'SetTrumpSuitFrequency', 'StepEffectFrequency', 'SlideEffectFrequency', 'LeapEffectFrequency', 'HopEffectFrequency', 'FromToEffectFrequency', 'SwapPiecesEffect', 'SwapPiecesEffectFrequency', 'ShootEffect', 'ShootEffectFrequency', 'MaxCapture', 'OffDiagonalDirection', 'Information', 'HidePieceType', 'HidePieceOwner', 'HidePieceCount', 'HidePieceRotation', 'HidePieceValue', 'HidePieceState', 'InvisiblePiece', 'End', 'LineDrawFrequency', 'ConnectionDraw', 'ConnectionDrawFrequency', 'GroupLossFrequency', 'GroupDrawFrequency', 'LoopLossFrequency', 'LoopDraw', 'LoopDrawFrequency', 'PatternLoss', 'PatternLossFrequency', 'PatternDraw', 'PatternDrawFrequency', 'PathExtentEndFrequency', 'PathExtentWinFrequency', 'PathExtentLossFrequency', 'PathExtentDraw', 'PathExtentDrawFrequency', 'TerritoryLoss', 'TerritoryLossFrequency', 'TerritoryDraw', 'TerritoryDrawFrequency', 'CheckmateLoss', 'CheckmateLossFrequency', 'CheckmateDraw', 'CheckmateDrawFrequency', 'NoTargetPieceLoss', 'NoTargetPieceLossFrequency', 'NoTargetPieceDraw', 'NoTargetPieceDrawFrequency', 'NoOwnPiecesDraw', 'NoOwnPiecesDrawFrequency', 'FillLoss', 'FillLossFrequency', 'FillDraw', 'FillDrawFrequency', 'ScoringDrawFrequency', 'NoProgressWin', 'NoProgressWinFrequency', 'NoProgressLoss', 'NoProgressLossFrequency', 'SolvedEnd', 'Behaviour', 'StateRepetition', 'PositionalRepetition', 'SituationalRepetition', 'Duration', 'Complexity', 'BoardCoverage', 'GameOutcome', 'StateEvaluation', 'Clarity', 'Narrowness', 'Variance', 'Decisiveness', 'DecisivenessMoves', 'DecisivenessThreshold', 'LeadChange', 'Stability', 'Drama', 'DramaAverage', 'DramaMedian', 'DramaMaximum', 'DramaMinimum', 'DramaVariance', 'DramaChangeAverage', 'DramaChangeSign', 'DramaChangeLineBestFit', 'DramaChangeNumTimes', 'DramaMaxIncrease', 'DramaMaxDecrease', 'MoveEvaluation', 'MoveEvaluationAverage', 'MoveEvaluationMedian', 'MoveEvaluationMaximum', 'MoveEvaluationMinimum', 'MoveEvaluationVariance', 'MoveEvaluationChangeAverage', 'MoveEvaluationChangeSign', 'MoveEvaluationChangeLineBestFit', 'MoveEvaluationChangeNumTimes', 'MoveEvaluationMaxIncrease', 'MoveEvaluationMaxDecrease', 'StateEvaluationDifference', 'StateEvaluationDifferenceAverage', 'StateEvaluationDifferenceMedian', 'StateEvaluationDifferenceMaximum', 'StateEvaluationDifferenceMinimum', 'StateEvaluationDifferenceVariance', 'StateEvaluationDifferenceChangeAverage', 'StateEvaluationDifferenceChangeSign', 'StateEvaluationDifferenceChangeLineBestFit', 'StateEvaluationDifferenceChangeNumTimes', 'StateEvaluationDifferenceMaxIncrease', 'StateEvaluationDifferenceMaxDecrease', 'BoardSitesOccupied', 'BoardSitesOccupiedMinimum', 'BranchingFactor', 'BranchingFactorMinimum', 'DecisionFactor', 'DecisionFactorMinimum', 'MoveDistance', 'MoveDistanceMinimum', 'PieceNumber', 'PieceNumberMinimum', 'ScoreDifference', 'ScoreDifferenceMinimum', 'ScoreDifferenceChangeNumTimes', 'Roots', 'Cosine', 'Sine', 'Tangent', 'Exponential', 'Logarithm', 'ExclusiveDisjunction', 'Float', 'HandComponent', 'SetHidden', 'SetInvisible', 'SetHiddenCount', 'SetHiddenRotation', 'SetHiddenState', 'SetHiddenValue', 'SetHiddenWhat', 'SetHiddenWho',
          #in train.columns not in test.columns
         'num_wins_agent1', 'num_draws_agent1', 'num_losses_agent1',
         #object
         'Behaviour', 'StateRepetition', 'Duration', 'Complexity', 'BoardCoverage', 'GameOutcome', 'StateEvaluation', 'Clarity', 'Decisiveness', 'Drama', 'MoveEvaluation', 'StateEvaluationDifference', 'BoardSitesOccupied', 'BranchingFactor', 'DecisionFactor', 'MoveDistance', 'PieceNumber', 'ScoreDifference','selection1', 'selection2', 'exploration_const1', 'exploration_const2', 'playout1', 'playout2', 'score_bounds1', 'score_bounds2',
        ]+drop_cols,axis=1,inplace=True,errors='ignore') 
        
        df=self.reduce_mem_usage(df)
        print(f"feature_count:{len(df.columns)}")
        print("-"*30)
        return df

    def CV_feats(self,df,mode='',model_name='',fold=0):
        str_cols=['EnglishRules', 'LudRules']#'agent1','agent2',
        for col in str_cols:
            df=self.clean(df,col)
            df[f'{col}_len']=df[col].apply(len)
            if mode=='train':
                tfidf = TfidfVectorizer(max_features=500,ngram_range=(2,3))
                tfidf_feats=tfidf.fit_transform(df[col]).toarray()
                for i in range(tfidf_feats.shape[1]):
                    df[f"{col}_tfidf_{i}"]=tfidf_feats[:,i]
                self.pickle_dump(tfidf,f'{model_name}_{fold}_{col}tfidf.model')
                self.tfidf_paths.append((model_name,fold,col))
            else:#mode=='test'
                for i in range(len(self.tfidf_paths)):
                    if (model_name,fold,col)==self.tfidf_paths[i]:
                        tfidf=self.pickle_load(f'/kaggle/input/models-v0/models_v0/{model_name}_{fold}_{col}tfidf.model')
                        tfidf_feats=tfidf.transform(df[col]).toarray()
                        for j in range(tfidf_feats.shape[1]):
                            df[f"{col}_tfidf_{j}"]=tfidf_feats[:,j]
        df.drop(str_cols+['agent1','agent2'],axis=1,inplace=True)
        return df 
    
    def RMSE(self,y_true,y_pred):
        return np.sqrt(np.mean((y_true-y_pred)**2))
    def train_model(self,):
        self.train=self.FE(self.train,mode='train')
        #https://www.kaggle.com/code/ravi20076/mcts2024-mlmodels-v1/notebook
        cat_params1={'task_type'           : "GPU",
               'eval_metric'         : "RMSE",
               'bagging_temperature' : 0.50,
               'iterations'          : 3096,
               'learning_rate'       : 0.08,
               'max_depth'           : 12,
               'l2_leaf_reg'         : 1.25,
               'min_data_in_leaf'    : 24,
               'random_strength'     : 0.25, 
               'verbose'             : 0,
              }
        
        cat_params2={'task_type'           : "GPU",
               'eval_metric'         : "RMSE",
               'bagging_temperature' : 0.60,
               'iterations'          : 3096,
               'learning_rate'       : 0.08,
               'max_depth'           : 12,
               'l2_leaf_reg'         : 1.25,
               'min_data_in_leaf'    : 24,
               'random_strength'     : 0.20, 
               'max_bin'             :2048,
               'verbose'             : 0,
              }
        models=[
                (CatBoostRegressor(**cat_params1),'cat1'),
                (CatBoostRegressor(**cat_params2),'cat2'),
               ]
        
        for (model,model_name) in models:
            print("start training")
            X=self.train.drop([self.target,'GameRulesetName'],axis=1)
            GameRulesetName=self.train['GameRulesetName']
            y=self.train[self.target]
            oof_preds=np.zeros(len(X))
            
            y_int=round(y*15)
            
            sgkf = StratifiedGroupKFold(n_splits=self.num_folds,random_state=2024,shuffle=True)

            for fold, (train_index, valid_index) in (enumerate(sgkf.split(X,y_int,GameRulesetName))):
                print(f"fold:{fold}")

                X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
                y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

                X_train=self.CV_feats(X_train,mode='train',model_name=model_name,fold=fold)
                X_valid=self.CV_feats(X_valid,mode='test',model_name=model_name,fold=fold)

                model.fit(X_train, y_train,
                      eval_set=(X_valid, y_valid),
                      early_stopping_rounds=100, verbose=100)
                
                oof_preds[valid_index]=model.predict(X_valid)

                self.pickle_dump(model,f'{model_name}_{fold}.model')
                self.model_paths.append((model_name,fold))

                del X_train,X_valid,y_train,y_valid
                gc.collect()
            
            np.save(f"{model_name}_oof.npy",np.clip(oof_preds*1.1,-0.985,0.985))
            
            print(f"RMSE:{self.RMSE(y.values,np.clip(oof_preds*1.1,-0.985,0.985) )}")
            
    def infer_model(self,test):
        test=self.FE(test,mode='test')
        test.drop(['GameRulesetName'],axis=1,inplace=True)
        test_preds=[]
        for i in range(len(self.model_paths)):
            model_name,fold=self.model_paths[i]
            test_copy=self.CV_feats(test.copy(),mode='test',model_name=model_name,fold=fold)
            model=self.pickle_load(f'/kaggle/input/models-v0/models_v0/{model_name}_{fold}.model')
            test_preds+=[np.clip(model.predict(test_copy)*1.1,-0.985,0.985)]
        return np.mean(test_preds,axis=0)
```

# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">EDA about CV and LB</h1></span>

#### Here is a brief analysis of the CV and LB of each version of my notebook.

#### On the one hand, it is really necessary to analyze CV and LB. On the other hand, my EDA files are not publicly available, so if someone directly copies my code and submits it, there should be an error. I hope my open-source notebook is for everyone to learn from, not just to get a score.

```python
import matplotlib.pyplot as plt#plot
preprocessor=Preprocessor(num_folds=5,train=train)
# CV=preprocessor.check['CV'].values
# LB=preprocessor.check['LB'].values
# plt.xlim(0.4,0.5)
# plt.ylim(0.4,0.5)
# plt.scatter(CV,LB)
# plt.show()
```

```python
# preprocessor.check[['CV','LB']].corr()
```

# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">train and predict</h1></span>

#### groupkfold:The ultimate goal of this competition is to predict the effects of agents on unknown games, so we also need to divide them according to game types during cross validation. Therefore, we have chosen groupkfold here.

### Explanation on post-processing:
 
#### 1.minimize $RMSE(W*oof+b,target)$:In general, everyone defaults to w=1 and b=0, but in fact, many times we can find better w and b, so we need post-processing.

#### 2.clip(pred,-0.985,0.985):For example, if our model predicts both data [-1,1] as 1, assuming that the model predicts both data as the same value, then obviously when the prediction is 0, the RMSE is the smallest. When the prediction result moves from 1 to 0, the RMSE of the second data is increasing, but the rate of increase is not as fast as the rate of decrease of the first data, ultimately leading to a decrease in their overall RMSE. Although the clip operation increases the rmse for data that is truly -1 or 1, it decreases for rmse that is truly other values but predicted to be 1.

```python
import os
#environment provided by competition hoster
import kaggle_evaluation.mcts_inference_server
counter = 0

def predict(test, submission):
    global counter
#     if counter == 0:
#         preprocessor.train_model()  
    counter += 1
    return submission.with_columns(pl.Series('utility_agent1', preprocessor.infer_model(test.to_pandas())))

inference_server = kaggle_evaluation.mcts_inference_server.MCTSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv',
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv'
        )
    )
```
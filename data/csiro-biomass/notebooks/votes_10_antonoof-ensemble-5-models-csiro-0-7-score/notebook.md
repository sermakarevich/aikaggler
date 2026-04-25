# ensemble 5 models CSIRO | 0.7 score

- **Author:** Antonoof
- **Votes:** 232
- **Ref:** antonoof/ensemble-5-models-csiro-0-7-score
- **URL:** https://www.kaggle.com/code/antonoof/ensemble-5-models-csiro-0-7-score
- **Last run:** 2026-01-12 22:38:32.057000

---

### This is my last public solution in this competition. Firsly, I would like to wish everyone a happy new year in 2026, and I would like to thank these people very much: Zhuang Jia(jiazhuang), Chika Komari(gothamjocker), CigarCat(takahitomizunobyts), samu2505(samu2505), Mattia Angeli(mattiaangeli), Kh0a(llkh0a). Thank you for your dedication in this competition, you have done a really great job! I strongly advise you to look at their work because they have different approaches to solving this problem.

### I would also like to express my deep gratitude to the users, thank you for supporting me and improving yourself. I think it's clear how I +- built the pipeline and made improvements. further improvements will be much more difficult to achieve. 😄 Good luck with your work!

### Time inference pipeline 7.45 +- hours

```python
!python /kaggle/input/5-ensemble-csiro/inference1.py
!python /kaggle/input/5-ensemble-csiro/inference2.py
!python /kaggle/input/5-ensemble-csiro/inference3.py
!python /kaggle/input/5-ensemble-csiro/inference4.py
!python /kaggle/input/5-ensemble-csiro/inference5.py
```

```python
import pandas as pd

submission1 = pd.read_csv('submission1.csv')
submission2 = pd.read_csv('submission2.csv')
submission3 = pd.read_csv('submission3.csv')
submission4 = pd.read_csv('submission4.csv')
submission5 = pd.read_csv('submission5.csv')

submission1 = submission1.rename(columns={'target': 'target_1'})
submission2 = submission2.rename(columns={'target': 'target_2'})
submission3 = submission3.rename(columns={'target': 'target_3'})
submission4 = submission4.rename(columns={'target': 'target_4'})
submission5 = submission5.rename(columns={'target': 'target_5'})

merged = pd.merge(submission1, submission2, on='sample_id')
merged = pd.merge(merged, submission3, on='sample_id')
merged = pd.merge(merged, submission4, on='sample_id')
merged = pd.merge(merged, submission5, on='sample_id')

weight1, weight2, weight3, weight4, weight5 = 0.19, 0.27, 0.22, 0.22, 0.10
merged['target'] = (merged['target_1'] * weight1 + 
                    merged['target_2'] * weight2 + 
                    merged['target_3'] * weight3 +
                    merged['target_4'] * weight4 +
                    merged['target_5'] * weight5)

submission = merged[['sample_id', 'target']]
submission.to_csv('submission.csv', index=False)
print(submission.head())
```
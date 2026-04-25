# Public 4th Place / Private 15th Place Solution - Konwinski Prize

- **Author:** ISAKA Tsuyoshi
- **Date:** 2025-03-18T03:08:42.623Z
- **Topic ID:** 568799
- **URL:** https://www.kaggle.com/competitions/konwinski-prize/discussion/568799
---

First and foremost, I would like to express my gratitude to my teammates ( @kazukishida , @nozomiyamamoto ), the organizer Andy, the Kaggle operations team, and everyone who participated in this competition.<br>
The three of us on the team have been long-time friends and will continue to participate in many competitions in the future.


**Key Differences Between the Konwinski Prize and the Original SWE-bench**
- Completely new data without any leakage
- High penalty for incorrect answers, necessitating an effective skipping mechanism
- Cannot use APIs, requiring the use of open-source LLMs

Our best Public LB score was LB = +0.056243 (4 correct, 0 incorrect), ranking 4th place!
To achieve this score—and to maximize our winning chances on the Private LB—we implemented the following strategies.


## 1. Validation Strategy
In summary, the dataset consists of:<br>
**Training Data**: Very small (only 6 instances)<br>
[**External Data**](https://www.kaggle.com/competitions/konwinski-prize/discussion/552605): About 6,400 instances, but with a slightly different distribution from the LB<br>
**Public LB**: 71 clean instances<br>
**Private LB**: [Even cleaner](https://www.kaggle.com/competitions/konwinski-prize/discussion/564884#3142741), with 150-200 instances<br>

[So our strategy is IGNORE CV, CARE ABOUT PUBLIC LB, AND TRUST METHODOLOGY.](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/discussion/169230)

While the competition's metric is undeniably unstable, we hoped it was not purely a lottery.<br>
Furthermore, [in the "LLM Prompt Recovery" competition, which had 0 training data, the distribution difference between Public and Private LB was minimal](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494438).<br>

## 2. Improving Accuracy
First and foremost, we aimed to increase the number of correct answers. We started by building on the excellent [public notebook Version 94](https://www.kaggle.com/code/huikang/starter-notebook-select-patch-verify?scriptVersionId=223324126) shared by @huikang , which achieved 7 correct answers. Thank you so much—you are undoubtedly the MVP of this competition!

### 2.1. Model
We used [DeepSeek-R1 · deepseek-r1-distill-qwen-32b-awq](https://www.kaggle.com/models/shelterw/deepseek-r1/Transformers/deepseek-r1-distill-qwen-32b-awq). Interestingly, the non-AWQ quantized model did not perform well. We tried various other models, but none were satisfactory. We had high expectations for QwQ-32B, which was recently released, but neither the AWQ-quantized nor non-quantized versions performed well. If any team successfully utilized this model, we'd love to hear about it.

### 2.2. Model Parameters
For the reading, patching, and verifying steps, we set the temperature parameters to 0.6/0.6/0.0. For the final step, based on my [experience in LLM 20 Questions](https://www.kaggle.com/competitions/llm-20-questions/discussion/529525), we set a lower temperature.<br>

### 2.3. Prompt Engineering
Based on the prompt from the public notebook, we devised an improved version incorporating the following enhancements. Our goal was to reduce the number of incorrect answers by allowing the model to output "None" when it lacks confidence, in addition to "Yes" and "No." However, since the difference was not significant, we decided not to include it in the final submission. We’re sharing this for reference.

```python
verifying_prompt: str = """
This is the problem statement.

{problem_statement}

These are the files that are thought to be relevant, which may not be complete.

{file_content_string}

This is the proposed patch to fix the problem.

{patch_string}

Evaluate whether the patch works
- The patch fully fixes the problem described in the problem statement.
- The patch does not cause side effects and make any other tests fail.

End your response with exactly one of the following:
- <label>Yes</label>, this fixes the problem.
- <label>No</label>, this does not fix the problem.
- <label>None</label>, I am unsure whether the patch fixes the problem.

Reminder
- Only evaluate, do not provide suggestions on how to fix.
- If you cannot confidently determine whether the patch fully fixes the problem, return exactly `<label>None</label>`.
- When you don't know how to do it, please honestly output `<label>None</label>`, which will be more pleasing than just randomly outputting it.
- Remember to write exactly either of <label>Yes</label> or <label>No</label> or <label>None</label> in the last line
""".strip()
```

### 2.4. Candidate Text Selection: Edit Distance Selection
We applied some refinements when selecting the final candidate texts.  

1. First, we selected only those cases where at least one final candidate text was available.  
2. Next, we filtered out generated patches where `len` exceeded 1000.  
3. Finally, we devised an algorithm called **Edit Distance Selection**. Specifically, for each candidate text, we calculated the total sum of edit distances to all other candidate texts.  

By selecting the text with the **smallest total edit distance**, we ensured that the most representative and averaged text was chosen, effectively filtering out outliers.  

Earlier in the competition, a submission that initially resulted in LB=-1 was improved to a valid LB score using Edit Distance Selection, suggesting that this approach may be effective.

```bash
=== Edit Distance Selection Check ===
Candidate Texts:
1: I like cats.
2: I love cats.
3: I love animals.
4: Animals are great.
5: Dogs are wonderful so much.

Selected (Most Average) Text: I love cats.
```

## 3. Skip Mechanism

### 3.1. Consideration of the Scoring Function
As a fundamental premise, achieving at least one correct answer is crucial, followed by the importance of the value of `correct - incorrect`. After securing correct answers, actively skipping is necessary to reduce incorrect answers. Therefore, we worked diligently on refining the skip mechanism.<br>

The possibility of LB=-1 is concerning, but since the Private dataset has approximately 2.5 times more instances than the Public dataset, we believe the likelihood of LB=-1 is low.<br>

Our goal is to win a gold medal.<br>
One month before the competition ended, we noticed that many teams in the gold medal range had `correct=1, incorrect=0`. [This score is expected to be a popular sweet spot in the Private LB as well](https://www.kaggle.com/competitions/konwinski-prize/discussion/564639). Three days before the competition ended, only 63 teams had achieved a Public LB score where correct answers exceeded incorrect answers (i.e., `correct=1, incorrect=0` or higher).<br>

Thus, to maximize the probability of surpassing this score, we simulated the winning probability.<br>
If the numbers of correct, incorrect, and extracted instances are determined, the probability of surpassing `correct=1, incorrect=0` can be calculated using the hypergeometric distribution.<br>
For example, let’s simulate the winning probability based on different numbers of challenges (draw attempts, non-skipped attempts) when the total of correct and incorrect answers is 10.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5927283%2F64babc47b8987bb15fede1afa76fac5c%2Ffig1.png?generation=1742266762135271&alt=media)


Odd numbers tend to show better winning probabilities compared to their adjacent even-numbered extractions. This is because even-numbered extractions include even results (e.g., `correct=2, incorrect=2`), which are considered a losing score in this scenario. To address this, we implemented a strategy to ensure that the challenge count more frequently ends as an odd number. See Section 3.5 for more details.

Initially, we hypothesized a scenario where `correct < incorrect`, planning to set `challenges=3`. A secondary benefit of this approach was that it provided a significant time buffer.<br>

P.S. If the goal were to achieve `correct=1, incorrect=0`, it could be reduced to the ["Secretary Problem"](https://en.wikipedia.org/wiki/Secretary_problem). However, we chose to overcome this challenge.<br>

However, an event occurred around March 2. [Version 113 of the public notebook achieved a score equivalent to 5th place at the time (`correct=3, incorrect=0`)](https://www.kaggle.com/code/huikang/starter-notebook-select-patch-verify/comments#3132127).<br>

Most of the time, this notebook resulted in LB=-1, but in rare cases, it achieved such a high score. Nevertheless, some participants copied and submitted this high-score notebook, and we anticipated that a few of them would achieve `correct=3, incorrect=0`.<br>

Therefore, we shifted our meta-target to this score. As a result, we changed `challenges=5`.<br>
Achieving `correct=5, incorrect=0` or `correct=4, incorrect=1` would very likely secure a gold medal, and even `correct=3, incorrect=2` might be sufficient.<br>

The assumed scores above allowed us to compete favorably against teams within particularly common score ranges three days before the competition ended:<br>
- (5,0) Our assumed score
- (4,1) Our assumed score
- (3,0) 12 teams ranked 6th-15th
- (3,2) Our assumed score
- (2,1) 18 teams ranked 35th-52nd
- (1,0) 11 teams ranked 53rd-63rd

We decided against using `challenges ≥ 7`, considering cases where correct answers were fewer than incorrect ones.<br>

To support this decision, we used the following simulation, considering the increase in instances from Public LB to Private LB, setting the total number of correct and incorrect answers to 15.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5927283%2F32b51cfad402c694bf7be47fd13df524%2Ffig2.png?generation=1742266901479774&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5927283%2F6a9d73253b66f60509041c268a5f2d7d%2Ffig3.webp?generation=1742266921974907&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5927283%2F2034490aa1292a0a4a76d533d01c3a70%2Ffig4.webp?generation=1742266943009304&alt=media) 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5927283%2F58641295a80bcd3076c0c43ebc1dfde2%2Ffig5.webp?generation=1742266949226861&alt=media)



### 3.2. Time Management
Interval Time Limits: Inference with LLM is particularly time-consuming. We set interval limits of 7 minutes, 15 minutes, and 19 minutes for reading, patching, and verifying, respectively. If these limits are exceeded, skipping the issue is a strong option.<br>
Global Time Limits: We set global time limits of 8 hours for Public LB and 23 hours for Private LB. The reason for allowing a 1-hour margin is that the penalty for skipping is minimal in this competition.<br>
By the way, I likely lost one of my final submissions in [AIMO1 due to a time overrun](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/516936)😂.


### 3.3. Problem Length
According to the [SWE-bench paper](https://arxiv.org/abs/2310.06770), problem length correlates with difficulty. Also, the penalty for skipping in this competition is extremely small.<br>
Therefore, we initially skipped issues with a length outside 1200-3800 characters. This approach provided additional time flexibility.<br>
By focusing only on lengths within this range, we reached (5,8), which corresponded to LB=-0.042335 (ranked 18th at the time).<br>
Further analysis showed that issues exceeding 3400 characters never resulted in correct answers.<br>
Shorter issues (under 1200 characters) also yielded only a single correct answer.<br>
We believe shorter issues performed worse because they lacked sufficient information or contained noisy issues.<br>
However, since [Private LB should be curated](https://www.kaggle.com/competitions/konwinski-prize/discussion/564884#3140753), our judgment might have been incorrect.<br>

For the final submission, further tuning (including parameter adjustments) resulted in skipping issues outside 1802-3400 characters. This achieved LB=+0.056243 (`correct=4, incorrect=0`), ranking us 4th.<br>




### 3.4. Number of File Contents  
By skipping instances where the number of file contents exceeded **500**, our score changed from (5, 8) to (5, 4), improving to LB=+0.013997 (rank 7 at the time).  

We further explored the threshold for file content count:  
- By skipping instances with more than **388** file contents, our score changed to (4, 2), improving to LB=+0.028077 (rank 6 at the time).  
- However, when loosening the restriction to **403**, the score became (5, 4), meaning one additional correct answer but two more incorrect ones.  

Therefore, we set the file contents threshold to **388** for the final submission.

By plotting problem statement length (x-axis) against file content count (y-axis), we estimated the dataset distribution based on our submission results. Given that there were only six training instances, tuning on the 71 public LB instances seemed reasonable, though there was a risk of overfitting.  

To better illustrate our approach, we simulated the distribution of correct, incorrect, and skipped instances based on multiple submission results.  
Since we only had range-based data, we sampled from a uniform distribution. The scatter plot is a conceptual visualization and may contain errors.  

The figures below show simulation results:  
- **Top:** Full range (x-axis = problem statement length, y-axis = file content count).  
- **Bottom:** Restricted y-axis (file content count = 1–500).  



![file_num vs len](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5927283%2F354be36ae17e4e18aa5ae9ba4f03f096%2FScatter%20Plot%20Of%20Problem%20Length%20Vs%20File%20Num%20With%20Labels.png?generation=1753319329605873&alt=media)



![file_num vs len (limited)](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5927283%2F3e621d0685b1c6c9e7605f1f08dbd90d%2FScatter%20Plot%20Of%20Problem%20Length%20Vs%20File%20Num%20With%20Labels(fn_range1-500).png?generation=1753319380379679&alt=media)

### 3.5. Contingency Plan
If a certain number of turns (`limit_turn`) passes without reaching the target challenge count, a method updates the parameters.<br>

```python
def update_to_plan_B(self):
  if self.challenge_count % 2 == 1:
    self.cfg.max_challenge_num = self.challenge_count  # End with this challenge count
  else:
    self.cfg.max_challenge_num = self.challenge_count + 1  # End with the next challenge count
    if self.challenge_count == 0:
      self.cfg.max_file_contents_num = self.calc_bottom_20(self.file_contents_num_list)
```

`limit_turn=100` allows an extra issue attempt if needed. In most cases, 5 challenges were completed within 100 turns, preventing the contingency plan from triggering.




## 4. Development Strategy  
As many of you may have felt, the implementation difficulty of this competition is extremely high. Therefore, we have carefully devised our development strategy.  
Thanks to the following strategic improvements, we encountered **only one critical bug**: an instance where we exceeded the interval limit per prediction. (Even this was quickly addressed by implementing the time management strategy described in **Section 3.2**.)  

### 4.1. Team Role Allocation  
We broadly divided our roles as follows:  
- **@isakatsuyoshi** focused on **Kaggle**,  
- **@kazukishida** handled **Development**,  
- **@nozomiyamamoto** took charge of **Research**.  

Of course, there were some overlaps in responsibilities, but this structure allowed us to work efficiently.  
We are proud that we were able to demonstrate outstanding teamwork, as we are long-time friends who have shared many experiences together.  

### 4.2. Development Using GitHub  
We actively conducted **code reviews**. @isakatsuyoshi and @kazukishida reviewed each other’s pull requests to ensure code quality.  
- **@kazukishida** transformed an excellent publicly available notebook into modular, loosely coupled scripts, making development more manageable.  
- **@isakatsuyoshi** conducted numerous trial experiments, paving the way for victory in this competition.  
- While **@isakatsuyoshi** focused on aggressive innovation, **@kazukishida** ensured system stability, allowing us to develop effectively.  
- Additionally, **@nozomiyamamoto** provided comprehensive support through research-based ideation, prompt engineering, and code validation.  

### 4.3. Test Code  
The **submission limits** in this competition are extremely strict, making any bugs potentially catastrophic. To mitigate this risk, we wrote thorough **test code**.  

- We developed **12 modular scripts** with **98 test cases**, achieving **80% code coverage**.  
- Our testing approach included **parallelized testing, fixtures, and parameterization**, among other optimizations.  

In summary, we wrote extensive test code, implemented exception handling and time management strategies, and developed a deep understanding of the limitations of LLMs. Our ultimate goal was to **maximize our chances of winning a gold medal**.  

We eagerly await the results in three months.  
Glory to all Kagglers! 🚀✨  


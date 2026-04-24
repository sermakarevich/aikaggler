# Sharing of our methods (longest leaderboard award submission)

- **Author:** Just a test
- **Date:** 2026-02-04T10:37:40.673Z
- **Topic ID:** 671835
- **URL:** https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/671835
---

First off, a huge thanks to the hosts and a big thank you to the many competitors for their selfless sharing. We are happy to share our approach.

While our solution shares the same high-level logic as many public notebooks (using a 120B model with code execution), our implementation details differ quite significantly. Instead of managing the loop in the inference notebook, we leverage the vLLM **response endpoint** which supports integrated tool calling via the Harmony protocol (GPT-OSS).

However, the out-of-the-box demo servers for these tools weren't quite ready for a high-concurrency competition environment. Here is how we patched the infrastructure to make it work.

## 1. The Architecture: Native Tool Calling
We use **vLLM** served with the `tool-server="demo"` parameter. This allows the model to reason, call a Python tool, observe the output, and continue reasoning in a loop behind the scenes.

To make this viable for the competition, we had to patch both **GPT-OSS** and **vLLM**.

## 2. Concurrent Stateful Jupyter Sessions
The default implementation of the tool server is generally stateless or single-session. To solve 50 questions efficiently within the time limit, we needed to run multiple questions in parallel, where each question maintains its own persistent Jupyter kernel state.

### The Fix:
*   We monkey-patched `vllm/tools.py` to pass the `request_id` into the `author.name` field of the message.
*   On the tool server side (GPT-OSS), we modified the `PythonTool` to read this `author.name`.
*   This allows the tool to identify which Jupyter session a request belongs to. If a session doesn't exist for that ID, it creates one; otherwise, it resumes the existing session.

## 3. Robustness & "Thinking Process" Mimicry
We implemented several features to make the code execution more robust, inspired by how the ChatGPT app handles analysis:

*   **Output Truncation:** To prevent blowing up the context window with massive dataframes or infinite loops, we truncate the output (e.g., keeping the first and last 300 characters). You often see this in ChatGPT as `...[output truncated]...`.
*   **Internal Timeouts (`timeoutify`):** We wrap the code execution in a custom decorator that uses `signal.setitimer`. This enforces a hard timeout on the execution inside the container/kernel before the external ZMQ socket gives up.

Additionally, some fail-safe method is also used:

*   **Auto-Restart:** If a kernel dies (SegFault or unresponsive), the tool detects the `KernelDeadError`, tears down the specific session, and transparently starts a fresh one for the next attempt.
*   **Formatting Fixes:** We noticed the model sometimes outputs `\displaystyle` instead of `\boxed`. We added a regex post-processor to capture these edge cases.

## 4. Time Management
We opted for simplicity over complex banking.
*   **Allocation:** A flat 10 minutes per question.
*   **Attempts:** 4 attempts per question.
*   **Selection:** We extract the final answer based on the most frequent non-trivial result (ignoring 0/1 unless they are the only options).

---

### Code Resources
*   **My code used to reach 39 (Dec 7 version):** [link1](https://www.kaggle.com/code/yiyangzheng/fork-of-v1207-rc-39-longest-leaderboard)
*   **My code used to reach 40 (Dec 9 version):** [link2](https://www.kaggle.com/code/yiyangzheng/fork-of-v1209-rc-40-longest-leaderboard)
*   **Siancy code to reach their highest score:** [link3](https://www.kaggle.com/code/shreyansh01m/41-50-same-as-parthenos-s-no-modifications)

---

## 5. Evaluation & Performance
To validate the stability and performance of our patched tool-use infrastructure, we ran an evaluation on a subset of **IMO-Bench (AnswerBench)**.

*   **Dataset:** 226 questions from AnswerBench (filtered for positive integer answers to match the competition format).
*   **Hardware:** Single B200 (192GB).
*   **Configuration:** 8 attempts per question.
*   **Total Runtime:** Under 5 hours.
*   **Performance:** Using simple majority voting without any additional test-time compute tricks, our baseline achieved **188 / 226 (83.2%)**. When using vote by number of tool calls you can usually get **84%+**.

#### Distribution of correct attempts per question:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14522089%2Fbe5e7ff26ebb8bdd3f42302355c48cc8%2Fperformance_distribution.svg?generation=1770200865202571&alt=media)

**We are also sharing the full raw logs for all attempts to help the community analyze where the model succeeds or fails:**
[link](https://www.kaggle.com/datasets/yiyangzheng/answerbench-gpt-oss/data)
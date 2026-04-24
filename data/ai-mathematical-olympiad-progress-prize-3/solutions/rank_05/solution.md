# Ideas behind my public submission scoring [14/50] and [21/50]

- **Author:** Tong Hui Kang
- **Date:** 2025-11-23T00:48:28.237Z
- **Topic ID:** 637882
- **URL:** https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/637882
---

I have shared a notebook that has scored [21/50](https://www.kaggle.com/code/huikang/streaming-inference?scriptVersionId=281680710), tying for the highest scoring notebook at publication. (Someone resubmitted and scored [24/50](https://www.kaggle.com/code/huxingyu/streaming-inference1))


Here are some ideas I hope people will adopt


- Streaming inference (hence the title) and terminating on the first submitted answer. You should already be familiar with the OpenAI completions API, which by default only replies when the completion is completed. However, you can stream the text, and by doing this you can interrupt the completion. The vLLM server will receive the interrupt and stop generating, saving the GPU capacity for the next question.
- Asking to verify for quick answers. This has helped to avoid the trivial answer of 62140 (2^20!) and prompt the model to think harder and return the correct answer of 21818 instead.
- Asking to guess after a certain token limit. This has managed to produce the correct answer of 57447 in the local runs instead of not returning any answer.
- API-ification. vLLM server is started and serves an OpenAI compatible API. You can also introduce more interesting logic as you can treat the vLLM service like an API to call. You can benchmark other models with the same code too. You can use [remote](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/643580) GPUs.
- Saving the data, so you can see the LLM failure modes. In the data I annotate the number of generated tokens, and the answer if any.
- Saving the vLLM logs. You can see the throughput and the GPU KV cache usage in the logs, which could inform what to tune next.




---


The following are ideas that are no longer relevant but have helped me to score [5th place](https://www.kaggle.com/code/huikang/streaming-inference?scriptVersionId=281020357), scoring 14/50 when the previous best notebook was scoring [8](https://www.kaggle.com/code/theunforgiven7/aimo-3-baseline-gpt-oss-20b?scriptVersionId=280576466).


- Termination based on GPU KV cache usage. When GPU KV cache usage reaches 100%, bad things happen. I stop adding more requests to generate, and also terminate generation. (This is no longer relevant because it seems that the strategy is to generate very few threads, we want to go deep instead of going wide. It is okay if generation is being preempted)
- Vote manipulation. There is no fixed number of generations that I run. I want to terminate when I am confident an answer is correct. The logic as of writing works like this - larger answers are given greater weight compared to smaller numbers. To win the vote, you need a minimum number of points, some proportion of the total number of points, and to be a certain number of points better than the next best answer. These points are tuned by staring at the data. (This is no longer relevant because the GPT-OSS on high reasoning will not knowingly guess the answer)




In the notebook I have logs on how much each answer is scoring. This is an example of the logs. 
```
score_list |    206.2 over 74 attempts
Current GPU usage 9.2
       512       74.9       12
         1       22.7       28
     57447       21.9        2
     98537       11.5        1
     59367       11.0        1
     58947       11.0        1
```
Even though only 28 completions end with the answer 1, the answer 512 is chosen even though there is only 12 completions ending with it (both answers are wrong, however). More weight is given to the larger answer. Since the answers could be between 0 to 99999, I expect the answers to be somewhat uniformly distributed rather than logarithmically distributed. Small answers are likely to be guesses when the AI has given up. (But this idea is likely losing its relevance very soon.)


---


Some ideas you can try
- Code execution. Currently I do not execute code. Maybe some problems can be solved much more easily and accurately with code.
- Branch selection and mixing. Instead of generations going in parallel, maybe you can share key results between generations.




I will write more on these ideas in separate posts (reading vLLM logs, optimizing token throughput)
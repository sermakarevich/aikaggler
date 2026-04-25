# Few lessons learned

- **Author:** Victor Shlepov
- **Date:** 2024-11-19T15:26:27.853Z
- **Topic ID:** 547060
- **URL:** https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/547060
---

Hey folks, I hope you’re enjoying the competition as much as I am.

Here's a few thoughts, observations, and hypotheses based on my experiences over the past month or so of working with the data. Please keep in mind that these are just my personal insights - after all, being somewhere around 30th to 40th place in the public rankings doesn’t really give me the authority to make any definitive statements! :)

1) Non-Stationary Data: In financial markets, there is no absolute ground truth to learn from past trends. While some patterns do exist, they tend to emerge and disappear in unpredictable ways. This may sound like a quote from the book, but in fact, it has major implications for model architecture, training schedules, and almost everything else.

2) Online Learning: Online learning is crucial in this context. In my experiments online training yields around 0.0030 points - that's quite a major gain, given that best public result for now is around  0.0090. I believe that any model in the top-100 is likely retrained during test submissions, which is where "lags" become useful. However, in my experience, lags are not particularly effective as model inputs because past prices and returns often provide poor predictions for future outcomes.

3) Cross-Validation: Cross-validation, and validation in general, can be less useful in this scenario. For instance, setting aside the last 120 days for validation would negatively impact the model's performance during testing. Additionally, it’s important to maintain the temporal order of the data. I find it beneficial to create metrics that incorporate momentum to observe how model performance evolves over time.

4) Architecture: A combination of various stateless and stateful gates (to capture temporal patterns) and attention mechanisms (to capture interdependencies across different symbols and features) are likely key components of successful architectures.

5) Loss Function: Zero-mean R² is a unique metric that behaves very differently from MSE around zero values. I'd use it as a loss function rather than relying on built-in options. [UPDATE: There's a valid point by @shlomoron that "[this makes no sense](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/547060#3050169)" - it does not fit my experimental results, but I'd carefully listen to what the guy is saying anyway]. I have also experimented extensively with different clipping and post-processing strategies, but so far, the most effective approach has been to allow the neural network to learn the solution on its own. 

Sorry, no code for now - it's meant to be a competition, right? :)

[UPDATES - 1]

6) You all know the data is "ragged", if not to say messy [no offence to Host, that's how markets work]. Different number of steps and "traded" symbols per day was mentioned quite some times. New symbols and features emerge along the way:
```python
                               symbol_id
date_id
0          [1, 7, 9, 10, 14, 16, 19, 33]
1                     [0, 2, 13, 15, 38]
2                                    [3]
3                                   [12]
4                                [8, 17]
8                                   [34]
13                                  [11]
20                                  [30]
484      [5, 20, 21, 22, 25, 26, 29, 36]
487                                 [23]
713                     [27, 28, 35, 37]
952                          [4, 24, 31]
1063                         [6, 18, 32]
```

Theres's more to that - every single axis of a [None, steps, symbols, features] is unstable - number of non-NaN features for the SAME "symbol_id" fluctuates over dates, some features start with non-zero time_id, number of non-NaN features for the SAME "date_id" differs for symbols, etc. It's all meant to say that it might make sense to think about (a) masking and (b) careful approach to normalization, which would be the next discussion topics.

7) Captain Obvious here. You'd want to mimic the submission API during the training. I mean, the model inputs and frequency of gradient updates should be exactly the same as during the hidden test. I did all those mistakes on the start [time series is not my piece of cake, really] - used a shifted true responders (with a causal masking, but still...) as a input to a decoder in an MLM-like encoder-decoder architecture, or just simply updated the weights on each time step during training. Both options turned to be a dead end - with a sky-level training metrics and non-existent performance on the hidden test.

[UPDATES - 2]

8) I've mentioned the validation already. After making a dozen of "blind" (based on the train results) submissions over the weekend I felt myself doing some monkey business with no control over it. So I end up putting aside the last 120 days for the validation. The downside here is that once you make a decision re the optimal amount of training based on the validation metrics - you'd likely have to retrain the model from scratch on the full dataset to avoid unnecessary  biases. Which means twice longer feedback loop, but - a greater control over it.
# ConvNeXt Approach - [CV 31.9 LB 36.4]

- **Author:** Bartley
- **Date:** 2025-05-20T18:43:51.937Z
- **Topic ID:** 579841
- **URL:** https://www.kaggle.com/competitions/waveform-inversion/discussion/579841
---

Here is another baseline trained on the full-resolution data. Like my previous notebook, I provide a few trained checkpoints and leave lots of room for customization.

Notebook [here](https://www.kaggle.com/code/brendanartley/convnext-baseline)
Weights [here](https://www.kaggle.com/datasets/brendanartley/openfwi-preprocessed-72x72)

Here are the CV scores for this approach. Cheers!

```python
+--------------+--------+
| Dataset      | Score  |
+--------------+--------+
| CurveFault_A |   6.07 |
| CurveFault_B |  92.54 |
| CurveVel_A   |  15.07 |
| CurveVel_B   |  53.67 |
| FlatFault_A  |   4.32 |
| FlatFault_B  |  37.60 |
| FlatVel_A    |   2.62 |
| FlatVel_B    |  12.89 |
| Style_A      |  37.38 |
| Style_B      |  57.70 |
+--------------+--------+
```
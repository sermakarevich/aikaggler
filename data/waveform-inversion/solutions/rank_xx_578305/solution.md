# HGNet-V2 Encoder - [CV 55.6 LB 60.9]

- **Author:** Bartley
- **Date:** 2025-05-09T23:31:42.583Z
- **Topic ID:** 578305
- **URL:** https://www.kaggle.com/competitions/waveform-inversion/discussion/578305
---

Excited to share a notebook that builds on the great starter notebook from @egortrushin. The updated notebook showcases training on 2x T4 GPUs to maximize our GPU quota!

Notebook [here](https://www.kaggle.com/code/brendanartley/hgnet-v2-starter-cv-65-2-lb-69-7)
Dataset [here](https://www.kaggle.com/datasets/brendanartley/openfwi-preprocessed-72x72)

I also provide a few trained checkpoints for a HGNet-V2-Unet model, flip augmentations, EMA, and more. Here are the CV scores by dataset type. Experiment away! 😀

```python
+--------------+--------+
| Dataset      | Score  |
+--------------+--------+
| CurveFault_A |  15.47 |
| CurveFault_B | 132.66 |
| CurveVel_A   |  37.10 |
| CurveVel_B   |  96.18 |
| FlatFault_A  |  14.07 |
| FlatFault_B  |  71.20 |
| FlatVel_A    |  19.20 |
| FlatVel_B    |  48.95 |
| Style_A      |  51.15 |
| Style_B      |  70.72 |
+--------------+--------+

```


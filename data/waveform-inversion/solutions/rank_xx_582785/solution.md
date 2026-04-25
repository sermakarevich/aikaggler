# CAFormer Improved - [CV 24.2 LB 28.8]

- **Author:** Bartley
- **Date:** 2025-06-02T18:38:52.270Z
- **Topic ID:** 582785
- **URL:** https://www.kaggle.com/competitions/waveform-inversion/discussion/582785
---

Happy to share an improved full resolution model. This model uses a CAFormer encoder with an improved decoder (pixel shuffle, SCSE, intermediate convolutions, etc.).

Notebook [here](https://www.kaggle.com/code/brendanartley/caformer-full-resolution-improved)
Dataset [here](https://www.kaggle.com/datasets/brendanartley/openfwi-preprocessed-72x72)

```
+--------------+----------+
| Dataset      | Caformer |
+--------------+----------+
| CurveFault_A |     4.00 |
| CurveFault_B |    71.06 |
| CurveVel_A   |     9.19 |
| CurveVel_B   |    38.60 |
| FlatFault_A  |     2.58 |
| FlatFault_B  |    24.29 |
| FlatVel_A    |     1.31 |
| FlatVel_B    |     7.20 |
| Style_A      |    34.34 |
| Style_B      |    48.90 |
+--------------+----------+
| Overall      |    24.15 |
+--------------+----------+
```

This will be my last notebook in this competition, but I am hopeful that we can push this architecture further as a community!

Happy Kaggling 😄
# recodai-luc-scientific-image-forgery-detection: cross-solution summary

This competition focused on copy-move forgery detection (CMFD), requiring robust methods to identify duplicated regions within images despite various post-processing distortions. Winning approaches heavily relied on traditional signal processing and geometric matching techniques rather than deep learning, specifically leveraging the Discrete Cosine Transform (DCT) and efficient nearest-neighbor search. The successful pipeline emphasized block-based feature extraction, sign-based coefficient analysis for noise resilience, and precise spatial matching to localize forged regions.

## Competition flows
- Raw images are split into overlapping blocks, processed through DCT and Cellular Automata to generate sign-based feature vectors, and matched via kd-tree nearest-neighbor search to identify duplicated regions.

## Data processing
- Images are divided into overlapping blocks; features are extracted via Discrete Cosine Transform (DCT), and feature vectors are constructed using Cellular Automata based on the sign information of DCT coefficients.

## Features engineering
- Feature vectors derived from the sign information of DCT coefficients using Cellular Automata.

## Notable individual insights
- Unranked (Copy-move forgery (CMFD) detection technique with DCT): The signs of DCT coefficients are more robust to post-processing attacks than their magnitudes.

## Solutions indexed
- ? [[solutions/rank_xx_613066/solution|Copy-move forgery (CMFD) detection technique with DCT (Discrete Cosine Transform)]]

## Papers cited
- [Copy-move forgery detection technique](https://www.sciencedirect.com/science/article/abs/pii/S2214212619307343)

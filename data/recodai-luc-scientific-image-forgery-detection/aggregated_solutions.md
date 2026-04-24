# recodai-luc-scientific-image-forgery-detection: cross-solution summary

This competition focused on copy-move image forgery detection, requiring algorithms to identify duplicated regions within manipulated images that may have undergone various post-processing attacks. The top solution demonstrates that carefully engineered classical signal processing techniques—specifically leveraging the signs of DCT coefficients, cellular automata, and kd-tree matching—can achieve superior robustness and localization accuracy compared to magnitude-based or purely deep learning approaches.

## Key challenges
- Resisting post-processing attacks (e.g., JPEG compression, additive white Gaussian noise)
- Accurately matching duplicated regions across overlapping image blocks
- Maintaining feature invariance while preserving spatial precision for forgery localization

## Models
- Cellular Automata
- kd-tree

## Preprocessing
- Dividing input image into overlapping blocks

## Feature engineering
- Sign information of DCT coefficients
- Feature vectors constructed via Cellular Automata

## What worked
- Using sign information of DCT coefficients instead of magnitudes significantly improves robustness to post-processing attacks.

## Critical findings
- Signs of DCT coefficients are inherently more robust to post-processing attacks (e.g., JPEG, AWGN) than their magnitudes.

## Notable individual insights
- null (Copy-move forgery (CMFD) detection technique with DCT (Discrete Cosine Transform)): Using the signs of DCT coefficients rather than their magnitudes provides significantly better robustness against post-processing attacks.
- null (Copy-move forgery (CMFD) detection technique with DCT (Discrete Cosine Transform)): Processing DCT coefficient signs through cellular automata effectively builds robust feature vectors that resist manipulation.

## Solutions indexed
- ? [[solutions/rank_xx_613066/solution|Copy-move forgery (CMFD) detection technique with DCT (Discrete Cosine Transform)]]

## Papers cited
- [Copy-move forgery (CMFD) detection technique with DCT (Discrete Cosine Transform)](https://www.sciencedirect.com/science/article/abs/pii/S2214212619307343)

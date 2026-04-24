# recodai-luc-scientific-image-forgery-detection: cross-solution summary

This submission focuses on copy-move forgery detection (CMFD) rather than a standard Kaggle competition pipeline, presenting a research-driven methodology for identifying duplicated image regions. The approach leverages the Discrete Cosine Transform (DCT) on overlapping image blocks, utilizing cellular automata to extract feature vectors from the signs of DCT coefficients. This strategy proves highly effective against common post-processing attacks, establishing a robust baseline for forgery detection in academic and practical contexts.

## Competition flows
- Not applicable; the text describes a research paper methodology rather than a Kaggle competition pipeline.

## Data processing
- Input images are divided into overlapping blocks.
- DCT is applied to each block to extract frequency-domain features.
- Cellular Automata are used to process the sign information of the DCT coefficients to build feature vectors.

## Notable individual insights
- Unranked (Copy-move forgery (CMFD) detection technique with DCT): The signs of DCT coefficients are significantly more robust to post-processing attacks than their magnitudes.
- Unranked (Copy-move forgery (CMFD) detection technique with DCT): Using overlapping blocks improves feature extraction stability for forgery detection.
- Unranked (Copy-move forgery (CMFD) detection technique with DCT): The method demonstrates exceptional robustness against combined post-processing attacks like JPEG compression and additive white Gaussian noise.
- Unranked (Copy-move forgery (CMFD) detection technique with DCT): Existing CMFD methods typically show limited robustness to post-processing, which this approach specifically overcomes.

## Solutions indexed
- ? [[solutions/rank_xx_613066/solution|Copy-move forgery (CMFD) detection technique with DCT (Discrete Cosine Transform)]]

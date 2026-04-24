# Copy-move forgery (CMFD) detection technique with DCT (Discrete Cosine Transform)

- **Author:** Marília Prata
- **Date:** 2025-10-24T01:43:31.270Z
- **Topic ID:** 613066
- **URL:** https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection/discussion/613066
---

## Copy-move forgery detection technique

A robust copy-move forgery detection technique based on discrete cosine transform and cellular automata

**Authors**: Gulnawaz Gani, Fasel Qadir

"Copy Move Forgery (CMF) is a type of digital image forgery in which an image region is copied and pasted to another location within the same image with malicious intent to misrepresent its meaning. To prevent misinterpretation of an image content, several Copy Move Forgery Detection (CMFD) methods have been proposed in the past. However, the existing methods show limited robustness on images altered with post-processing attacks such as noise addition, compression, blurring etc."

"In this paper, the authors proposed a robust method for detecting copy-move forgeries under different post-processing attacks. They used Discrete Cosine Transform (DCT) to extract features from each block. Next, Cellular Automata is employed to construct feature vectors based on the sign information of the DCT coefficients."

"Finally, feature vectors are matched using the kd-tree based nearest-neighbor searching method to find the duplicated areas in the image. Experimental results show that the proposed method performs exceptionally well relative to the other state-of-the-art methods from the literature even when an image is heavily affected by the post-processing attacks, in particular, JPEG compression and additive white Gaussian noise."

"Furthermore, experiments confirm the robustness of the proposed method against the range of combined attacks."

"The method works by dividing the input image into overlapping blocks and extracting features from these blocks using **Discrete Cosine Transform** and Cellular Automata. The observation that signs of the DCT coefficients tend to be more robust to different post-processing attacks than their magnitudes itself." 

https://www.sciencedirect.com/science/article/abs/pii/S2214212619307343
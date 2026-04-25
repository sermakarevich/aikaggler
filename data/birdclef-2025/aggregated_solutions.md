# birdclef-2025: cross-solution summary

BirdCLEF 2025 focused on audio classification with significant challenges from unlabeled secondary bird calls, dataset shift, and poor cross-validation correlation. Winning approaches universally leveraged mel spectrogram optimization, iterative pseudo-labeling or self-distillation pipelines to enrich training data, and seed-diverse model ensembling combined with targeted post-processing to stabilize leaderboard performance.

## Competition flows
- Raw audio chunking, mel spectrogram conversion, CNN/SED training via supervised learning and multi-iterative noisy student self-training, followed by overlapping framewise averaging and delta shift TTA.
- Mel spectrogram conversion treated as normalized images, image-based augmentations, EfficientNet training with FocalBCE loss, Public LB tracking, and model combination with post-processing.
- Human voice removal, audio truncation, mel spectrogram conversion, multi-stage self-distillation with EfficientNet backbones, and overlapping/smoothed weighted ensemble.
- 5s segment extraction, CNN backbone fine-tuning with class balancing, iterative pseudo-label augmentation from soundscapes, and 3-model ensemble with probability-scaling post-processing.
- Human voice filtering, mel spectrogram conversion, two-stage SED/CNN training with pseudo-labeling, ONNX conversion, TTA, and weighted ensemble with threshold smoothing.
- Multi-resolution mel spectrogram extraction, human/non-target filtering, stage1 SED pseudo-label generation, stage2 CNN/SED fine-tuning with hybrid loss, and 10-model ensemble with kernel smoothing.

## Data reading
- 20-second audio chunks normalized by absmax, converted to mel spectrograms (32kHz, 224 mel bins, 0-16kHz, n_fft=4096, hop=1252, top_db=80) and repeated 3 times as input channels
- Mel spectrograms computed from raw audio (sr=32000, 192 mel bins, fmin=20, fmax=15000, window_size=2048, hop_size=768), converted to log scale via log(melspec+1e-6)

## Data processing
- 20-second chunking with absmax normalization
- Mel spectrogram conversion with specified parameters
- 0-1 normalization and channel repetition
- External Xeno-Canto data filtering (duration < 60s)
- MixUp augmentation (p=0.5) on raw audio with left-side zero padding
- Power transform on pseudo-labels to reduce noise
- WeightedRandomSampler based on sum of max label probabilities
- Smoothing and delta shift TTA during inference
- Convert melspectrograms to images and apply normalization
- Apply RandAug, RandomErasing, Time and Frequency Masking, and Mixup (probability 1.0)
- Filter/Process CSA recordings by isolating human sound
- Train on random 5-second segments instead of the first 5 seconds
- Silero VAD used to detect and manually remove human voice segments
- Manual curation for underrepresented classes (n < ~30)
- Truncation to first 60s for cleaned files and 30s for others
- Duplication for classes with <20 samples
- Augmentations: Resampling, Gain, FilterAugment, FrequencyMasking, TimeMasking, Sumix on mel domain
- Logarithmic normalization: log(melspec+1e-6)
- 5s random segment extraction
- manual/automatic removal of non-vocalization "alien speech"
- filtering undersampled species during pretraining
- soft pseudo-label generation (trimming probabilities < 0.1 to zero)
- 40% probability replacement sampling for pseudo-data
- post-processing scaling of chunk predictions by file-level top probability
- Removed 50% human voice from audio
- Converted raw signals to Mel-spectrograms using torchaudio transforms (SR=32000, n_fft=2048, n_mels=256/128, f_min=20, f_max=16000/14000, dynamic hop_length)
- Applied RMS sampling over random sampling
- Applied EMA during training
- Converted models to ONNX format for inference
- Applied TTA with 2-second window on 10s chunks
- Smoothed predictions with [0.2, 0.6, 0.2] weights
- Set fmax=2000 during inference
- Removed samples with obvious human speech (train_audio_clean)
- Used trained models to remove segments lacking target species, sampling at 5-second intervals (train_audio_clean_v2)
- Extracted mel spectrograms with resolutions 384x160, 384x256, 320x192, 320x160; fmin=0, fmax=16000, n_fft=1536/2048
- Applied Mixup with additive, frequency, and time masking
- Applied noise augmentation, label smoothing, and flip augmentations
- Filtered pseudo-labels using high and low confidence thresholds on train_soundscapes

## Models
- SED head
- EfficientNet
- EfficientNetV2
- RegNetY
- ECA-NFNet
- ConvNeXt
- ResNeXt
- SEResNeXt
- Multi-layer Perceptron

## Frameworks used
- OpenVINO
- timm
- PyTorch
- torchaudio
- Streamlit
- Silero VAD
- ONNX

## Loss functions
- CrossEntropy
- FocalBCE
- FocalLoss (gamma=2)
- Focal loss
- BCE
- Label smoothing (0.005)
- CE+BCE
- Modified hybrid loss (CE + negative penalty on top 95% of negative logits exceeding threshold -5)
- Positive confidence penalty

## CV strategies
- 5 folds with at least 1 sample per label per fold; validated primarily on Public LB due to poor CV/LB correlation
- Public Leaderboard validation (explicitly noted that k-fold cross-validation is invalid for this competition's validation scheme)
- 5-fold cross-validation for Stage 1; no folds used in Stages 2 and 3
- Stratification by primary_label and grouping by author. Tested strategies for undersampled species including adding samples to folds, removing them from train, or adding them only to train folds. Used validation splits #1 and #3
- Validated directly on the leaderboard due to dataset shift and lack of a same-distribution validation set

## Ensembling
- Final submission used equal weights across 7 models from different training stages and backbones, combined with overlapping framewise prediction averaging, smoothing, and delta shift TTA
- Combined models that achieved 0.854, 0.856, 0.858, and 0.859 scores to reach a final public LB score of 0.872, along with post-processing steps referenced from a previous notebook
- Predictions from two model groups with different random seeds were combined using 2.5-second overlap, weighted with alpha=0.5, and smoothed with a [0.1, 0.8, 0.1] window before final submission
- Final submission combined 3 models (two tf_efficientnetv2_s and one eca_nfnet_l0) trained on different validation splits with varying pseudo-iterations and sampling strategies, combined without TTA and applied with probability-scaling post-processing
- Weighted average was used to ensemble models trained with different random seeds, with no extensive threshold tuning or calibration beyond prediction smoothing and frequency cutoff adjustments
- Ensembled 10 models (mix of ConvNeXt Tiny and EfficientNetV2-S trained on different data splits and pseudo-label proportions) with public/private scores of 0.915/0.921, applying kernel smoothing and class-weighted probability averaging during post-processing

## Insights
- Longer 20-second input chunks optimally balance model performance and inference speed compared to shorter durations
- MixUp with a constant 0.5 blending weight prevents suppression of meaningful signals when mixing clean and noisy pseudo-labeled data
- Applying a power transform greater than 1 to pseudo-label probabilities effectively mitigates noise accumulation in multi-iterative self-training
- WeightedRandomSampler based on the sum of maximum label probabilities stabilizes training by prioritizing higher-quality pseudo-labeled samples
- Treating mel spectrograms as images rather than raw audio waves significantly improves performance
- Melspectrogram parameter tuning is highly impactful and should be explored visually
- Combining middle features and passing them into the global pooling layer in EfficientNet models outperforms standard convolutional heads
- Visualizing model attentions (CAM) on training soundscapes helps diagnose what the model is learning
- Many bird calls in the training data were unlabeled because recorders focused on target species, making accurate secondary label assignment the core challenge
- Iterative self-distillation effectively enriches training labels with true secondary labels that were missing from the original annotations
- Mixing train_audio and train_soundscapes at a 1:1 ratio during later distillation stages further improved performance
- Pseudo-labeling is a critical driver of performance in semi-supervised audio classification
- Validation scores do not strongly correlate with Public LB scores when making small tweaks (~1% AUC)
- Pretraining on a massive, filtered Xeno-Canto snapshot is a highly effective initialization strategy
- Models benefit from some noise; strictly avoiding false positives can hurt generalization
- RMS sampling outperformed random sampling for audio data
- Applying EMA during training improved model robustness and reduced performance variance across epochs
- Random seed selection had a surprisingly large impact on final scores, necessitating ensemble diversity across seeds
- Converting models to ONNX with multi-threading caused inference timeouts, requiring single-threaded execution for successful submission
- Leveraging unlabeled train_soundscapes via iterative pseudo-labeling with confidence thresholds significantly boosted performance despite dataset shift
- A modified hybrid loss penalizing high-confidence negative predictions effectively reduced false positives compared to standard CE or FocalBCE
- Filtering training data to remove human speech and non-target segments improved model diversity and cleanliness
- Class-weighted post-processing averaging, scaled by reference frequency, helped balance recall across rare and common classes

## Critical findings
- Family-level Insecta labels from Xeno-Canto worsened results when used as-is, but assigning unique species-level labels improved performance
- CrossEntropy outperformed BCE/Focal losses because it penalizes overrepresented negative classes more strongly when rare positives are misclassified
- Stochastic Depth only improved results during self-training, not supervised training, confirming the noisy student dynamic
- Applying augmentations or processing directly on raw audio waves hurts model performance
- Simple models overfit extremely quickly, making public LB tracking essential since k-fold cross-validation is invalid for this competition's validation scheme
- Training on random 5-second audio segments yields better results than using the first 5 seconds
- Power adjustment for low-ranked classes improved the LB score but was ultimately dropped due to overfitting risks
- Using the same random seeds across model groups (Group A and B) did not negatively impact the final ensemble performance
- Self-distillation proved highly effective for enriching secondary labels, with iterative rounds consistently boosting LB scores across all backbone architectures
- Skipping random 5s periods without vocalization actually reduced LB score, suggesting models use those segments for generalization
- Validation correlation with Public LB broke down completely after breaking the 0.9 milestone
- Selecting checkpoints based on best validation ROC AUC outperformed using the last or averaged checkpoints
- Using the last Xeno-Canto snippet for pretraining performed worse than using older 2024 pretrains
- Removing all human voice from audio degraded model performance, so only 50% was removed instead
- Ideas that failed on the public leaderboard (like resampling pseudo-labels by probability and adjusting probabilities by max/mean) actually improved private leaderboard scores
- Multi-threaded ONNX inference led to timeouts despite appearing viable, revealing a hidden deployment bug
- Standard CE loss with ConvNeXt produced excessive false positives, necessitating a custom negative penalty loss to stabilize training
- Despite public notebooks favoring ConvNeXt, the author found SED + CE loss to be the strongest baseline architecture for this specific setup
- Leaderboard instability prevented optimal model combination selection, highlighting the risk of relying on LB metrics for ensembling decisions

## What did not work
- Concatenating pseudo-labeled data into training batches separately failed to improve results
- Using Beta distribution parameters too low for MixUp blending weights suppressed meaningful signals
- Training deeper models for the dedicated Amphibia/Insecta group dropped scores
- Raising the minimum samples per species to 5 for the dedicated model dropped scores
- Standard BCE with various weight types underperformed compared to FocalBCE, and using the first 5 seconds of audio segments was less effective than random 5-second segments
- CNN-based models
- 1D models
- Too many data augmentations
- Training on soft labels of the main training data
- Using pretrained weights from the latest Xeno Canto snippet
- Using additional iNaturalist or XC data
- Additional augmentations like Time Flip
- raw signal model
- ensemble with rank average
- rms sample with energy weight
- use common name as auxiliary target
- lower rank power postprocessing
- add guassian noise into raw signal
- Using standard CE loss with ConvNeXt led to a lot of noisy predictions (many false positives)

## Notable individual insights
- rank 1 (1st Place Solution: Multi-Iterative Noisy Student Is All You Need): Applying a power transform greater than 1 to pseudo-label probabilities effectively mitigates noise accumulation in multi-iterative self-training.
- rank 2 (2nd Place. Journey Down the Rabbit Hole of Pseudo Labels): Validation correlation with Public LB broke down completely after breaking the 0.9 milestone, highlighting the risk of relying on LB metrics for ensembling decisions.
- rank 5 (5th place solution: Self-Distillation is All You Need): Many bird calls in the training data were unlabeled because recorders focused on target species, making accurate secondary label assignment the core challenge.
- rank 9 (9th place solution): Removing all human voice from audio degraded model performance, so only 50% was removed instead.
- rank 10 (10th Solution): A modified hybrid loss penalizing high-confidence negative predictions effectively reduced false positives compared to standard CE or FocalBCE.
- rank 9 (9th place solution): Multi-threaded ONNX inference led to timeouts despite appearing viable, revealing a hidden deployment bug.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Solution: Multi-Iterative Noisy Student Is All You Need]]
- #2 [[solutions/rank_02/solution|2nd Place. Journey Down the Rabbit Hole of Pseudo Labels]]
- #5 [[solutions/rank_05/solution|5th place solution: Self-Distillation is All You Need]]
- #9 [[solutions/rank_09/solution|9th place solution]]
- #10 [[solutions/rank_10/solution|10th Solution]]
- ? [[solutions/rank_xx_573066/solution|Recipe to Public LB 0.872]]

## GitHub links
- [myso1987/BirdCLEF-2025-5th-place-solution](https://github.com/myso1987/BirdCLEF-2025-5th-place-solution) _(solution)_ — from [[solutions/rank_05/solution|5th place solution: Self-Distillation is All You Need]]
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad) _(library)_ — from [[solutions/rank_05/solution|5th place solution: Self-Distillation is All You Need]]
- [VSydorskyy/BirdCLEF_2025_2nd_place](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place) _(solution)_ — from [[solutions/rank_02/solution|2nd Place. Journey Down the Rabbit Hole of Pseudo Labels]]
- [frednam93/FilterAugSED](https://github.com/frednam93/FilterAugSED) _(library)_ — from [[solutions/rank_09/solution|9th place solution]]

## Papers cited
- [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/abs/1911.04252)
- [Design Choices for Enhancing Noisy Student Self-Training](https://openaccess.thecvf.com/content/WACV2024/papers/Radhakrishnan_Design_Choices_for_Enhancing_Noisy_Student_Self-Training_WACV_2024_paper.pdf)
- [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

# recodai-luc-scientific-image-forgery-detection: top public notebooks

The top-voted notebooks primarily focus on inference pipelines and baseline architectures for scientific image forgery detection. They predominantly leverage frozen DINOv2 vision transformers paired with lightweight CNN decoders for pixel-wise segmentation. Common themes include gradient-enhanced adaptive thresholding, test-time augmentation, and RLE-encoded mask generation, with a few exploring hybrid architectures, ELA integration, and regularization techniques to combat overfitting.

## Common purposes
- inference
- baseline
- training

## Models
- DINOv2-base
- DINOv2-large
- DINOv2
- DinoTinyDecoder
- DinoMultiTask
- Mask R-CNN (MobileNetV3-Small backbone)
- ProgressiveBilinearDecoder
- Custom CNN Encoder-Decoder
- Custom CNN Decoder
- DinoLargeDecoderRegularized

## Frameworks
- PyTorch
- Torch
- Transformers
- OpenCV
- OpenCV-Python
- NumPy
- Pandas
- PIL
- Matplotlib
- Scikit-learn
- Torchvision
- Albumentations

## CV strategies
- Holdout split 80/20
- Train/val split 80/20
- Train/test split 20% (stratified by class)

## Preprocessing
- Resize to 256x256
- Resize to 512x512
- Resize to 518x518
- Resize to 718x718
- Normalize pixel values to [0, 1]
- Convert to RGB
- Mask binarization (>0)
- DINOv2 AutoImageProcessor normalization
- Max-projection for 3D mask arrays
- Dynamic ViT grid handling and CLS token stripping
- ImageNet normalization
- Multi-channel mask collapse via np.any
- Nearest-neighbor mask resizing
- Assign zero masks to authentic images
- ELA re-compression at JPEG quality 90 with absolute difference scaling

## Augmentations
- Horizontal flip TTA
- Vertical flip TTA
- 90-degree rotation (with inverse mask alignment)
- Albumentations Resize (256x256)
- Albumentations Normalize (ImageNet stats)
- ToTensorV2
- Confidence-weighted TTA

## Loss functions
- BCEWithLogitsLoss
- CrossEntropyLoss
- Mask R-CNN composite loss (sum of classification, box regression, and mask losses)
- RegularizedDiceBCELoss (BCE + Dice loss weight=0.4 + L2 regularization weight=1e-5)

## Ensemble patterns
- Averages sigmoid-activated predictions from a DINOv2-based segmenter and a custom CNN encoder-decoder
- 50/50 balanced ensemble of DINOv2-base and DINOv2-large model predictions
- Combines RGB model predictions, ELA model predictions (70% weight), and direct ELA heatmap signals (30% weight) into a single probability map

## Post-processing
- Gradient-enhanced adaptive thresholding (Sobel gradients blended with alpha=0.35)
- Gaussian blur (3x3)
- Morphological closing (5x5) and opening (3x3)
- Area and mean probability filtering for binary classification
- Grid search for threshold optimization
- RLE encoding for submission
- Fallback to 'authentic' when mask area or confidence falls below thresholds
- Temperature scaling (1.2-1.5) for probability calibration
- Dynamic morphological kernel sizing based on predicted mask area
- Brightness/contrast-adaptive thresholding
- Score thresholding (0.4/0.5) with logical OR mask combination

## What worked
- Adding dropout, weight decay, gradient clipping, and L2 regularization prevented overfitting seen in the unregularized large model
- Adaptive thresholds and confidence-weighted TTA improved robustness across varying image conditions

## What did not work
- The unregularized large model overfitted, dropping validation score from ~0.324 to 0.246
- Conservative thresholds and high precision requirements occasionally miss forged images with low activation

## Critical findings
- High-contrast images require lower thresholds (area_thr=60, mean_thr=0.18) to detect subtle forgeries, while low-contrast images need higher thresholds to avoid false positives
- Temperature scaling and confidence weighting significantly reduce overconfidence in calibration

## Notable techniques
- Frozen DINOv2 feature extraction with spatial reshaping
- Lightweight Conv2d decoder for pixel-wise prediction
- Gradient-enhanced adaptive thresholding
- Area/mean confidence filtering to suppress false positives
- Grid search on validation set for threshold optimization
- Custom RLE encoder for Kaggle submission
- Progressive bilinear upsampling decoder with intermediate convolutions
- Dual-condition classification rule (area < 400 or mean prob < 0.35) to filter false positives
- Confidence-weighted TTA (horizontal/vertical flips)
- Dynamic morphological kernel sizing based on predicted mask area
- Brightness/contrast-adaptive thresholding
- Temperature scaling for probability calibration
- ELA re-compression at JPEG 90 to expose compression history anomalies
- Fusing raw ELA heatmap with model predictions via weighted averaging

## Notable individual insights
- votes 462 (Scientific Image Forgery Detection | DINOv2): Inference-only pipeline using frozen DINOv2 with a lightweight decoder and gradient-enhanced adaptive thresholding to produce pixel-wise masks.
- votes 277 (EDA + R-CNN model): Demonstrates a beginner-friendly Mask R-CNN baseline with MobileNetV3-Small, showing how to convert binary segmentation masks to bounding boxes for training.
- votes 212 (RLE metric + Simple submission): Highlights a fallback strategy to submit 'authentic' when mask area or confidence falls below thresholds, justifying it for highly imbalanced datasets.
- votes 143 (image detection 🔬🔬 | CNN | score[0.324]🔥🔥): Reveals that high-contrast images require lower thresholds (area_thr=60, mean_thr=0.18) to catch subtle forgeries, while low-contrast images need higher thresholds to avoid false positives.
- votes 126 (🔥 Scientific image detection | CNN | 0.325): Introduces Error Level Analysis (ELA) re-compression at JPEG 90 to expose compression artifacts, fusing the ELA heatmap with model predictions via weighted averaging.
- votes 288 (DINOv2-Base Multi-Task Forgery Detection): Shows how to dynamically handle CLS token stripping and reconstruct spatial feature maps from ViT hidden states for multi-task learning.

## Notebooks indexed
- #462 votes [[notebooks/votes_01_ravaghi-scientific-image-forgery-detection-dinov2/notebook|Scientific Image Forgery Detection | DINOv2]] ([kaggle](https://www.kaggle.com/code/ravaghi/scientific-image-forgery-detection-dinov2))
- #343 votes [[notebooks/votes_02_pankajiitr-scientific-forensics-dinov2-cnn-ipynb/notebook|Scientific-Forensics-DINOv2-CNN.ipynb]] ([kaggle](https://www.kaggle.com/code/pankajiitr/scientific-forensics-dinov2-cnn-ipynb))
- #288 votes [[notebooks/votes_03_djamilabenchikh-dinov2-base-multi-task-forgery-detection/notebook|DINOv2-Base Multi-Task Forgery Detection]] ([kaggle](https://www.kaggle.com/code/djamilabenchikh/dinov2-base-multi-task-forgery-detection))
- #277 votes [[notebooks/votes_04_antonoof-eda-r-cnn-model/notebook|EDA + R-CNN model]] ([kaggle](https://www.kaggle.com/code/antonoof/eda-r-cnn-model))
- #217 votes [[notebooks/votes_05_pankajiitr-cnn-dinov2-progressivebilineardecoder/notebook|CNN–DINOv2 ProgressiveBilinearDecoder ]] ([kaggle](https://www.kaggle.com/code/pankajiitr/cnn-dinov2-progressivebilineardecoder))
- #212 votes [[notebooks/votes_06_antonoof-rle-metric-simple-submission/notebook|RLE metric + Simple submission]] ([kaggle](https://www.kaggle.com/code/antonoof/rle-metric-simple-submission))
- #206 votes [[notebooks/votes_07_maheenriaz1122-cnn-dinov2-hybrid-c0bb37/notebook|CNN–DINOv2 Hybrid c0bb37]] ([kaggle](https://www.kaggle.com/code/maheenriaz1122/cnn-dinov2-hybrid-c0bb37))
- #195 votes [[notebooks/votes_08_djamilabenchikh-cnn-dinov2-hybrid/notebook|CNN–DINOv2 Hybrid]] ([kaggle](https://www.kaggle.com/code/djamilabenchikh/cnn-dinov2-hybrid))
- #143 votes [[notebooks/votes_09_abhishekgodara-image-detection-cnn-score-0-324/notebook|image detection 🔬🔬 | CNN | score[0.324]🔥🔥]] ([kaggle](https://www.kaggle.com/code/abhishekgodara/image-detection-cnn-score-0-324))
- #126 votes [[notebooks/votes_10_hossam82-scientific-image-detection-cnn-0-325/notebook|🔥 Scientific image detection  | CNN | 0.325]] ([kaggle](https://www.kaggle.com/code/hossam82/scientific-image-detection-cnn-0-325))

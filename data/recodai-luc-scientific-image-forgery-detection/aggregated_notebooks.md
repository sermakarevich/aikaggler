# recodai-luc-scientific-image-forgery-detection: top public notebooks

The community's top-voted notebooks predominantly focus on inference pipelines and lightweight training strategies for scientific image forgery detection, heavily favoring hybrid architectures that pair frozen DINOv2 vision transformers with custom CNN decoders. Contributors emphasize robust post-processing techniques—such as gradient-enhanced adaptive thresholding, morphological refinement, and confidence-based filtering—to suppress false positives on authentic images. Additionally, several notebooks explore foundational baselines, multi-task setups, and forensic feature integration like Error Level Analysis (ELA) to maximize segmentation precision and submission reliability.

## Common purposes
- inference
- training
- tutorial

## Competition flows
- Inference pipeline: loads test images, extracts frozen DINOv2 features, passes through lightweight decoder, applies TTA and gradient-based post-processing, outputs RLE-encoded masks or "authentic" labels.
- Inference with threshold optimization: loads train/test data, extracts features, applies TTA and adaptive masking, optimizes thresholds via grid search, generates RLE CSV.
- Multi-task training: loads authentic/forged images with masks, extracts features, trains dual heads (segmentation + classification), evaluates with IoU/Dice/Accuracy, generates submission.
- Tutorial/EDA: performs EDA on variable image sizes, trains Mask R-CNN for segmentation, generates RLE submission.
- Hybrid segmentation: loads data, extracts features, trains CNN decoder on binary masks, applies adaptive post-processing, generates CSV.
- Baseline validation: loads test images, runs TTA on pre-trained models, applies filtering, encodes to RLE, validates pipeline with authentic-only baseline.
- Regularized training: loads data, splits sets, trains regularized CNN decoder on frozen encoder, generates submission with calibrated TTA and ensembling.
- Forensic fusion: loads test images, applies ELA and DINOv2 segmentation, fuses signals, applies post-processing, exports RLE masks.

## Data reading
- Images loaded via PIL.Image.open, converted to RGB, resized to fixed dimensions, normalized to [0,1] or ImageNet mean/std, permuted to CHW tensor format, and moved to target device.
- Masks loaded from .npy files using np.load, with 3D masks collapsed by taking the maximum over the channel axis or converted to binary via thresholding/np.any.
- Paths constructed using pathlib.Path, os.listdir, and sorted lists from train/test directories.
- Sample submission CSV loaded via pandas for merging outputs.

## Data processing
- Resizing to fixed dimensions (256x256, 512x512, 518x518, 718x718)
- Normalization to [0, 1] or ImageNet mean/std
- TTA via horizontal/vertical flips and 90-degree rotations
- Gradient enhancement using Sobel filters
- Gaussian blurring
- Adaptive thresholding (mean + 0.3*std or brightness/contrast-based)
- Morphological closing/opening operations
- Mask resizing with nearest-neighbor interpolation
- Converting multi-channel masks to binary via thresholding/np.any
- Converting binary masks to bounding boxes via OpenCV contours
- RLE encoding for submission
- ELA computation (re-compression at JPEG 90, absolute differences, scaling)
- Confidence/area filtering
- Deterministic seeding for reproducibility

## Features engineering
- Domain-derived ELA heatmap and compression artifact score
- Gradient magnitude features from probability maps for adaptive thresholding

## Models
- DINOv2 (base/large, frozen)
- DinoTinyDecoder (custom CNN/convolutional decoder)
- DinoSegmenter (hybrid wrapper/segmenter)
- ProgressiveBilinearDecoder (custom CNN)
- Linear classification head
- Mask R-CNN with MobileNetV3-Small backbone
- Custom CNN encoder-decoder

## Frameworks used
- PyTorch
- Transformers
- OpenCV
- Pandas
- NumPy
- PIL/Pillow
- scikit-learn
- Torchvision
- Albumentations
- Matplotlib
- tqdm
- pathlib/os

## Loss functions
- BCEWithLogitsLoss
- CrossEntropyLoss
- Mask R-CNN internal composite loss (classification, box regression, mask IoU)
- RegularizedDiceBCELoss (BCE + Dice + L2 regularization)

## CV strategies
- Holdout split (80/20 train/validation) via sklearn/torch random_split
- Holdout split applied independently to authentic and forged classes
- Holdout split with deterministic seeding for reproducibility

## Ensembling
- Test-time augmentation (TTA) averaging flipped/rotated views
- Model ensembling (mean/weighted averaging of CNN and DINOv2 outputs)
- Confidence-weighted TTA with temperature scaling for probability calibration
- Multi-signal fusion (RGB model, ELA model, and ELA heatmaps via weighted sum)

## Insights
- Frozen foundation models like DINOv2 can serve as strong feature extractors for specialized tasks without requiring encoder fine-tuning.
- Combining gradient-based enhancement with adaptive thresholding and morphological operations effectively refines raw segmentation outputs.
- Confidence filtering based on mask area and mean probability significantly reduces false positives for authentic images.
- Freezing a large self-supervised vision transformer and attaching a lightweight CNN decoder is an efficient way to adapt pre-trained representations for forensic segmentation tasks.
- Gradient magnitude enhancement significantly improves mask boundary precision compared to raw probability maps.
- Sequentially training segmentation and classification heads allows independent optimization of localization and authenticity prediction without gradient interference.
- Mask R-CNN can be repurposed for forgery detection by framing tampered regions as detectable object instances.
- ELA effectively highlights compression inconsistencies that correlate with image forgery.
- Fusing forensic signals with deep features improves localization over using either alone.
- Temperature scaling improves probability calibration, making model outputs more reliable for thresholding.

## Critical findings
- The unregularized Large model previously overfitted and scored lower (0.246) than the simpler Base model (0.324), highlighting the risk of training large architectures on small datasets without regularization.
- Conservative decision logic requiring minimum area, mean probability, and max probability thresholds is necessary to avoid false positives on authentic images.
- ELA is only reliable for images originally saved as JPEGs, as applying it to lossless images introduces artifacts.
- The author empirically determined a hidden JPEG detection threshold of 1.2 and an ELA compression quality of 90 for optimal artifact amplification.

## What did not work
- The unregularized Large model overfitted during training, resulting in a lower validation F1 (0.246) compared to the Base model (0.324), prompting the switch to a regularized decoder and conservative post-processing.

## Notable individual insights
- votes 143 (image detection 🔬🔬 | CNN | score[0.324]🔥🔥): Unregularized large models overfit on small datasets; switching to regularized decoders and conservative post-processing thresholds significantly improved validation F1.
- votes 126 (🔥 Scientific image detection | CNN | 0.325): ELA reliably highlights compression inconsistencies but is only valid for originally JPEG-saved images; empirically found optimal ELA compression quality at 90 and a hidden JPEG detection threshold of 1.2.
- votes 212 (RLE metric + Simple submission): Submitting 'authentic' for all images is a fast, effective baseline for highly imbalanced segmentation tasks to validate pipelines and understand metric behavior.
- votes 288 (DINOv2-Base Multi-Task Forgery Detection): Sequentially training segmentation and classification heads allows independent optimization of localization and authenticity prediction without gradient interference.
- votes 462 (Scientific Image Forgery Detection | DINOv2): Confidence filtering based on mask area and mean probability significantly reduces false positives for authentic images.
- votes 277 (EDA + R-CNN model): Mask R-CNN can be repurposed for forgery detection by framing tampered regions as detectable object instances, enabling a beginner-friendly segmentation approach.

## Notebooks indexed
- #462 votes [[notebooks/votes_01_ravaghi-scientific-image-forgery-detection-dinov2/notebook|Scientific Image Forgery Detection | DINOv2]] ([kaggle](https://www.kaggle.com/code/ravaghi/scientific-image-forgery-detection-dinov2))
- #345 votes [[notebooks/votes_02_pankajiitr-scientific-forensics-dinov2-cnn-ipynb/notebook|Scientific-Forensics-DINOv2-CNN.ipynb]] ([kaggle](https://www.kaggle.com/code/pankajiitr/scientific-forensics-dinov2-cnn-ipynb))
- #288 votes [[notebooks/votes_03_djamilabenchikh-dinov2-base-multi-task-forgery-detection/notebook|DINOv2-Base Multi-Task Forgery Detection]] ([kaggle](https://www.kaggle.com/code/djamilabenchikh/dinov2-base-multi-task-forgery-detection))
- #277 votes [[notebooks/votes_04_antonoof-eda-r-cnn-model/notebook|EDA + R-CNN model]] ([kaggle](https://www.kaggle.com/code/antonoof/eda-r-cnn-model))
- #217 votes [[notebooks/votes_05_pankajiitr-cnn-dinov2-progressivebilineardecoder/notebook|CNN–DINOv2 ProgressiveBilinearDecoder ]] ([kaggle](https://www.kaggle.com/code/pankajiitr/cnn-dinov2-progressivebilineardecoder))
- #212 votes [[notebooks/votes_06_antonoof-rle-metric-simple-submission/notebook|RLE metric + Simple submission]] ([kaggle](https://www.kaggle.com/code/antonoof/rle-metric-simple-submission))
- #206 votes [[notebooks/votes_07_maheenriaz1122-cnn-dinov2-hybrid-c0bb37/notebook|CNN–DINOv2 Hybrid c0bb37]] ([kaggle](https://www.kaggle.com/code/maheenriaz1122/cnn-dinov2-hybrid-c0bb37))
- #195 votes [[notebooks/votes_08_djamilabenchikh-cnn-dinov2-hybrid/notebook|CNN–DINOv2 Hybrid]] ([kaggle](https://www.kaggle.com/code/djamilabenchikh/cnn-dinov2-hybrid))
- #143 votes [[notebooks/votes_09_abhishekgodara-image-detection-cnn-score-0-324/notebook|image detection 🔬🔬 | CNN | score[0.324]🔥🔥]] ([kaggle](https://www.kaggle.com/code/abhishekgodara/image-detection-cnn-score-0-324))
- #126 votes [[notebooks/votes_10_hossam82-scientific-image-detection-cnn-0-325/notebook|🔥 Scientific image detection  | CNN | 0.325]] ([kaggle](https://www.kaggle.com/code/hossam82/scientific-image-detection-cnn-0-325))

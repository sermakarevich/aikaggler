# waveform-inversion: cross-solution summary

This competition focused on seismic velocity inversion, framing the task as an image-to-image regression problem where models must predict subsurface velocity models from seismic waveforms. Winning approaches heavily leveraged physics-grounded synthetic data generation, architectural shifts toward Vision Transformers and hybrid Convformer/CAFormer models, and advanced refinement techniques like regularized Full Waveform Inversion (FWI) and differentiable simulator optimization. Success consistently depended on bridging the CV-LB gap through massive synthetic augmentation, iterative pseudo-labeling, and family-specific post-processing or checkpoint ensembling.

## Competition flows
- Reshaping non-aligned seismic/velocity data into square images for Vision Transformer training with iterative pseudo-data generation and style filtering.
- Training ConvNeXt on full-resolution data for multi-variant seismic task submission.
- Training full-resolution CAFormer with custom decoder on preprocessed 72x72 datasets.
- Processing data through modified OpenFWI forward model, synthetic velocity blending, and multi-stage ViT training with Muon optimizer.
- Splitting data into 8 folds, training CaFormer/ConvNeXtV2 with flat LR, dynamic CUDA-accelerated wave propagation augmentation, and median checkpoint ensembling.
- Feeding seismograms into a DL model for initial velocity estimation, then refining via regularized FWI with custom CUDA kernels and dataset-specific priors.
- Processing data through modified CaFormer with physics-based velocity scaling, forward modeling augmentation, linear head supervision, and median TTA.
- Loading preprocessed 72x72 data, training HGNet-V2-Unet with flips/EMA, and evaluating across dataset types.
- Generating 10M synthetic samples via noise/Mixup/DDIM, training ConvNeXtV2/CaFormer, and fine-tuning on test-derived synthetic data.
- Augmenting velocity images with 15x synthetic seismic data, training Convformer/Caformer UNets with staged LR decay, and refining via differentiable simulator gradient optimization.
- Converting synthetic velocity to seismic via GPU-accelerated tool, training interlaced ViT+2D conv model with MAE+confidence loss, and applying family-specific post-processing.
- Training ConvNeXt UNet, converting test predictions to pseudo-labels via forward model for iterative fine-tuning, and refining via L1 backpropagation before blending.

## Data processing
- Resizing/interpolation (350x350, 896x896, 72x72, 500x70, 288x70)
- FiveCrop, RandomAffine, horizontal/vertical flips, stretches, CutMix, MixUp
- Synthetic velocity blending & velocity scaling augmentation (shrink/expand/pad/truncate)
- Zeroing last row of seismic data, padding with zeros, truncation
- Dynamic shift/scale/rotate, custom intensity shifting, Gaussian noise with rotations/scaling/shifts
- On-the-fly wave propagation via custom CUDA kernel
- Forward modeling to generate seismic data from velocity predictions / pseudo-labels
- In-batch semi-supervised data generation, pseudo-labeling test data, iterative fine-tuning cycles
- Downcasting to float16/bfloat16, storing in specific formats
- Clipping outputs to (1500, 4500), rounding to int, bias restoration via grouping
- TTA with median aggregation (10 alpha values)
- Cosine LR schedule with adjusted sample weights
- Perlin noise blob generation, artificial faults, wave deformations, smooth multiplications
- Upweighting challenging classes in synthetic data

## Models
- UNet variants (modified, public, ConvNeXt UNet)
- Vision Transformer (ViT) variants (encoder-decoder, ViT with RoPE, custom ~100M ViT+2D conv, interlaced transformer/conv blocks, MaxViT 2D conv)
- EVA02 (EVA02-small, eva02_base_patch16_clip_224, eva02_large_patch14_clip_224)
- Dinov2 (vit_small_patch14_reg4_dinov2.lvd142m)
- ConvNeXt / ConvNeXtV2 variants (ConvNeXt, ConvNeXtV2-Base, convnextv2_huge.fcmae, ConvNeXt Small/Large)
- CAFormer variants (CAFormer, CaFormer, caformer_b36.sail_in22k_ft_in1k, CaFormer-B36)
- Convformer
- HGNet-V2-Unet
- Variational auto-encoder
- Linear head
- Custom decoder (pixel shuffle, SCSE, intermediate convolutions)

## Frameworks used
- pytorch
- torch compile
- PyTorch

## Loss functions
- MAE
- L1 loss
- MAE + confidence loss
- Total Variation loss (p < 1)
- FWI cost function (prior + L2 residual)
- Gaussian Process prior
- Simulation error minimization (seis_pred vs seis_true)

## CV strategies
- Holdout split (440k train, 30k val)
- Data split by file into 8 folds (initially focusing on CurveFault_B)
- Validation on 20 files from a public notebook (CV-LB correlation breakdown noted)
- Holdout split of 10K samples (1K per family) for validation and ensemble weights

## Ensembling
- Top4 model blending with style-specific prediction replacement
- Multi-stage model blending (20%+30%+50% ratio) with value clipping
- Median of last 10 checkpoints per run, averaged across runs
- TTA with 10 velocity scaling alphas aggregated via median
- Cross-architecture ensembling (ConvNeXtV2-Base + CaFormer-B36) across training stages
- Hill-climb optimization for model combination with family-wise gradient-based refinement
- Multi-checkpoint ensembling (~10 per model) with classifier-derived family-specific weights, clipping, rounding, and bias restoration
- Weighted average blending of fine-tuned model with teammates' contributions

## Notable individual insights
- rank 1 (1st Place Solution): Rotary Positional Embeddings (RoPE) were a major bottleneck in EVA02, making Dinov2 a superior backbone choice.
- rank 3 (3rd place solution): Switching from AdamW to MuonWithAuxAdam dropped validation MAE from 9.x to 7.x, marking the biggest performance takeaway.
- rank 9 (9th place - Custom CUDA kernel for wave propagation): The original CPU-based wave propagation took ~3s per target, making on-the-fly augmentation impossible until a custom CUDA kernel provided a ~100x speedup.
- rank 2 (2nd place solution with code: refinement with FWI): FWI converges significantly better when initialized close to the true solution, making a rough deep learning prior essential.
- rank 5 (Team FAMAS. 5th place solution.): A differentiable forward simulator can be leveraged for post-processing by optimizing predictions to minimize the discrepancy between simulated and ground-truth seismic outputs.
- rank 6 (6th Place Solution Summary): Test-time fine-tuning on synthetic data derived from the test set provided a substantial leaderboard improvement in the final hours.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Solution]]
- #2 [[solutions/rank_02/solution|2nd place solution with code: refinement with FWI]]
- #3 [[solutions/rank_03/solution|3rd place solution]]
- #4 [[solutions/rank_04/solution|4th place solution for the GWI competition  ]]
- #5 [[solutions/rank_05/solution|Team FAMAS. 5th place solution.]]
- #6 [[solutions/rank_06/solution|6th Place Solution Summary]]
- #9 [[solutions/rank_09/solution|9th place - Custom CUDA kernel for wave propagation]]
- #10 [[solutions/rank_10/solution|10th Place Solution (monnu part)]]
- #20 [[solutions/rank_20/solution|20th solution - Caformer + data generation]]
- ? [[solutions/rank_xx_579841/solution|ConvNeXt Approach - [CV 31.9 LB 36.4]]]
- ? [[solutions/rank_xx_582785/solution|CAFormer Improved - [CV 24.2 LB 28.8]]]
- ? [[solutions/rank_xx_578305/solution|HGNet-V2 Encoder - [CV 55.6 LB 60.9]]]

## GitHub links
- [tascj/kaggle-waveform-inversion](https://github.com/tascj/kaggle-waveform-inversion) _(solution)_ — from [[solutions/rank_03/solution|3rd place solution]]
- [lu-group/fourier-deeponet-fwi](https://github.com/lu-group/fourier-deeponet-fwi) _(reference)_ — from [[solutions/rank_20/solution|20th solution - Caformer + data generation]]
- [tensorflow/models](https://github.com/tensorflow/models) _(reference)_ — from [[solutions/rank_04/solution|4th place solution for the GWI competition  ]]
- [shlomoron/GWI-solution](https://github.com/shlomoron/GWI-solution) _(solution)_ — from [[solutions/rank_04/solution|4th place solution for the GWI competition  ]]

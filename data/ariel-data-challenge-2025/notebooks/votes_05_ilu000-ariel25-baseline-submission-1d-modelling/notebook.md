# ARIEL25 ⚡ Baseline Submission 1D modelling

- **Author:** Pascal Pfeiffer
- **Votes:** 212
- **Ref:** ilu000/ariel25-baseline-submission-1d-modelling
- **URL:** https://www.kaggle.com/code/ilu000/ariel25-baseline-submission-1d-modelling
- **Last run:** 2025-07-09 16:06:13.837000

---

```python
!pip install scikit_learn==1.5.1 --no-index --find-links=/kaggle/input/ariel24-pip-installs
```

```python
import os

# Set to "1" to directly force test df, this is much quicker to commit
os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"
```

```python
%%writefile preprocess.py

import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import multiprocessing as mp
from astropy.stats import sigma_clip
import os
import torch
import torch.nn.functional as F


ROOT = "/kaggle/input/ariel-data-challenge-2025/"
VERSION = "v2"

BINNING = 15

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    MODE = "test"
else:
    MODE = "train"


sensor_sizes_dict = {
    "AIRS-CH0": [[11250, 32, 356], [32, 356]],
    "FGS1": [[135000, 32, 32], [32, 32]],
}  # input, mask

# 16 center pixels, rest is just noise
cl = 8
cr = 24


def get_gain_offset():
    """
    Get the gain and offset for a given planet and sensor

    Unlike last year's challenge, all planets use the same adc_info.
    We can just hard code it.
    """
    gain = 0.4369
    offset = -1000.0
    return gain, offset


def read_data(planet_id, sensor, mode):
    """
    Read the data for a given planet and sensor
    """
    # get all noise correction frames and signal
    signal = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_signal_0.parquet",
        engine="pyarrow",
    )
    dark_frame = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_0/dark.parquet",
        engine="pyarrow",
    )
    dead_frame = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_0/dead.parquet",
        engine="pyarrow",
    )
    linear_corr_frame = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_0/linear_corr.parquet",
        engine="pyarrow",
    )
    flat_frame = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_0/flat.parquet",
        engine="pyarrow",
    )
    # read_frame = pd.read_parquet(
    #     f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration/read.parquet",
    #     engine="pyarrow",
    # )

    # reshape to sensor shape and cast to float64
    signal = signal.values.astype(np.float64).reshape(sensor_sizes_dict[sensor][0])[
        :, cl:cr, :
    ]
    dark_frame = dark_frame.values.astype(np.float64).reshape(
        sensor_sizes_dict[sensor][1]
    )[cl:cr, :]
    dead_frame = dead_frame.values.reshape(sensor_sizes_dict[sensor][1])[cl:cr, :]
    flat_frame = flat_frame.values.astype(np.float64).reshape(
        sensor_sizes_dict[sensor][1]
    )[cl:cr, :]
    # read_frame = read_frame.values.reshape(sensor_sizes_dict[sensor][1])
    linear_corr = linear_corr_frame.values.astype(np.float64).reshape(
        [6] + sensor_sizes_dict[sensor][1]
    )[:, cl:cr, :]

    return (
        signal,
        dark_frame,
        dead_frame,
        linear_corr,
        flat_frame,
        # read_frame,
    )


def ADC_convert(signal, gain, offset):
    """
    Step 1: Analog-to-Digital Conversion (ADC) correction

    The Analog-to-Digital Conversion (adc) is performed by the detector to convert the
    pixel voltage into an integer number. We revert this operation by using the gain
    and offset for the calibration files 'train_adc_info.csv'.
    """

    return signal / gain + offset


def mask_hot_dead(signal, dead, dark):
    """
    Step 2: Mask hot/dead pixel

    The dead pixels map is a map of the pixels that do not respond to light and, thus,
    can't be accounted for any calculation. In all these frames the dead pixels are
    masked using python masked arrays. The bad pixels are thus masked but left
    uncorrected. Some methods can be used to correct bad-pixels but this task,
    if needed, is left to the participants.
    """

    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))

    # Set values to np.nan where dead or hot pixels are found
    signal[dead] = np.nan
    signal[hot] = np.nan
    return signal


def apply_linear_corr(c, signal):
    """
    Step 3: linearity Correction

    The non-linearity of the pixels' response can be explained as capacitive leakage
    on the readout electronics of each pixel during the integration time. The number
    of electrons in the well is proportional to the number of photons that hit the
    pixel, with a quantum efficiency coefficient. However, the response of the pixel
    is not linear with the number of electrons in the well. This effect can be
    described by a polynomial function of the number of electrons actually in the well.
    The data is provided with calibration files linear_corr.parquet that are the
    coefficients of the inverse polynomial function and can be used to correct this
    non-linearity effect.
    Using horner's method to evaluate the polynomial
    """
    assert c.shape[0] == 6  # Ensure the polynomial is of degree 5

    return (
        (((c[5] * signal + c[4]) * signal + c[3]) * signal + c[2]) * signal + c[1]
    ) * signal + c[0]


def clean_dark(signal, dark, dt):
    """
    Step 4: dark current subtraction

    The data provided include calibration for dark current estimation, which can be
    used to pre-process the observations. Dark current represents a constant signal
    that accumulates in each pixel during the integration time, independent of the
    incoming light. To obtain the corrected image, the following conventional approach
    is applied: The data provided include calibration files such as dark frames or
    dead pixels' maps. They can be used to pre-process the observations. The dark frame
    is a map of the detector response to a very short exposure time, to correct for the
    dark current of the detector.

    image - (dark * dt)

    The corrected image is conventionally obtained via the following: where the dark
    current map is first corrected for the dead pixel.
    """

    dark = torch.tile(dark, (signal.shape[0], 1, 1))
    signal -= dark * dt[:, None, None]
    return signal


def get_cds(signal):
    """
    Step 5: Get Correlated Double Sampling (CDS)

    The science frames are alternating between the start of the exposure and the end of
    the exposure. The lecture scheme is a ramp with a double sampling, called
    Correlated Double Sampling (CDS), the detector is read twice, once at the start
    of the exposure and once at the end of the exposure. The final CDS is the
    difference (End of exposure) - (Start of exposure).
    """

    return torch.subtract(signal[1::2, :, :], signal[::2, :, :])


def bin_obs(signal, binning):
    """
    Step 5.1: Bin Observations

    The data provided are binned in the time dimension. The binning is performed by
    summing the signal over the time dimension.
    """

    assert signal.shape[0] % binning == 0  # Ensure the binning is possible

    # cds_transposed = signal.transpose(0, 2, 1)
    cds_binned = torch.zeros(
        (
            signal.shape[0] // binning,
            signal.shape[1],
            signal.shape[2],
        ),
        device="cuda:0",
    )
    for i in range(signal.shape[0] // binning):
        cds_binned[i, :, :] = torch.sum(
            signal[i * binning : (i + 1) * binning, :, :], axis=0
        )
    return cds_binned


def correct_flat_field(flat, signal):
    """
    Step 6: Flat Field Correction

    The flat field is a map of the detector response to uniform illumination, to
    correct for the pixel-to-pixel variations of the detector, for example the
    different quantum efficiencies of each pixel.
    """

    return signal / flat


def nan_interpolation(tensor):
    # Assume tensor is of shape (batch, height, width)
    nan_mask = torch.isnan(tensor)

    # Replace NaNs with zero temporarily
    tensor_filled = torch.where(
        nan_mask, torch.tensor(0.0, device=tensor.device), tensor
    )

    # Create a binary mask (0 where NaNs were and 1 elsewhere)
    ones = torch.ones_like(tensor, device=tensor.device)
    weight = torch.where(nan_mask, torch.tensor(0.0, device=tensor.device), ones)

    # Perform interpolation by convolving with a kernel
    # using bilinear interpolation
    kernel = torch.ones(1, 1, 1, 3, device=tensor.device, dtype=tensor.dtype)

    # Apply padding to the tensor and weight to prevent boundary issues
    tensor_padded = F.pad(
        tensor_filled.unsqueeze(1), (1, 1, 0, 0), mode="replicate"
    ).squeeze(1)
    weight_padded = F.pad(weight.unsqueeze(1), (1, 1, 0, 0), mode="replicate").squeeze(
        1
    )

    # Convolve the filled tensor and the weight mask
    tensor_conv = F.conv2d(tensor_padded.unsqueeze(1), kernel, stride=1)
    weight_conv = F.conv2d(weight_padded.unsqueeze(1), kernel, stride=1)

    # Compute interpolated values (normalized by weights)
    interpolated_tensor = tensor_conv / weight_conv

    # Apply the interpolated values only to the positions of NaNs
    result = torch.where(nan_mask, interpolated_tensor.squeeze(1), tensor)

    return result


def process_planet(planet_id):
    """
    Process a single planet's data
    """
    axis_info = pd.read_parquet(ROOT + "axis_info.parquet")
    dt_airs = axis_info["AIRS-CH0-integration_time"].dropna().values

    for sensor in ["FGS1", "AIRS-CH0"]:
        # load all data for this planet and sensor
        signal, dark_frame, dead_frame, linear_corr, flat_frame = read_data(
            planet_id, sensor, mode=MODE
        )
        gain, offset = get_gain_offset()

        # Step 1: ADC correction
        signal = ADC_convert(signal, gain, offset)

        # Step 2: Mask hot/dead pixel
        signal = mask_hot_dead(signal, dead_frame, dark_frame)

        # clip at 0
        signal = signal.clip(0)

        # Step 3: linearity Correction
        signal = apply_linear_corr(
            torch.tensor(linear_corr).to("cuda:0"), torch.tensor(signal).to("cuda:0")
        )

        # Step 4: dark current subtraction
        if sensor == "FGS1":
            dt = torch.ones(len(signal), device="cuda:0") * 0.1
            dt[1::2] += 4.5
        elif sensor == "AIRS-CH0":
            dt = torch.tensor(dt_airs).to("cuda:0")
            dt[1::2] += 0.1

        signal = clean_dark(signal, torch.tensor(dark_frame).to("cuda:0"), dt)

        # Step 5: Get Correlated Double Sampling (CDS)
        signal = get_cds(signal)

        # Step 5.1: Bin Observations
        if sensor == "FGS1":
            signal = bin_obs(signal, binning=BINNING * 12)
        elif sensor == "AIRS-CH0":
            signal = bin_obs(signal, binning=BINNING)

        # Step 6: Flat Field Correction
        signal = correct_flat_field(torch.tensor(flat_frame).to("cuda:0"), signal)

        # Step 7: Interpolate NaNs (twice!)
        signal = nan_interpolation(signal)
        signal = nan_interpolation(signal)

        # Step 8: Sum over spatial axis
        if sensor == "FGS1":
            signal = torch.nanmean(signal, axis=[1, 2]).cpu().numpy()
        elif sensor == "AIRS-CH0":
            signal = torch.nanmean(signal, axis=1).cpu().numpy()

        # save the processed signal
        np.save(
            f"{planet_id}_{sensor}_signal_{VERSION}.npz",
            signal.astype(np.float64),
        )


if __name__ == "__main__":
    star_info = pd.read_csv(ROOT + f"/{MODE}_star_info.csv")
    star_info["planet_id"] = star_info["planet_id"].astype(int)
    star_info = star_info.set_index("planet_id")
    planet_ids = star_info.index.tolist()

    with mp.Pool(processes=4) as pool:
        list(tqdm(pool.imap(process_planet, planet_ids), total=len(planet_ids)))

    signal_train = []

    for planet_id in planet_ids:
        f_raw = np.load(f"{planet_id}_FGS1_signal_{VERSION}.npz.npy")
        a_raw = np.load(f"{planet_id}_AIRS-CH0_signal_{VERSION}.npz.npy")

        # flip a_raw
        signal = np.concatenate([f_raw[:, None], a_raw[:, ::-1]], axis=1)
        signal_train.append(signal)

    signal_train = np.array(signal_train)
    np.save(f"signal_{VERSION}.npy", signal_train, allow_pickle=False)

    print("Processing complete!")
```

```python
!python preprocess.py
```

```python
!rm -rf *FGS1_signal*
```

```python
!rm -rf *AIRS-CH0_signal*
```

```python
!ls
```

```python
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


MODEL_VERSION = "v1"
DATA_VERSION = "v2"
PRE_BINNED_TIME = 15

ROOT = "/kaggle/input/ariel-data-challenge-2025/"

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    MODE = "test"
else:
    MODE = "train"

star_info = pd.read_csv(ROOT + f"/{MODE}_star_info.csv")
star_info["planet_id"] = star_info["planet_id"].astype(int)
star_info = star_info.set_index("planet_id")
wavelengths = pd.read_csv(ROOT + "/wavelengths.csv")

signal_train = np.load(f"signal_{DATA_VERSION}.npy")
cut_inf, cut_sup = 36, 318
signal_train = np.concatenate(
    [signal_train[:, :, 0][:, :, None], signal_train[:, :, cut_inf:cut_sup]], axis=2
)
print(signal_train.shape)
# signal_train = signal_train.mean(axis=2)

def smooth_data(data, window_size):
    return savgol_filter(data, window_size, 3)  # window size 51, polynomial order 3


# find transit zones
def phase_detector(signal_orig, binning=15, smooth_window=11, verbose=False):
    signal = signal_orig.reshape(-1, binning).mean(-1)  # collapse by 15; 375
    signal = savgol_filter(signal, smooth_window, 2)  # smooth
    first_derivative = np.gradient(signal)
    phase1 = np.argmin(first_derivative)
    phase2 = np.argmax(first_derivative)

    if verbose:
        plt.plot(signal_orig, color="grey", alpha=0.5, label="original")
        plt.plot(signal, color="blue", alpha=0.9, label="smoothed")
        plt.axvline(phase1, color="r")
        plt.axvline(phase2, color="r")
        plt.show()
        plt.plot(first_derivative, color="green", alpha=0.9, label="first derivative")
        plt.show()

    assert phase1 < phase2
    assert phase1 >= 0
    assert phase2 <= signal.shape[0]
    return phase1 * binning, phase2 * binning


def get_breakpoints(x, pre_binned_time, verbose=False):
    bp = np.zeros(x.shape[0], dtype=np.int32)
    bp2 = np.zeros(x.shape[0], dtype=np.int32)
    for i in range(x.shape[0]):
        signal = x[i].mean(-1)
        p1, p2 = phase_detector(
            signal, binning=15 // pre_binned_time, smooth_window=19, verbose=verbose
        )
        bp[i] = p1
        bp2[i] = p2

    return [bp, bp2]


# breakpoint detection
all_bp, all_bp2 = get_breakpoints(signal_train, PRE_BINNED_TIME, verbose=False)


def poly_exp_fit(data, optimized_breakpoints, buffer_size, degree=3):
    # Define the three regions
    x1 = np.arange(optimized_breakpoints[0] - buffer_size)
    y1 = data[: optimized_breakpoints[0] - buffer_size]

    x2 = np.arange(
        optimized_breakpoints[0] + buffer_size,
        optimized_breakpoints[1] - buffer_size,
    )
    y2 = data[
        optimized_breakpoints[0] + buffer_size : optimized_breakpoints[1] - buffer_size
    ]

    x3 = np.arange(optimized_breakpoints[1] + buffer_size, len(data))
    y3 = data[optimized_breakpoints[1] + buffer_size :]

    # Concatenate the x-values and y-values for regions 1 and 3
    x_combined = np.concatenate([x1, x3])
    y_combined = np.concatenate([y1, y3])

    def fit_function(x, *params):
        poly_params = params[: degree + 1]
        y_fit = np.polyval(poly_params, x)
        return y_fit

    # Define the polynomial fit function with an additional shift parameter for region 2
    def fit_function_with_shift(x, shift, *poly_params):
        x1_adjusted = x[: len(x1)]
        x2_adjusted = x[len(x1) : len(x1) + len(x2)]
        x3_adjusted = x[len(x1) + len(x2) :]
        y1_fit = np.polyval(poly_params, x1_adjusted)
        y2_fit = np.polyval(poly_params, x2_adjusted) * shift
        y3_fit = np.polyval(poly_params, x3_adjusted)
        return np.concatenate([y1_fit, y2_fit, y3_fit])

    # Define the combined x-values (including region 2)
    x_combined_with_region2 = np.concatenate([x1, x2, x3])
    y_combined_with_region2 = np.concatenate([y1, y2, y3])

    # Initial guesses for the polynomial coefficients and shift
    poly_guess = np.polyfit(x_combined, y_combined, degree)

    p0 = list(poly_guess)

    initial_shift_guess = 1.0
    p0 = [initial_shift_guess] + list(p0)

    # Fit the polynomial and the shift using curve_fit
    popt, _ = curve_fit(
        fit_function_with_shift,
        x_combined_with_region2,
        y_combined_with_region2,
        p0=p0,
        maxfev=10000,
    )

    # Extract the optimized shift and polynomial coefficients
    optimized_shift = popt[0]
    assert optimized_shift > 0.8
    optimized_poly_params = popt[1:]

    return fit_function, optimized_poly_params, optimized_shift


def feature_engineering(signal_train):
    """Create a dataframe with two features from the raw data.

    Parameters:
    f_raw: ndarray of shape (n_planets, 67500)
    a_raw: ndarray of shape (n_planets, 5625)

    Return value:
    df: DataFrame of shape (n_planets, 2)
    """

    y_shifts = []

    for IDX in tqdm(range(len(signal_train))):
        data = signal_train[IDX]

        buffer_size_poly = 150 // PRE_BINNED_TIME

        optimized_breakpoints = [all_bp[IDX].item(), all_bp2[IDX].item()]

        fit_func, params, y_shift = poly_exp_fit(
            data[:, 1:].mean(1) / data[:, 1:].mean(1).mean(),
            optimized_breakpoints,
            buffer_size_poly,
            degree=2,
        )

        y_shifts.append(y_shift)

    y_shifts = np.array(y_shifts)

    df = pd.DataFrame(
        1 - y_shifts,
        index=star_info.index,
    )

    return df


df = feature_engineering(signal_train)
```

```python
df
```

```python
predictions = df.values
```

```python
predictions.shape
```

```python
star_info = pd.read_csv(ROOT + f"/{MODE}_star_info.csv")
star_info["planet_id"] = star_info["planet_id"].astype(int)
star_info = star_info.set_index("planet_id")
star_info
```

```python
def postprocessing(pred_array, index, sigma_pred):
    """Create a submission dataframe from its components

    Parameters:
    pred_array: ndarray of shape (n_samples, 283)
    index: pandas.Index of length n_samples with name 'planet_id'
    sigma_pred: series of length n_samples or float

    Return value:
    df: DataFrame of shape (n_samples, 566) with planet_id as index
    """
    if isinstance(sigma_pred, float):
        expanded_sigmas = np.ones(len(pred_array)) * sigma_pred
    else:
        expanded_sigmas = sigma_pred

    expanded_sigmas = np.repeat(expanded_sigmas[:, np.newaxis], 283, axis=1)
    if pred_array.shape[1] == 1:
        pred_array = np.repeat(pred_array, 283, axis=1)
    return pd.concat(
        [
            pd.DataFrame(
                pred_array.clip(0, None), index=index, columns=wavelengths.columns
            ),
            pd.DataFrame(
                expanded_sigmas,
                index=index,
                columns=[f"sigma_{i}" for i in range(1, 284)],
            ),
        ],
        axis=1,
    )
```

```python
sub_df = postprocessing(predictions, star_info.index, sigma_pred=0.0008)
```

```python
sub_df
```

```python
sub_df.to_csv('submission.csv')
```

```python
pd.read_csv("submission.csv")
```

```python
!rm signal_v0.npy preprocess.py
```

```python
!ls
```

```python
# Adapted from https://www.kaggle.com/code/metric/ariel-gaussian-log-likelihood
import numpy as np
import pandas as pd
import scipy.stats


class ParticipantVisibleError(Exception):
    pass


def competition_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    naive_mean: float,
    naive_sigma: float,
    sigma_true: float = 0.00001,
    row_id_column_name: str = "planet_id",
) -> float:
    """
    This is a Gaussian Log Likelihood based metric. For a submission, which contains
    the predicted mean (x_hat) and variance (x_hat_std), we calculate the Gaussian
    Log-likelihood (GLL) value to the provided ground truth(x). We treat each pair
    of x_hat, x_hat_std as a 1D gaussian, meaning there will be 283 1D gaussian
    distributions, hence 283 values for each test spectrum, the GLL value for one
    spectrum is the sum of all of them.

    Inputs:
        - solution: Ground Truth spectra (from test set)
            - shape: (nsamples, n_wavelengths)
        - submission: Predicted spectra and errors (from participants)
            - shape: (nsamples, n_wavelengths*2)
        naive_mean: (float) mean from the train set.
        naive_sigma: (float) standard deviation from the train set.
        sigma_true: (float) essentially sets the scale of the outputs.
    """

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if submission.min().min() < 0:
        raise ParticipantVisibleError("Negative values in the submission")
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f"Submission column {col} must be a number")

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths * 2:
        raise ParticipantVisibleError("Wrong number of columns in the submission")

    y_pred = submission.iloc[:, :n_wavelengths].values
    # Set a non-zero minimum sigma pred to prevent division by zero errors.
    sigma_pred = np.clip(
        submission.iloc[:, n_wavelengths:].values, a_min=10**-15, a_max=None
    )
    y_true = solution.values

    GLL_pred = np.sum(scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred))
    GLL_true = np.sum(
        scipy.stats.norm.logpdf(
            y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true)
        )
    )
    GLL_mean = np.sum(
        scipy.stats.norm.logpdf(
            y_true,
            loc=naive_mean * np.ones_like(y_true),
            scale=naive_sigma * np.ones_like(y_true),
        )
    )

    submit_score = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)
    return float(np.clip(submit_score, 0.0, 1.0))
```

```python
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    train_labels = pd.read_csv(ROOT + "/train.csv", index_col="planet_id")
    
    gll_score = competition_score(
        train_labels.copy().reset_index(),
        sub_df.copy().reset_index(),
        naive_mean=train_labels.values.mean(),
        naive_sigma=train_labels.values.std(),
    )

    print(f"# Estimated competition score: {gll_score:.3f}")
```
# How to query Google Earth Engine data using Kaggle

- **Author:** Paul Mooney
- **Votes:** 49
- **Ref:** paultimothymooney/how-to-query-google-earth-engine-data-using-kaggle
- **URL:** https://www.kaggle.com/code/paultimothymooney/how-to-query-google-earth-engine-data-using-kaggle
- **Last run:** 2025-05-15 17:05:36.977000

---

# How to query Google Earth Engine data using Kaggle

## Step 1: Pass your credentials to the Earth Engine API

To use the [Google Earth Engine API](https://github.com/google/earthengine-api) you need to first create and download a Google Earth Engine Service Account Key from [Google Cloud Platform](https://cloud.google.com/gcp). Your private key .JSON file can then be saved as a Kaggle Dataset and accessed according to the code snippets provided below. To download your private key .JSON file you will need to:
 1. Create a Google Cloud Project (if you don't have one)
  - Go to the Google Cloud Console
  - Click on the project dropdown at the top of the page
  - Click "New Project"
  - Enter a project name and click "Create"
 2. Enable the Earth Engine API
  - Go to the Google Cloud API Library
  - Search for "Earth Engine"
  - Select "Earth Engine API"
  - Click "Enable"
 3. Create a Service Account
  - In the Google Cloud Console, navigate to "IAM & Admin" > "Service Accounts"
  - Click "Create Service Account" at the top
  - Fill in the service account details
  - Click "Create and Continue"
 4. Create and Download the Key File
  - In the Service Accounts list, find your newly created service account
  - Click on the three dots (⋮) at the end of the row for your service account
  - Select "Manage keys"
  - Click "Add Key" > "Create new key"
  - Choose "JSON" as the key type
  - Click "Create"
  - The private key JSON file will automatically download to your computer. This is the only time you can download this key, so store it securely.
 5. Register the Service Account with Earth Engine
  - Go to the Earth Engine Registration Page
  - Log in with your Google account that has Earth Engine access
  - Enter your service account email (it will look like service-account-name@project-id.iam.gserviceaccount.com)
  - Click "Register"
 6. Upload the Key to Kaggle
  - Go to Kaggle Datasets
  - Click "New Dataset"
  - Upload your JSON key file
  - Make sure to set the dataset to "Private" to keep your credentials secure
  - Click "Create"
  - Take note of the file path where your JSON file is being stored 



For more detail see [here](https://developers.google.com/earth-engine/guides/python_install) and [here](https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api).

```python
import ee
import io
import os
import matplotlib.pyplot as plt
import urllib.request
from kaggle_secrets import UserSecretsClient
from PIL import Image
```

```python
def load_secret(name):
    """Loads secret from Colab/Kaggle."""

    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            pass 
    else:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception: 
            pass

    return 'Secret not found'
```

```python
iam_service_account = load_secret('iam_service_account') # the address of your project's IAM service account
ee_credentials_json = load_secret('ee_credentials') # the file path for the JSON file containing the relevant credentials
ee_creds = ee.ServiceAccountCredentials(iam_service_account, ee_credentials_json) # fetch your service account credentials
ee.Initialize(ee_creds) # initialize earth engine using your service account credentials
```

# Step 2: Access data from the Earth Engine API

## Print the elevation of Sugarloaf Mountain

```python
coordinates = [-43.1566, -22.9486]
dem = ee.Image('USGS/SRTMGL1_003')
xy = ee.Geometry.Point(coordinates)
elev = dem.sample(xy, 30).first().get('elevation').getInfo()
print('Sugarloaf Mountain elevation (m):', elev)
```

## View a satellite image of Sugarloaf Mountain, Rio de Janeiro

```python
coordinates = [-43.1566, -22.9486]
sugarloaf = ee.Geometry.Point(coordinates)
region = sugarloaf.buffer(3000).bounds()

collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(sugarloaf) \
    .filterDate('2024-01-01', '2024-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .sort('CLOUDY_PIXEL_PERCENTAGE')

sentinel = collection.first()

if sentinel is None:
    raise ValueError("No suitable Sentinel-2 image found.")

vis_params = {
    'bands': ['B4', 'B3', 'B2'],
    'min': 0,
    'max': 3000,
    'gamma': 1.3
}

url = sentinel.getThumbURL({
    'region': region, 
    'dimensions': '800', 
    'format': 'jpg',
    'bands': vis_params['bands'],
    'min': vis_params['min'],
    'max': vis_params['max']
})

response = urllib.request.urlopen(url)
img_data = response.read()
img = Image.open(io.BytesIO(img_data))

plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.title('Sugarloaf Mountain - Sentinel-2 Image')
plt.axis('off')
plt.annotate('Sugarloaf Mountain', xy=(400, 400), xytext=(500, 350),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.show()

print("Image coordinates: ", coordinates)
```
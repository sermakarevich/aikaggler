# openai-to-z-challenge: top public notebooks

The top-voted notebooks primarily focus on practical tutorials for integrating cloud geospatial APIs (Google Earth Engine, OpenAI) with Kaggle's environment, alongside exploratory spatial analyses of archaeological datasets. Several entries demonstrate advanced remote sensing pipelines that combine pre-computed satellite embeddings, NDVI filtering, and vision-language models to triage large-scale search areas. Collectively, they emphasize secure credential management, reproducible geospatial workflows, and hybrid human-AI strategies for archaeological prospecting.

## Common purposes
- tutorial
- eda
- utility
- other

## Competition flows
- Demonstrates how to securely store and retrieve an OpenAI API key on Kaggle, initialize the OpenAI client, and send a text prompt to generate a model completion.
- Loads multi-source archaeological site coordinates and OpenStreetMap river data, computes 10km spatial buffers around rivers, and outputs an interactive HTML map visualizing the spatial overlap between sites and waterways.
- Loads an OpenAI API key from Kaggle secrets, sends a prompt to GPT-4o-mini to generate a hypothetical plan for analyzing satellite imagery, and prints the model's response.
- Loads known archaeological site coordinates and pre-computed Major TOM embeddings, matches them spatially using a BallTree, trains an XGBoost classifier on the extracted features, and scores the full embedding dataset to generate ranked probability maps of potential geoglyph locations.
- Loads multiple Excel files containing archaeological site coordinates, radiocarbon dates, and skeletal/funerary metadata, then generates static and interactive visualizations to explore regional distributions and sample characteristics without building any predictive models.
- Authenticates with GEE and OpenAI APIs, fetches a Sentinel-2 satellite image, encodes it to base64, and queries GPT-4o-mini for a textual description and location/time guess.
- Loads GEE credentials from Kaggle secrets, initializes the Earth Engine API, queries elevation and satellite imagery data, and displays the results.
- Downloads Sentinel-2 satellite imagery over a defined AOI, tiles and filters it using NDVI thresholds, runs batched inference with a quantized Gemma-3 vision-language model to identify potential earthworks, and maps the results for interactive exploration.
- Loads and fuses topographic, hydrological, and ecological datasets to filter candidate tiles, applies spatial clustering on GEDI canopy anomalies to generate bounding boxes, and uses GPT-4.1 to triage and plan follow-up surveys for five landscape anomaly clusters.

## Data reading
- Reads five CSV files (mound_villages_acre.csv, casarabe_sites_utm.csv, amazon_geoglyphs_sites.csv, submit.csv, science.ade2541_data_s2.csv) using pandas, handling mixed coordinate systems (UTM vs lat/lon) and inconsistent column naming conventions across datasets.
- Reads GeoJSON and Shapefile boundaries using geopandas
- Loads embedding metadata and feature vectors from multiple Parquet files using polars and duckdb
- Extracts coordinates and embedding arrays into NumPy/Pandas structures
- pd.read_excel() is used to load four separate Excel files: 'Brazilian radiocarbon bioarchaeological samples.xlsx', 'individuals.xlsx', 'human_individuals_funerary.xlsx', and 'sampled_skeletal_part.xlsx'.
- ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') filtered by bounds, date range, and cloud cover, then downloaded as a JPEG via urllib.request
- Queries GEE datasets via `ee.Image` and `ee.ImageCollection` APIs, filters by bounds and dates, fetches image thumbnails via HTTP URL, and parses the binary response into a PIL Image.
- pystac_client queries AWS Earth Search for Sentinel-2 L2A scenes
- rasterio reads B08, B04, B03 bands as float32 arrays, scales reflectance to uint8, and writes a false-color composite GeoTIFF
- GeoJSON and Shapefile inputs loaded via geopandas.read_file
- DEM tiles loaded via rasterio.open
- All geometries reprojected to EPSG:3857
- CSV files via pandas.read_csv
- GeoJSON and GDB files via geopandas.read_file
- SRTM DEM via OpenTopography API downloaded as GeoTIFF and opened with rasterio
- GBIF species occurrences via pygbif API
- GEDI L2A metrics via pre-downloaded GeoJSON

## Data processing
- Converts UTM coordinates to WGS84 lat/lon using pyproj transformers, drops rows with missing or invalid coordinates, samples large datasets for rendering performance, and reprojects geometries to UTM (EPSG:32719) for accurate metric buffering before converting back to WGS84.
- Reprojects geographic data to Web Mercator (EPSG:3857) for plotting
- Converts lat/lon to radians for haversine distance calculations
- Filters embeddings within a 1120m radius of known sites using BallTree
- Removes embedding IDs associated with multiple conflicting object types
- Stacks embeddings and labels into unified NumPy arrays
- Binarizes/multiplies labels for 3-class classification
- Basic cleaning includes resetting the index and sorting by calibrated date limits. No normalization, tokenization, or augmentation is applied.
- Encodes the PIL Image to a base64 string for API transmission
- Applies RGB band selection (B4, B3, B2), min/max scaling, and gamma correction for visualization
- Filters Sentinel-2 collection by geographic bounds, date range, and cloud cover (<20%), sorts by cloudiness, selects the clearest image, and applies RGB band mapping (B4, B3, B2) with min/max scaling and gamma correction for visualization.
- NDVI calculation from NIR and Red bands
- 1km² tiling into PNGs
- Filtering by mean NDVI < 0.2 and >2% black pixel threshold
- UTM to WGS84 coordinate conversion
- Batched image loading and prompt construction for VLM inference
- Coordinate reference system transformations (EPSG:4326 ↔ EPSG:3857)
- Spatial buffering (9 km around rivers, 20 km around palm points)
- Tile grid generation and spatial joins (gpd.sjoin)
- DEM masking per tile to compute mean elevation
- Elevation quantile filtering (min to top 10%)
- Centroid extraction, spatial clustering, convex hull/bounding box generation
- Canopy height calculation (rh100 - rh95) and outlier scoring

## Features engineering
- Generates 10km spatial buffer polygons around extracted river networks using geopandas and shapely, and calculates implicit site-to-river proximity for spatial overlay analysis.
- River-to-geoglyph distance
- Tile mean elevation and elevation quantile cutoffs
- Palm tree proximity buffers
- GEDI canopy height (rh100 - rh95)
- HDBSCAN outlier scores and DBSCAN cluster labels
- Bounding box coordinates and elevation values per footprint

## Models
- GPT-4o-mini
- XGBoost
- Gemma-3-4B-Vision (4-bit quantized via Unsloth)
- DBSCAN
- HDBSCAN
- Major TOM (pre-trained encoder)
- OpenAI o3

## Frameworks used
- openai
- kaggle_secrets
- pandas
- numpy
- folium
- pyproj
- geopandas
- shapely
- requests
- pickle
- matplotlib
- seaborn
- openpyxl
- xgboost
- scikit-learn
- polars
- duckdb
- contextily
- rasterio
- plotly
- google-earth-engine
- PIL
- earthengine-api
- urllib
- unsloth
- transformers
- bitsandbytes
- accelerate
- torch
- datasets
- pystac-client
- pillow
- pygbif
- hdbscan
- scipy
- fiona

## Loss functions
- mlogloss

## CV strategies
- Holdout split (train_test_split with test_size=0.2, random_state=42)

## Insights
- Kaggle provides a native secrets manager that can be accessed programmatically via the kaggle_secrets package.
- A single helper function can be written to abstract secret retrieval across both Kaggle and Google Colab environments.
- The OpenAI Python SDK supports direct integration with Kaggle secrets for secure, production-like API usage.
- River proximity strongly correlates with ancient settlement locations in Amazonia.
- Tiled querying with local caching effectively bypasses OpenStreetMap API rate limits for large-scale spatial extraction.
- Standardizing heterogeneous coordinate formats is critical for multi-source archaeological data integration.
- Kaggle secrets can be securely accessed in both Kaggle kernels and Google Colab using a unified helper function.
- The OpenAI Python SDK allows straightforward integration of LLMs into Kaggle workflows for prompt-based tasks.
- API usage on Kaggle requires explicit credit management, as free tiers or personal keys may not automatically apply without configuration.
- Pre-computed SSL embeddings can effectively proxy spatial similarity for archaeological search when direct visual detection is limited by satellite resolution.
- A lightweight gradient boosting model trained on embedding features successfully distinguishes between archaeological land cover types and ranks unknown locations by similarity.
- The model's high performance is largely driven by detecting deforestation patterns correlated with glyph locations rather than the glyphs themselves.
- The dataset compiles radiocarbon and stable isotope data from 71 archaeological sites across Brazil, spanning from 18 kyr BP to modern times.
- Inhumation is the predominant body disposal type among the sampled individuals.
- Ribs and skulls are the most frequently sampled skeletal parts, with bone and teeth being the primary sample types.
- The Amazon region is underrepresented in the dataset, with only Amapá and Pará states contributing samples.
- Integrating GEE with multimodal LLMs enables zero-shot geospatial analysis without custom training.
- Base64 encoding is a reliable method for transmitting satellite imagery to OpenAI's API.
- Filtering Sentinel-2 data by cloud cover and sorting by percentage ensures high-quality input images for vision models.
- Securely passing GEE service account credentials to Kaggle kernels requires uploading the JSON key as a private dataset and loading it via `kaggle_secrets`.
- GEE collections can be efficiently filtered by geographic bounds, date ranges, and metadata like cloud cover percentage before downloading.
- Satellite imagery can be fetched and visualized directly in Python using GEE's thumbnail URL generation and standard image processing libraries.
- NDVI-based filtering effectively reduces the search space by excluding dense vegetation and empty tiles before running expensive VLM inference.
- Quantized open-source vision-language models can serve as cost-effective, local alternatives to proprietary APIs for large-scale remote sensing triage.
- Batch processing with HuggingFace Datasets significantly improves GPU throughput for multimodal inference pipelines.
- Pre-computed satellite embeddings drastically reduce storage and compute requirements compared to raw imagery.
- Spatial filtering combined with terrain visualization enables efficient large-scale archaeological screening.
- LLMs can effectively automate expert-level feature detection and methodological critique when provided with properly formatted visual and textual context.
- Integrating ecological proxies like peach palm distributions with vertical canopy structure data significantly improves the precision of archaeological site prediction in Amazonia.
- Deterministic spatial clustering and fixed random seeds ensure bit-for-bit reproducibility across pipeline runs.
- GPT-4.1 can effectively triage spatial anomaly clusters and generate actionable, domain-specific follow-up strategies for remote sensing campaigns.

## Critical findings
- Approximately 80% of archaeological sites fall within a 10km river buffer, while only 20% lie beyond this threshold.
- Large geospatial datasets require aggressive sampling to maintain interactive map rendering performance.
- Sentinel-2's 10m ground sampling distance is too coarse to directly distinguish geoglyphs, making embedding-space filtering a necessary workaround.
- The classifier inadvertently learns to detect cleared land (deforestation) rather than the archaeological features, highlighting a confounding variable in the results.
- Pre-Columbian settlements in the region strongly correlate with mid-order rivers and specific elevation ranges derived from prior machine learning predictions.
- Peach palm (Bactris gasipaes) groves serve as reliable living indicators of ancient agroforestry and human activity.
- GEDI-derived canopy height anomalies successfully flag potential earthworks or vegetation shifts that correlate with known geoglyph locations.

## What did not work
- A previous cosine similarity approach to find similar embeddings did not work well, prompting the switch to an XGBoost classifier which produced more plausible results.
- DEM and Hillshade analysis was abandoned because it required external API credentials that were not available.

## Notable individual insights
- votes 120 (I Follow Rivers - Visualizing 7500+ known Sites): River proximity strongly correlates with ancient settlement locations in Amazonia, with ~80% of sites falling within a 10km buffer.
- votes 62 (Major TOM Embedding Search): Pre-computed SSL embeddings effectively proxy spatial similarity for archaeological search, though the classifier inadvertently learned to detect deforestation patterns rather than the glyphs themselves.
- votes 42 (Starter - Amazon Basin Archaeological Search): NDVI-based filtering combined with quantized open-source vision-language models (Gemma-3) provides a cost-effective alternative to proprietary APIs for large-scale remote sensing triage.
- votes 35 (Checkpoint 2: Pixels, Palms, and the Past): Integrating ecological proxies like peach palm distributions with GEDI canopy height data significantly improves archaeological site prediction precision.
- votes 49 (How to ask GPT-4o about Google Earth Engine data): Base64 encoding enables reliable transmission of Sentinel-2 satellite imagery to multimodal LLMs for zero-shot geospatial analysis without custom training.

## Notebooks indexed
- #264 votes [[notebooks/votes_01_paultimothymooney-how-to-use-openai-models-on-kaggle/notebook|How to use OpenAI models on Kaggle]] ([kaggle](https://www.kaggle.com/code/paultimothymooney/how-to-use-openai-models-on-kaggle))
- #120 votes [[notebooks/votes_02_fafa92-i-follow-rivers-visualizing-7500-known-sites/notebook|I Follow Rivers - Visualizing 7500+ known Sites]] ([kaggle](https://www.kaggle.com/code/fafa92/i-follow-rivers-visualizing-7500-known-sites))
- #68 votes [[notebooks/votes_03_mpwolke-tad-jones-and-the-paititi-legend/notebook|Tad Jones and the Paititi Legend]] ([kaggle](https://www.kaggle.com/code/mpwolke/tad-jones-and-the-paititi-legend))
- #62 votes [[notebooks/votes_04_fnands-major-tom-embedding-search/notebook|Major TOM Embedding Search]] ([kaggle](https://www.kaggle.com/code/fnands/major-tom-embedding-search))
- #57 votes [[notebooks/votes_05_mpwolke-brazilian-archaeological-radiocarbon-dating/notebook|Brazilian Archaeological Radiocarbon dating]] ([kaggle](https://www.kaggle.com/code/mpwolke/brazilian-archaeological-radiocarbon-dating))
- #49 votes [[notebooks/votes_06_paultimothymooney-how-to-ask-gpt-4o-about-google-earth-engine-data/notebook|How to ask GPT-4o about Google Earth Engine data]] ([kaggle](https://www.kaggle.com/code/paultimothymooney/how-to-ask-gpt-4o-about-google-earth-engine-data))
- #49 votes [[notebooks/votes_07_paultimothymooney-how-to-query-google-earth-engine-data-using-kaggle/notebook|How to query Google Earth Engine data using Kaggle]] ([kaggle](https://www.kaggle.com/code/paultimothymooney/how-to-query-google-earth-engine-data-using-kaggle))
- #42 votes [[notebooks/votes_08_aaronbornstein-starter-amazon-basin-archaeological-search/notebook|🛰️ Starter - Amazon Basin Archaeological Search ]] ([kaggle](https://www.kaggle.com/code/aaronbornstein/starter-amazon-basin-archaeological-search))
- #37 votes [[notebooks/votes_09_fnands-search-the-amazon-with-remote-sensing-and-ai/notebook|🛰️Search the Amazon with Remote Sensing and AI🤖]] ([kaggle](https://www.kaggle.com/code/fnands/search-the-amazon-with-remote-sensing-and-ai))
- #35 votes [[notebooks/votes_10_ceeluna-checkpoint-2-pixels-palms-and-the-past/notebook| Checkpoint 2: Pixels, Palms, and the Past]] ([kaggle](https://www.kaggle.com/code/ceeluna/checkpoint-2-pixels-palms-and-the-past))

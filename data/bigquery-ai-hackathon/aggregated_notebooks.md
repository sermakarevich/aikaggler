# bigquery-ai-hackathon: top public notebooks

The community's top-voted notebooks primarily focus on BigQuery AI and SQL-based workflows for data engineering, semantic search, and time-series forecasting rather than traditional ML modeling. They emphasize leveraging cloud-native functions, external APIs (OpenAI, Vertex AI), and pandas for EDA, data processing, and visualization. The collection spans tutorials and utilities covering synthetic data generation, geospatial analysis, multimodal patent processing, and market sentiment tracking.

## Common purposes
- tutorial
- utility
- eda

## Competition flows
- Generates synthetic retail data via pandas, provisions BigQuery tables via SQL, runs analytical queries for segmentation and forecasting, and simulates AI embedding/search functions using SQL logic.
- Queries Stack Overflow public data via BigQuery, extracts and categorizes tickets/solutions using SQL, implements a custom word/category overlap similarity function, and outputs live matching demos with business impact metrics.
- Queries public OpenStreetMap geospatial data via BigQuery SQL, filters for specific amenity tags, and exports the results to pandas DataFrames for basic inspection and geospatial bounding box visualization.
- Queries historical temperature data from BigQuery's public NOAA dataset, applies three different forecasting methods (Prophet, BigQuery ML linear regression, and BigQuery AI.FORECAST), and outputs 30-day temperature forecasts with evaluation metrics.
- Generates text embeddings via OpenAI, loads them into BigQuery, creates a vector index, performs similarity search, and generates summaries using BigQuery AI functions.
- Reads raw patent PDFs from GCS via BigQuery Object Tables, uses Gemini models via SQL to extract structured knowledge graphs and dual embeddings, normalizes vectors, and builds a weighted semantic search index and portfolio analysis dashboard.
- Queries COVID-19 tabular data from BigQuery, generates semantic embeddings via Vertex AI, computes manual cosine similarity in SQL to find contextually similar records, and generates natural language summaries using BigQuery ML with Gemini Pro.
- Fetches Google Trends time-series data and RSS news articles, applies sentiment analysis and keyword extraction, then generates static and interactive visualizations along with a JSON export of the aggregated market intelligence.
- Reads synthetic survey data for visualization, then constructs and simulates quantum circuits using Qiskit to demonstrate state evolution, measurement outcomes, and simulation method comparisons.
- Fetches Google Trends and RSS news data for insurance-related keywords, performs sentiment and trend analysis, generates visualizations, and exports structured results to JSON.

## Data reading
- Synthetic data is created in-memory using pandas DataFrames and numpy random generators.
- Data is loaded into BigQuery via bigquery.Client.load_table_from_dataframe and client.query with CREATE OR REPLACE TABLE statements.
- Queries `bigquery-public-data.stackoverflow.posts_questions` and `posts_answers` directly via the `google-cloud-bigquery` Python client, converting results to pandas DataFrames.
- Uses the google.cloud.bigquery client to execute SQL queries against the bigquery-public-data.geo_openstreetmap dataset, converting query results directly into pandas DataFrames via .to_dataframe().
- Loads data from bigquery-public-data.noaa_gsod.gsod* using the google-cloud-bigquery Python client, filtering by station ID (stn = '488200') and year range (2020-2022), then converts the query result to a pandas DataFrame.
- Hardcoded text list converted to a pandas DataFrame and uploaded to BigQuery using `client.load_table_from_dataframe`.
- Loaded via BigQuery Object Tables pointing to gs://gcs-public-data--labeled-patents/*.pdf, with figure metadata joined from bigquery-public-data.labeled_patents.figures using GCS paths.
- Queries the covid_data table from BigQuery using the google.cloud.bigquery client, converts results to a pandas DataFrame, and limits the sample to 100 rows.
- Loads external data via the pytrends API for Google Trends time-series and regional metrics.
- Parses RSS XML feeds using feedparser and requests for news articles.
- Loads a local text file via standard Python open() and constructs synthetic DataFrames directly from hardcoded dictionaries containing country, experience, and quantum application metadata.
- Retrieves data dynamically via the pytrends API for Google Trends and feedparser for RSS feeds from insurance news websites; no local competition datasets are loaded.

## Data processing
- None explicitly implemented; relies on SQL ROUND, CASE, and DATE_DIFF functions for formatting and basic transformations.
- No explicit cleaning, normalization, or augmentation pipelines are present.
- Filters data by `accepted_answer_id IS NOT NULL`, `LENGTH(title) > 10`, `score >= 1`, and `creation_date >= '2020-01-01'`.
- Cleans text by lowercasing, removing non-alphanumeric characters via regex, and splitting into words.
- Filters stop words and short words during tokenization.
- Filters raw OSM node data using SQL WHERE clauses on nested all_tags arrays for specific amenity values; constructs a geospatial bounding box polygon using BigQuery GIS functions (ST_MAKEPOLYGON, ST_MAKELINE, ST_GEOGPOINT); attempts to intersect historical nodes with the bounding box (though the execution contains a pandas read_sql_query error).
- Filters out invalid temperature values (temp IS NOT NULL AND temp != 9999.9), creates a continuous daily date index, and fills missing values using time-based interpolation.
- None explicitly shown; relies on API-generated embeddings and BigQuery's native vector indexing.
- Multimodal parsing of PDFs using ML.GENERATE_TEXT for text and diagram descriptions; JSON cleaning of LLM outputs; L2 normalization of generated vectors; weighted averaging of patent context and component function embeddings via a custom JS UDF; comprehensive data quality validation (null rates, duplicates, schema consistency, statistical outlier detection).
- Formats each row into a single text string using CONCAT and TO_JSON_STRING, batches the text for Vertex AI embedding generation (batch size 5, 10s delay), pads failed embeddings with None, and exports the results to JSONL for BigQuery import.
- Strips HTML tags from news summaries using BeautifulSoup.
- Filters articles by a 30-day publication cutoff to ensure recency.
- Combines VADER and TextBlob polarity scores to assign sentiment labels.
- Calculates trend volatility and percentage changes over time.
- Aggregates keyword frequencies across titles and summaries.
- Transpiles quantum circuits for the Aer simulator; no explicit data cleaning or normalization is applied to the synthetic visualization data.
- Filters news articles to the last 30 days, strips HTML tags from summaries, combines VADER and TextBlob sentiment scores, calculates recent vs. historical trend percentages, handles missing data and API errors gracefully, and batches keyword requests to respect rate limits.

## Features engineering
- SQL-based customer segmentation (VIP/Premium/Regular based on loyalty/income).
- Demand prediction thresholds derived from rating and price ranges.
- Annualized CLV calculation using DATE_DIFF and total_spent.
- Visual compatibility scoring via string matching (LIKE) and weighted sums.
- Simulated embedding vectors via categorical mapping and Euclidean magnitude calculations.
- Extracts `issue_category` via regex-based CASE statements (error, database, authentication, api, payment, frontend, backend, general).
- Extracts `key_terms` by filtering words longer than 3 characters and excluding common stop words.
- Computes word overlap and key term overlap scores dynamically during query time.
- day_of_year
- year_num
- Text embeddings generated via OpenAI API (`text-embedding-ada-002`).
- Constructs a unified text representation for each record by concatenating geographic identifiers (Country_Region, Continent, WHO_Region) with the full JSON serialization of the row.

## Models
- Prophet
- Linear Regression (BigQuery ML)
- AI.FORECAST (BigQuery managed time-series model)
- text-embedding-ada-002
- BigQuery vector index model
- BigQuery AI text generation model
- gemini-2.5-flash
- gemini-embedding-001
- text-embedding-004
- gemini_pro_model

## Frameworks used
- pandas
- numpy
- matplotlib
- seaborn
- google-cloud-bigquery
- db-dtypes
- bigquery
- plotly
- prophet
- openai
- vertexai
- textblob
- vaderSentiment
- feedparser
- beautifulsoup4
- requests
- pytrends
- folium
- qiskit
- qiskit-aer
- pyvis
- ipywidgets

## Insights
- BigQuery's SQL-based AI functions can be effectively prototyped using standard SQL logic and window functions.
- Retail analytics workflows like segmentation and forecasting can be implemented efficiently without traditional ML frameworks.
- Visual similarity and semantic search can be approximated through categorical mapping and distance calculations in SQL.
- Traditional keyword search fails on paraphrased customer issues, but combining lexical overlap with categorical matching improves semantic retrieval.
- BigQuery's native SQL capabilities can handle text processing and similarity scoring without external ML libraries.
- Filtering by community validation metrics (score, accepted_answer_id) ensures high-quality training data.
- Public BigQuery datasets can be queried directly without local data downloads.
- OSM data stores tags in nested arrays, requiring UNNEST in SQL to filter by specific keys and values.
- BigQuery's built-in GIS functions enable geospatial filtering and bounding box creation directly in SQL.
- BigQuery ML enables training and prediction entirely in SQL without downloading data.
- BigQuery AI.FORECAST provides a zero-code, server-side alternative for rapid prototyping.
- Local Prophet implementation offers full control over seasonality parameters and component visualization.
- BigQuery AI natively supports vector similarity search and text generation without requiring external ML frameworks.
- External embedding APIs like OpenAI can be easily integrated into BigQuery workflows for scalable data indexing.
- SQL-based pipelines can automate the entire process from embedding generation to document summarization.
- BigQuery's native multimodal AI functions enable complete unstructured data processing and knowledge graph construction without external ML frameworks.
- Weighted averaging of patent-level context and component-level function embeddings significantly improves semantic search relevance by prioritizing specific technical roles.
- Technical diagrams contain critical architectural complexity that text-only analysis misses, as evidenced by outlier patents with high component counts.
- Cloud-native vector search and SQL-based portfolio metrics allow scalable, cost-effective strategic patent analysis.
- Semantic search effectively captures contextual relationships that keyword matching misses.
- BigQuery ML allows for fully serverless, scalable AI pipelines without managing external infrastructure.
- Combining vector embeddings with generative AI transforms complex tabular data into actionable, human-readable insights.
- Combining VADER and TextBlob polarity scores yields a more balanced sentiment label than relying on a single library.
- Google Trends regional data can be directly mapped to US state codes to quickly identify geographic hotspots for insurance-related searches.
- Filtering RSS feeds by a strict 30-day publication window prevents historical noise from skewing current market sentiment analysis.
- Different Qiskit Aer simulation methods (statevector, stabilizer, density matrix, etc.) can be directly compared for the same circuit to analyze output formats and computational behavior.
- Quantum states can be initialized programmatically using random statevectors, density matrices, stabilizers, or unitaries before circuit execution.
- Intermediate quantum states and unitaries can be saved and retrieved during circuit execution for detailed analysis.
- Combining search volume trends with news sentiment provides a more comprehensive view of market interest than either metric alone.
- Dynamic package installation and compatibility patches are necessary for maintaining external API integrations in restricted Kaggle environments.
- Batching API requests and adding sleep intervals prevents rate-limiting errors when fetching large keyword sets.
- Sentiment analysis can be effectively simplified by averaging compound scores from VADER and TextBlob for robust classification.

## Critical findings
- Patents with unusually high component counts are strongly correlated with the presence of technical diagrams, proving multimodal analysis is essential for capturing true invention complexity.
- AI-generated extraction achieved less than 1.2% null rates across key fields, demonstrating high reliability for production-grade knowledge graph construction.

## What did not work
- The author explicitly notes that using pd.read_sql_query on a pandas DataFrame caused an AttributeError because DataFrames lack a cursor attribute, and the initial attempt to find shoe stores in Milan failed due to incorrect bounding box coordinates and SQL logic.

## Notable individual insights
- votes 75 (ARIA: BigQuery AI for E-commerce): BigQuery's SQL-based AI functions can be effectively prototyped using standard SQL logic and window functions.
- votes 62 (Ticket IQ 🤖): Traditional keyword search fails on paraphrased customer issues, but combining lexical overlap with categorical matching improves semantic retrieval.
- votes 48 (Women's BigQuery AI? Shoes!!): OSM data stores tags in nested arrays, requiring UNNEST in SQL to filter by specific keys and values.
- votes 43 (Simple Tutorial: Weather Forecasting): BigQuery AI.FORECAST provides a zero-code, server-side alternative for rapid prototyping.
- votes 32 (BigQuery AI: The Patent Analyst Project): Patents with unusually high component counts are strongly correlated with the presence of technical diagrams, proving multimodal analysis is essential for capturing true invention complexity.
- votes 27 (Semantic COVID-19 Intelligence with BigQuery AI): Semantic search effectively captures contextual relationships that keyword matching misses.

## Notebooks indexed
- #75 votes [[notebooks/votes_01_erenata-aria-bigquery-ai-for-e-commerce/notebook|ARIA: BigQuery AI for E-commerce]] ([kaggle](https://www.kaggle.com/code/erenata/aria-bigquery-ai-for-e-commerce))
- #62 votes [[notebooks/votes_02_annastasy-ticket-iq/notebook|Ticket IQ 🤖]] ([kaggle](https://www.kaggle.com/code/annastasy/ticket-iq))
- #48 votes [[notebooks/votes_03_mpwolke-women-s-bigquery-ai-shoes/notebook|Women's BigQuery AI? Shoes!!]] ([kaggle](https://www.kaggle.com/code/mpwolke/women-s-bigquery-ai-shoes))
- #43 votes [[notebooks/votes_04_daosyduyminh-simple-tutorial-weather-forecasting/notebook| Simple Tutorial: Weather Forecasting]] ([kaggle](https://www.kaggle.com/code/daosyduyminh/simple-tutorial-weather-forecasting))
- #33 votes [[notebooks/votes_05_vet231199-multimodal-data-processing-with-bigquery-ai/notebook|Multimodal Data Processing with BigQuery AI]] ([kaggle](https://www.kaggle.com/code/vet231199/multimodal-data-processing-with-bigquery-ai))
- #32 votes [[notebooks/votes_06_fissalalsharef-bigquery-ai-the-patent-analyst-project/notebook|BigQuery AI: The Patent Analyst Project]] ([kaggle](https://www.kaggle.com/code/fissalalsharef/bigquery-ai-the-patent-analyst-project))
- #27 votes [[notebooks/votes_07_kavindhiranc-semantic-covid-19-intelligence-with-bigquery-ai/notebook|Semantic COVID-19 Intelligence with BigQuery AI]] ([kaggle](https://www.kaggle.com/code/kavindhiranc/semantic-covid-19-intelligence-with-bigquery-ai))
- #27 votes [[notebooks/votes_08_daniyalatta-full-testing-bigquery-ai-single-cell/notebook|Full Testing(BigQuery AI) Single Cell]] ([kaggle](https://www.kaggle.com/code/daniyalatta/full-testing-bigquery-ai-single-cell))
- #22 votes [[notebooks/votes_09_mos3santos-sensores-qu-nticos/notebook|Sensores Quânticos]] ([kaggle](https://www.kaggle.com/code/mos3santos/sensores-qu-nticos))
- #21 votes [[notebooks/votes_10_masonlai42-experimental/notebook|Experimental]] ([kaggle](https://www.kaggle.com/code/masonlai42/experimental))

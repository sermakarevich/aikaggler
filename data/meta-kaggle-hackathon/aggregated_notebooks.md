# meta-kaggle-hackathon: top public notebooks

The community's top-voted notebooks focus heavily on exploratory data analysis and platform meta-analysis using Kaggle's internal Meta Kaggle dataset, rather than traditional ML modeling. They systematically track historical trends in competition hosting, user demographics, solution velocity, and Python package adoption. The work emphasizes programmatic data extraction, temporal aggregation, and interactive visualization to reveal structural shifts in Kaggle's ecosystem and AI tooling landscape.

## Common purposes
- eda
- tutorial
- utility

## Competition flows
- Downloads the Meta Kaggle and Meta Kaggle Code datasets, parses their directory structures to extract notebook source code, and links code files to kernel/version metadata for hackathon exploration.
- Loads Kaggle Meta Kaggle and AI report CSVs, processes competition metadata and forum titles, and generates word clouds and HTML tables summarizing top solution discussions across featured competitions.
- Downloads the Meta-Kaggle dataset via kagglehub, parses and filters competition, submission, and team metadata using Polars and Pandas, aggregates temporal trends, and generates interactive HTML visualizations of host segments, reward structures, submission behavior, and team formation patterns.
- Downloads the Meta-Kaggle dataset, joins and filters submission/team/kernel/user tables via DuckDB, engineers cohort and engagement features, and generates interactive Plotly visualizations to track participation trends and cohort behaviors.
- Fetches public Kaggle profile and competition metadata via Meta Kaggle datasets -> aggregates and clusters user activity metrics -> generates interactive Plotly visualizations -> serves them through a Gradio web app.
- Downloads the MetaKaggle dataset via kagglehub, inspects its schema and user metadata, visualizes geographical user distribution, and applies statistical forecasting to predict future user growth.
- Downloads Kaggle Meta Kaggle datasets, calculates and aggregates solution speed metrics by year, detects temporal change points, and generates visualizations correlating speed trends with AI revolution events and platform milestones.
- Loads kernel metadata and source code from the Meta Kaggle dataset, extracts and aggregates Python package imports over time, and visualizes usage trends, popularity shifts, and co-import patterns.

## Data reading
- Uses kagglehub.dataset_download to fetch datasets.
- Loads CSV metadata tables (Kernels.csv, KernelVersions.csv) with pd.read_csv and sets Id as the index.
- Reads .ipynb files via codecs.open and parses them as JSON to extract cell contents.
- Reads .py files via Path.read_text().splitlines().
- Loads CSV files from `/kaggle/input/` using `pd.read_csv()`, including the 2023 Kaggle AI report and Meta Kaggle datasets (Competitions, ForumTopics, CompetitionTags, Tags).
- Parses date columns with `pd.to_datetime()` and sets display options for wide DataFrames.
- Loads CSV files (Competitions.csv, Submissions.csv, Teams.csv, TeamMemberships.csv) from the local Meta-Kaggle dataset path using polars.read_csv and pandas.read_csv.
- Parses dates using str.strptime in Polars and pd.to_datetime in Pandas with %m/%d/%Y or %m/%d/%Y %H:%M:%S formats.
- Loads multiple CSV files from the kaggle/meta-kaggle dataset via kagglehub and pd.read_csv.
- Executes SQL queries with duckdb to join submissions, teams, kernels, and user metadata.
- Downloads kaggle/meta-kaggle and kaggle/meta-kaggle-code via kagglehub.dataset_download()
- Reads specific CSV files (Users.csv, Teams.csv, Submissions.csv, Competitions.csv, CompetitionTags.csv, Tags.csv, KernelVersions.csv, Kernels.csv, KernelLanguages.csv) using pd.read_csv() with usecols and low_memory=False
- Uses kagglehub.dataset_download to fetch the kaggle/meta-kaggle and kaggle/meta-kaggle-code datasets.
- Loads CSV files (Competitions.csv, KernelVersions.csv, Submissions.csv, KernelVersionCompetitionSources.csv) from the downloaded kaggle/meta-kaggle and kaggle/meta-kaggle-code datasets using pd.read_csv, with explicit dtype handling for mixed-type columns.
- Loads Kernels.csv from the kaggle/meta-kaggle dataset via kagglehub with pandas adapter.
- Parses .ipynb files directly from the local filesystem using nbformat and ast to extract import statements.

## Data processing
- Filters .ipynb JSON structures to isolate only code cells.
- Maps kernel/version IDs to file paths using zero-padded string formatting and glob pattern matching.
- Extracts file extensions and suppresses pandas RuntimeWarnings.
- Filters competitions by `HostSegmentTitle` (Featured, Research, Recruitment) and sorts by enabled date.
- Merges tag metadata dataframes to map competition IDs to tag names.
- Applies string-based filtering to forum discussion titles to identify solution-related posts, excluding common false positives.
- Filters null/empty strings in categorical columns (HostSegmentTitle, RewardType).
- Parses and extracts year, month, and week from raw date columns, handling missing values with strict=False and errors='coerce'.
- Groups and aggregates counts/sums by year and host, pivots categorical host segments into time-series columns, and converts Polars DataFrames to Pandas for visualization.
- Performs memory cleanup using gc.collect() and sys.getsizeof to monitor DataFrame sizes.
- Floors dates to days, filters out benchmark/after-deadline submissions, staff users, and non-core competition segments.
- Removes rows with missing performance tiers or negative competition days, drops duplicates, and maps raw accelerator/tier IDs to categorical groups.
- Converts dates with pd.to_datetime(..., errors='coerce') and drops NaNs
- Maps numeric medal IDs to string labels
- Groups and aggregates data by weekly/monthly periods
- Fills missing text fields with empty strings before embedding
- Normalizes sentence embeddings and applies KMeans clustering
- Reduces high-dimensional tag features to 2D using UMAP
- Maps dataset country names to GeoJSON standards for geospatial visualization; minimal explicit cleaning or transformation is shown in the provided cells.
- Converts timestamp columns to datetime, drops rows with missing dates, filters kernels by vote count (>=1 or >=2) or creation date (>=2020/2022), merges datasets on kernel/competition IDs, calculates DaysToSolution, filters to 1-365 days, and groups by year to compute median, mean, count, and average votes.
- Samples kernels at fixed intervals (e.g., every 10 days) for scalability.
- Extracts top-level package names from import and from-import statements using AST parsing.
- Aggregates usage counts and ratios across sampled kernels.
- Filters packages with usage ratio > 0.001 for year-over-year comparison.
- Converts package counts to ordered categoricals for visualization.

## Features engineering
- Extracts temporal features (year, month, week) from raw date strings.
- Creates aggregated metrics (total USD rewards per year, submission counts per month/week, average team size per year).
- Pivots categorical host segments into time-series columns for aligned trend analysis.
- Computes user competition sequence (UserCompNumber), time gaps between competitions (UserDaysSinceLastComp), and competition duration usage percentage (CompDaysUsedPct).
- Derives team size categories, proxy engagement scores (raw and log-adjusted), cumulative medal counts, and cohort assignment year.
- Aggregates cumulative medal counts over time
- Calculates weekly submission frequency
- Extracts and counts competition tags for medal-winning teams
- Clusters competition titles/subtitles/overviews using sentence embeddings to infer competition domains
- Maps tags to domains via UMAP projections
- usage_ratio (package count / total kernels)
- rank by usage
- growth_ratio between years
- frequent_itemsets via Apriori
- association_rules (lift-based)

## Models
- all-MiniLM-L6-v2
- KMeans
- UMAP

## Frameworks used
- kagglehub
- pandas
- codecs
- glob
- json
- pathlib
- numpy
- matplotlib
- seaborn
- plotly
- wordcloud
- IPython
- polars
- plotly.express
- plotly.graph_objs
- plotly.subplots
- IPython.display
- duckdb
- plotly.graph_objects
- gradio
- sentence-transformers
- scikit-learn
- umap-learn
- networkx
- folium
- scipy
- graphviz
- tqdm
- plotnine
- mlxtend
- nbformat

## Insights
- The Meta Kaggle Code dataset organizes files in a hierarchical directory structure based on zero-padded kernel version IDs.
- Kaggle notebooks can be programmatically parsed by treating .ipynb files as JSON and filtering for code cells.
- Kernel metadata and version history are stored in separate CSV tables that can be joined via ScriptId and Id fields.
- Kaggle competitions provide a dynamic, evergreen evaluation framework that overcomes the limitations of static benchmarks in generative AI.
- Tracking competition metadata and community discussions reveals how model capabilities and evaluation standards evolve over time.
- Openly sharing solution discussions and performance metrics is critical for advancing empirical rigor in AI research.
- Competition hosting and prize distribution are heavily concentrated among specific host segments over time.
- USD-awarded competitions show distinct yearly trends that correlate with host segment activity.
- Submission volume exhibits clear monthly and weekly patterns, with notable differences between pre- and post-deadline behavior.
- Team formation and average team size have evolved consistently year-over-year, reflecting changing collaboration norms on the platform.
- Submission volume growth is primarily driven by Playground competitions and increased entry-level GPU adoption.
- Recent user cohorts compete more frequently, favor solo participation, and submit their final versions earlier than historical cohorts.
- Larger teams, medalists, and accelerator users tend to submit later in the competition timeline, while solo and non-medalists submit earlier.
- Adjusted engagement metrics reveal that higher-tier users still dominate relative engagement despite overall score increases.
- Most users only compete once, but a persistent minority returns, improves, and climbs performance tiers over time.
- Kaggle progress is better measured by consistent activity and skill evolution than just medals or prize money.
- Domain expertise can be effectively inferred from competition titles using sentence embeddings and clustering.
- Tracking submission consistency reveals true engagement patterns beyond medal counts.
- Medal timelines and tag breakdowns highlight personal strengths and growth trajectories over time.
- Kaggle's user base exhibits significant geographical diversity that can be effectively mapped and tracked over time.
- Statistical forecasting can reliably project future platform growth trends based on historical user registration data.
- Exploring the MetaKaggle schema reveals the complex relational structure underlying Kaggle's competition and user ecosystem.
- Solution development speed accelerated by approximately 49.5% from 2014 to 2025, dropping from a median of 317 days to 18 days.
- Major acceleration breakpoints align with the launch of Kaggle Kernels in 2017 and the ChatGPT revolution in 2022.
- Participation volume exploded exponentially post-2022, indicating AI democratization, while average solution quality slightly declined.
- The 2014 outlier is likely a platform artifact or data collection limitation rather than genuine user behavior due to tiny sample sizes and manual tracking.
- The Kaggle tooling landscape has shifted toward deep learning frameworks, with PyTorch and transformers rapidly gaining adoption while Keras and traditional NLP libraries decline.
- There is a negative correlation between the number of imported packages and kernel upvotes, suggesting that simpler notebooks are often more appreciated by the community.
- Package usage follows a long-tail distribution where a small number of libraries dominate adoption while most packages have very low usage ratios.
- Market basket analysis reveals strong, consistent co-import patterns that reflect standard data science and machine learning workflows on Kaggle.

## Critical findings
- Some users hold medals but remain in the Unranked tier due to missing profile engagement steps, not lack of skill.
- Playground competitions rarely award ranking points, but the dataset contains rare exceptions where medals were officially granted.
- Raw competition metadata may contain missing or miscounted medals compared to official Kaggle profiles, indicating potential data reliability issues.
- The 2014 median speed of 317 days is statistically an outlier driven by a tiny sample size (32 solutions) and pre-automation platform limitations.
- Average solution quality dropped significantly in the modern era despite massive volume growth, highlighting a clear speed-quality trade-off.
- Year-over-year acceleration is not linear; structural breaks occur sharply around 2017 and 2022 rather than gradually.

## Notable individual insights
- votes 121 (How to get started with the Meta Kaggle Hackathon): Demonstrates parsing .ipynb files as JSON to extract code cells and linking kernel metadata to version history via zero-padded directory structures.
- votes 59 (Gold Standard Evaluations: MetaKaggle): Highlights Kaggle competitions as dynamic, evergreen benchmarks that overcome static evaluation limits in generative AI.
- votes 28 (Kaggle Journeys: Cohorts and Competition Shifts): Reveals that raw metadata may miscount medals compared to official profiles, indicating data reliability issues.
- votes 25 (What if This Was Built in Kaggle?): Shows how sentence embeddings and UMAP clustering can effectively infer competition domains and map user expertise.
- votes 25 (From 317 to 18 Days: AI Speed Revolution Analysis): Quantifies a 49.5% acceleration in solution speed (317 to 18 days) with structural breaks aligning with Kaggle Kernels (2017) and ChatGPT (2022).
- votes 23 (Python Package Usage Trends & Patterns on Kaggle): Uncovers a negative correlation between imported package count and kernel upvotes, suggesting community preference for simplicity.

## Notebooks indexed
- #121 votes [[notebooks/votes_02_paultimothymooney-how-to-get-started-with-the-meta-kaggle-hackathon/notebook|How to get started with the Meta Kaggle Hackathon]] ([kaggle](https://www.kaggle.com/code/paultimothymooney/how-to-get-started-with-the-meta-kaggle-hackathon))
- #59 votes [[notebooks/votes_03_mpwolke-gold-standard-evaluations-metakaggle/notebook|Gold Standard Evaluations: MetaKaggle]] ([kaggle](https://www.kaggle.com/code/mpwolke/gold-standard-evaluations-metakaggle))
- #28 votes [[notebooks/votes_05_dnkumars-metakaggle7-contests-rewards/notebook|MetaKaggle7|Contests & Rewards]] ([kaggle](https://www.kaggle.com/code/dnkumars/metakaggle7-contests-rewards))
- #28 votes [[notebooks/votes_06_stevensio-kaggle-journeys-cohorts-and-competition-shifts/notebook|Kaggle Journeys: Cohorts and Competition Shifts]] ([kaggle](https://www.kaggle.com/code/stevensio/kaggle-journeys-cohorts-and-competition-shifts))
- #25 votes [[notebooks/votes_07_ahsuna123-what-if-this-was-built-in-kaggle/notebook|What if This Was Built in Kaggle? ]] ([kaggle](https://www.kaggle.com/code/ahsuna123/what-if-this-was-built-in-kaggle))
- #25 votes [[notebooks/votes_08_dnkumars-metakaggle-user-demographics-forecast/notebook|MetaKaggle|User 🏰 Demographics & Forecast]] ([kaggle](https://www.kaggle.com/code/dnkumars/metakaggle-user-demographics-forecast))
- #25 votes [[notebooks/votes_09_fernandosr85-from-317-to-18-days-ai-speed-revolution-analysis/notebook|From 317 to 18 Days: AI Speed Revolution Analysis]] ([kaggle](https://www.kaggle.com/code/fernandosr85/from-317-to-18-days-ai-speed-revolution-analysis))
- #23 votes [[notebooks/votes_10_alijalali4ai-python-package-usage-trends-patterns-on-kaggle/notebook|Python Package Usage Trends & Patterns on Kaggle]] ([kaggle](https://www.kaggle.com/code/alijalali4ai/python-package-usage-trends-patterns-on-kaggle))

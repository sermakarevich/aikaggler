# BigQuery AI: The Patent Analyst Project

- **Author:** Veysel Serifoglu
- **Votes:** 32
- **Ref:** fissalalsharef/bigquery-ai-the-patent-analyst-project
- **URL:** https://www.kaggle.com/code/fissalalsharef/bigquery-ai-the-patent-analyst-project
- **Last run:** 2025-10-22 20:48:15.977000

---

Licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
© 2025 [Veysel Serifoglu]
https://creativecommons.org/licenses/by/4.0/

# The AI Patent Analyst: From Unstructured PDFs to a Queryable Knowledge Graph

--- 

##### To experience the full power of this project, please visit our live, interactive Streamlit application. It provides a polished user interface for the semantic search engine and the strategic analysis dashboards.

-   **Live Demo:** [**https://patent-search-analytics.streamlit.app/**](https://patent-search-analytics.streamlit.app/)

---

## 1. High-Level Summary

This project solves the critical challenge of analyzing unstructured patent PDFs by building an end-to-end pipeline that transforms them into a structured, queryable Knowledge Graph entirely within Google BigQuery.

The final solution is an interactive analysis engine that delivers significant cost savings by automating tasks that would otherwise require hundreds of hours of expensive expert analysis from patent lawyers or R&D engineers. It answers:

*   **Deep Architectural Analysis:** Use standard SQL with `UNNEST` and `GROUP BY` to discover the most common design patterns and technical component connections across hundreds of patents.

*   **Component Search:** Go beyond patent-level search to find specific, functionally similar technical parts across different domains (e.g., "find a mechanism for encrypting data").

*   **Quantitative Portfolio Analysis:** Compare patent applicants by the complexity (average component count) and breadth (number of domains) of their innovations.

--- 

## 2. The Workflow: A Multi-Stage AI Pipeline

Our solution follows a three-stage process, showcasing a powerful combination of BigQuery's multimodal, generative, and vector search capabilities.

### Stage 1: Multimodal Data Processing (🖼️ Pioneer)
We use **Object Tables** to directly read and process raw PDFs from Cloud Storage. The Gemini model is then used with `ML.GENERATE_TEXT` to analyze the both the text and the technical diagrams within the PDFs.

### Stage 2: Generative Knowledge Graph Extraction (🧠 Architect)
The consolidated patent text is fed into the `AI.GENERATE_TABLE` function. A custom prompt instructs the AI to act as an expert analyst, extracting a structured table of high-level insights (`invention_domain`, `problem_solved`) and a detailed, nested graph of all technical components, their functions, and their interconnections.

### Stage 3: Component-Level Semantic Search (🕵️‍♀️ Detective)
To enable deep discovery, we build a novel search engine that understands context. We use `ML.GENERATE_EMBEDDING` to create two separate vectors:
1.  One for the patent's high-level context (title, abstract)
2.  Another for each component's specific function

These vectors are mathematically averaged into a single, final vector for each component via BigQuery's UDF (User-Defined Functions).

Finally, `VECTOR_SEARCH` is used on these combined vectors, creating a powerful search that returns highly relevant, context-aware results.

---

## 3. Dataset Overview
- **403 PDFs** (197 English, others in FR/DE) at `gs://gcs-public-data--labeled-patents/*.pdf`.
- **Tables**: `extracted_data` (metadata), `invention_types` (labels), `figures` (91 diagram coordinates).
- **Source**: [Labeled Patents](https://console.cloud.google.com/marketplace/product/global-patents/labeled-patents?inv=1&invt=Ab5j9A&project=bq-ai-patent-analyst&supportedpurview=organizationId,folder,project) (1TB/mo free tier).

---

## 4. Code
*   **Notebook & Repository:** [https://github.com/veyselserifoglu/bq-ai-patent-analyst/blob/main/notebooks/bigquery-ai-the-patent-analyst-project.ipynb](https://github.com/veyselserifoglu/bq-ai-patent-analyst/blob/main/notebooks/bigquery-ai-the-patent-analyst-project.ipynb)

--- 

## 5. Architecture Pipeline

```python
from IPython.display import HTML

# Display Architecture pipeline

HTML(f'''
<div style="text-align: center; padding: 15px;">
    <a href="https://github.com/veyselserifoglu/bq-ai-patent-analyst/blob/main/doc/Patent%20Analysis%20Pipeline%20Architecture%20-%20PNG.png?raw=true" 
       target="_blank" 
       style="cursor: pointer; display: inline-block; text-decoration: none;">
        <div style="position: relative; display: inline-block;">
            <img src="https://github.com/veyselserifoglu/bq-ai-patent-analyst/blob/main/doc/Patent%20Analysis%20Pipeline%20Architecture%20-%20PNG.png?raw=true" 
                 width="300" 
                 height="200"
                 style="border: 2px solid #e0e0e0; border-radius: 8px; transition: all 0.3s ease; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
                 onmouseover="this.style.borderColor='#4285F4'; this.style.boxShadow='0 6px 12px rgba(66, 133, 244, 0.3)'"
                 onmouseout="this.style.borderColor='#e0e0e0'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.1)'">
            <div style="position: absolute; top: 8px; right: 8px; background: rgba(255,255,255,0.9); border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 14px;">
                ↗
            </div>
        </div>
    </a>
    <p style="margin-top: 12px; color: #5f6368; font-size: 13px; font-style: italic;">Click to explore the full architecture</p>
</div>
''')
```

```python
# For visualization purposes
%pip install -q pyvis
%pip install -q plotly
%pip install -q ipywidgets
```

```python
# BigQuery
import os
from google.cloud import bigquery
from kaggle_secrets import UserSecretsClient
import pandas as pd
from pyvis.network import Network
import plotly.express as px
from google.cloud import bigquery
from IPython.display import Image, display, HTML, IFrame
import ipywidgets as widgets
from ipywidgets import Layout
import warnings


# pd.set_option('display.max_colwidth', None)

# Suppress the specific UserWarning from the BigQuery client
# warnings.filterwarnings("ignore", message="BigQuery Storage module")
```

# Google Cloud Project Setup

This guide outlines the one-time setup required in Google Cloud and Kaggle to enable the analysis.

---

### 1. Google Cloud Project Configuration

First, configure your Google Cloud project.

1.  **Select or Create a Project**
    * Ensure you have a Google Cloud project.
    * Copy the **Project ID** (e.g., `my-project-12345`), not the project name.

2.  **Enable Required APIs**
    * In your project, enable the following two APIs:
        * **Vertex AI API**
        * **BigQuery Connection API**

3.  **Create a Service Account for the Notebook**
    * This service account allows the Kaggle notebook to act on your behalf.
    * Navigate to **IAM & Admin** > **Service Accounts**.
    * Click **+ CREATE SERVICE ACCOUNT**.
    * Give it a name (e.g., `kaggle-runner`).
    * Grant it these three roles: `Be sure to follow the principle of least privilege.`  
        * `BigQuery Connection User`
        * `BigQuery Data Viewer`
        * `BigQuery User`
    * After creating the account, go to > manage keys > create a new key. A file will be downloaded to your computer.

---

### 2. Kaggle Notebook Configuration

Next, configure this Kaggle notebook to use your project.

1.  **Add Kaggle Secrets**
    * In the notebook editor, go to the **"Add-ons"** menu and select **"Secrets"**.
    * Add two secrets:
        * **`GCP_PROJECT_ID`**: Paste your Google Cloud **Project ID** here.
        * **`GCP_SA_KEY`**: Open the downloaded JSON key file, copy its entire text content, and paste it here.

---

### 3. Final Permission Step (After Running Code)

The first time you run the setup cells in the notebook, a new BigQuery connection will be created. This connection has its own unique service account that needs permission to use AI models.

1.  **Find the Connection Service Account**
    * After running the setup cells, go to **BigQuery** > **External connections** in your Google Cloud project.
    * Click on the connection named `llm-connection`.
    * Copy its **Service Account ID** (it will look like `bqcx-...@...gserviceaccount.com`).

2.  **Grant Permission**
    * Go to the **IAM & Admin** page.
    * Click **+ Grant Access**.
    * Paste the connection's service account ID into the **"New principals"** box.
    * Give it the single role of **`Vertex AI User`**.
    * Click **Save**.

---

With this setup complete, the notebook has secure access to your Google Cloud project and can run all subsequent analysis cells.

```python
user_secrets = UserSecretsClient()
project_id = user_secrets.get_secret("GCP_PROJECT_ID")
gcp_key_json = user_secrets.get_secret("GCP_SA_KEY")
location = 'US'
```

```python
# Write the key to a temporary file in the notebook's environment
key_file_path = 'gcp_key.json'
try:
    with open(key_file_path, 'w') as f:
        f.write(gcp_key_json)
    
    # Remove "> /dev/null 2>&1" to show the output.
    # Authenticate the gcloud tool using the key file
    !gcloud auth activate-service-account --key-file={key_file_path} > /dev/null 2>&1
    
    # Configure the gcloud tool to use your project
    !gcloud config set project {project_id} > /dev/null 2>&1
    
finally:
    # Securely delete the key file immediately after use
    if os.path.exists(key_file_path):
        os.remove(key_file_path)

# Enable the Vertex AI and BigQuery Connection APIs. Run only once Or Enable using the Cloud Interface.
# !gcloud services enable aiplatform.googleapis.com bigqueryconnection.googleapis.com > /dev/null 2>&1
```

```python
# This command creates the connection resource. Remove "> /dev/null 2>&1" to show the output.
!bq mk --connection --location={location} --connection_type=CLOUD_RESOURCE llm-connection > /dev/null 2>&1
```

```python
# This command shows the details of your connection. Remove "> /dev/null 2>&1" to show the output.
!bq show --connection --location={location} llm-connection > /dev/null 2>&1
```

# BigQuery Resource Creation

This section creates the necessary resources for our analysis inside our BigQuery project.

---

## 1. Create a Dataset in the Correct Region.

First, we create a new dataset named `patent_analysis` in our chosen region. This dataset acts as a container for the AI models and the object table of the dataset.

---

## 2. Create a Reference to the AI MultiModel.

Next, we create a "shortcut" to Google's `gemini-2.5-flash` model. This command gives us an easy name, `gemini_vision_analyzer`, to use in our analysis queries.

---

## 3. Create an Object Table for the PDFs.

Next, we create an object table named `patent_documents_object_table`. This is a special "map" that points directly to all the raw PDF files in the public Google Cloud Storage bucket, making them ready for analysis.

---

## 4. Create a Reference to the AI Embedding Model.

Next, we create a "shortcut" to Google's `gemini-embedding-001` model. This command gives us an easy name, `embedding_model`, to use in our embedding tasks.

---

## 5. Create a Reference to do L2 Normalization

Next, We create a custom SQL function to standardize and normalize our vectors.

---

## 6. Create a Reference to perform a weighted average of two vectors.

Finally, we create a custom UDF (user defined function) to intelligently blend our two different types of embeddings (patent context and component function) into a single, more powerful context-aware vector.

---

```python
# Initiate BigQuery client.
client = bigquery.Client(project=project_id, location=location)
client
```

```python
# 1. Create the new dataset "patent_analysis"
patent_analysis = "patent_analysis"

create_dataset_query = f"""
CREATE SCHEMA IF NOT EXISTS `{project_id}.{patent_analysis}`
OPTIONS(location = '{location}');
"""
print(f"Creating dataset 'patent_analysis' in {location}...")
job = client.query(create_dataset_query)
try:
    job.result()
except Exception as e:
    print(f"❌ FAILED to create dataset. Error:\n\n{e}")


# 2. Create the AI model reference inside the new dataset
create_model_query = f"""
CREATE OR REPLACE MODEL `{project_id}.{patent_analysis}.gemini_vision_analyzer`
  REMOTE WITH CONNECTION `{location}.llm-connection`
  OPTIONS (endpoint = 'gemini-2.5-flash');
"""
print("\nCreating the AI model reference...")
job = client.query(create_model_query)
try:
    job.result()
except Exception as e:
    print(f"❌ FAILED to create the AI Model reference. Error:\n\n{e}")


# 3. Create the Object Table
# This query creates the "map" to the PDF files inside the local 'patent_analysis' dataset.
object_table_query = f"""
CREATE OR REPLACE EXTERNAL TABLE `{project_id}.{patent_analysis}.patent_documents_object_table`
WITH CONNECTION `{location}.llm-connection`
OPTIONS (
    object_metadata = 'SIMPLE',
    uris = ['gs://gcs-public-data--labeled-patents/*.pdf'] 
);
"""
print("Creating the object table...")
job = client.query(object_table_query)
try:
    job.result()
except Exception as e:
    print(f"❌ FAILED to create the object table. Error:\n\n{e}")


# 4. Create a remote connection for the embedding model.
sql_query = f"""
CREATE OR REPLACE MODEL `{project_id}.{patent_analysis}.embedding_model`
  REMOTE WITH CONNECTION `{location}.llm-connection`
  OPTIONS (endpoint = 'gemini-embedding-001');
"""

print("Creating the AI Embedding Model reference...")
job = client.query(sql_query)
try:
    job.result()
except Exception as e:
    print(f"❌ FAILED to create the AI Embedding Model reference. Error:\n\n{e}")


# 5. creates a helper function to perform L2 normalization on a vector.
create_classification_model = f"""
CREATE OR REPLACE FUNCTION `{project_id}.{patent_analysis}.L2_NORMALIZE`(vec ARRAY<FLOAT64>)
RETURNS ARRAY<FLOAT64> AS ((
  
  -- Calculate the L2 Norm (magnitude) of the vector.
  WITH vector_norm AS (
    SELECT SQRT(SUM(element * element)) AS norm
    FROM UNNEST(vec) AS element
  )
  
  -- Divide each element by the norm to create a unit vector.
  -- Handle the case where the norm is 0 to avoid division by zero errors.
  SELECT
    ARRAY_AGG(
      IF(norm = 0, 0, element / norm)
    )
  FROM
    UNNEST(vec) AS element, vector_norm
));
"""
print("Creating a Vector Normalization UDF...")
job = client.query(create_classification_model)
try:
    job.result()
except Exception as e:
    print(f"❌ FAILED to create the Vector Normalization reference. Error:\n\n{e}")


# 6. This creates a helper function to perform a weighted average of two vectors.
sql_query = f"""
CREATE OR REPLACE FUNCTION `{project_id}.{patent_analysis}.VECTOR_WEIGHTED_AVG`(
  vec1 ARRAY<FLOAT64>, weight1 FLOAT64,
  vec2 ARRAY<FLOAT64>, weight2 FLOAT64
)
RETURNS ARRAY<FLOAT64>
LANGUAGE js AS r'''
  if (!vec1 || !vec2 || vec1.length !== vec2.length) {{
    return null;
  }}
  let weighted_vec = [];
  for (let i = 0; i < vec1.length; i++) {{
    weighted_vec.push((vec1[i] * weight1) + (vec2[i] * weight2));
  }}
  return weighted_vec;
''';
"""

print("Creating a weighted average vector UDF...")
job = client.query(sql_query)
try:
    job.result()
except Exception as e:
    print(f"❌ FAILED to create the weighted average UDF reference. Error:\n\n{e}")
```

# Utilities
---

This section contains reusable utility functions that handle repetitive tasks, such as formatting our Pandas DataFrames into styled HTML tables for a clean and professional presentation.

---

```python
# 1. DataFrame Styler
def display_styled_df(df: pd.DataFrame, title: str):
    """
    Takes a DataFrame and returns a styled HTML table for better readability.
    """
    if df.empty:
        print("⚠️ DataFrame is empty.")
        return

    styler = df.style \
        .set_caption(f"<h3>{title}</h3>") \
        .set_properties(**{
            'text-align': 'left',
            'white-space': 'normal', # Crucial for wrapping long text
            'font-size': '14px',
            'vertical-align': 'top', # Aligns text to the top of the cell
            'border': '1px solid #444',
            'padding': '8px'
        }) \
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'left'), ('font-size', '16px'), ('background-color', '#333')]},
            {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-size', '18px'), ('text-align', 'center')]}
        ])

    display(HTML(styler.to_html()))
```

# Data Extraction & Knowledge Graph Creation

--- 

## . What did we build?
We created two foundational data assets that power our analysis.

1. the `ai_text_extraction` table: transforms the raw PDFs into structured text, capturing the title and abstract.
2. the `patent_knowledge_graph` table: builds on this, creating a queryable graph of technical components and their connections.

---

## . Why is this important?
- Automates Expert Work, saving hundreds of expert hours. 
- Accelerates Time-to-Insight, analyzing patents in seconds.

---

## . How did we do it?
The process used a sequence of BigQuery's native AI functions:

1. **Multimodal Analysis**:
   - we used `ML.GENERATE_TEXT` to analyze the text and the technical diagrams within each patent's PDF.

2. **Knowledge Graph Extraction**:
   - Next, we fed all the consolidated text into the `AI.GENERATE_TABLE` function, to extract:
     - A nested table of all technical components.
     - Their functions.
     - Their connections for each patent.
  
---

```python
# 1. Multimodal Analysis - only texts - ai_text_extraction table

prompt_text = """From this patent document, perform the following tasks:

1.  **Extract these fields**: title, inventor, abstract, 
    the **Filed**, the **Date of Patent**, the international classification code, and the applicant.
    
2.  **Translate**: If the original title and abstract are in German or French, translate them into English.

3.  **Identify Language**: Determine the original language of the document.

Return ONLY a valid JSON object with EXACTLY these ten keys: 
"title_en", "inventor", "abstract_en", "filed", "date_of_patent", "class_international", "applicant", and "original_language".

**Formatting Rule**: For any key that has multiple values (like "inventor" or "class_international" or "applicant"), 
combine them into a single string, separated by a comma and a space. For example: "Igor Karp, Lev Stesin".

The "original_language" value must be one of these three strings: 'EN', 'FR', or 'DE'.
If any other field is unavailable, use null as the value.
"""

# The main SQL query.
sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.ai_text_extraction` AS (
  WITH raw_json AS (
      SELECT
        uri,
        ml_generate_text_llm_result AS llm_result
      FROM
        ML.GENERATE_TEXT(
          MODEL `{project_id}.{patent_analysis}.gemini_vision_analyzer`,
          TABLE `{project_id}.{patent_analysis}.patent_documents_object_table`,
          STRUCT(
            '''{prompt_text}''' AS prompt,
            2048 AS max_output_tokens,
            0.2 AS temperature,
            TRUE AS flatten_json_output
          )
        )
    ),
    parsed_json AS (
      -- Step 2: Clean and parse the JSON output.
      SELECT
        uri,
        llm_result,
        SAFE.PARSE_JSON(
          REGEXP_REPLACE(llm_result, r'(?s)```json\\n(.*?)\\n```', r'\\1')
        ) AS json_data
      FROM
        raw_json
    )
  SELECT
    uri,
    llm_result,
    
    SAFE.JSON_VALUE(json_data, '$.original_language') AS original_language,
    SAFE.JSON_VALUE(json_data, '$.title_en') AS extracted_title_en,
    SAFE.JSON_VALUE(json_data, '$.inventor') AS extracted_inventor,
    SAFE.JSON_VALUE(json_data, '$.abstract_en') AS extracted_abstract_en,
    SAFE.JSON_VALUE(json_data, '$.filed') AS filed_date,
    SAFE.JSON_VALUE(json_data, '$.date_of_patent') AS official_patent_date,
    SAFE.JSON_VALUE(json_data, '$.class_international') AS class_international,
    SAFE.JSON_VALUE(json_data, '$.applicant') AS applican
    
  FROM
    parsed_json
);
"""

print("Attempting to create the ai text extraction table...")
job = client.query(sql_query)
try:
    job.result()
    print("✅ Success: The `ai_text_extraction` table was created.")

    print("\nFetching a sample of 5 records from the new table:")
    sql_select_sample_query = f"""
    SELECT 
        ate.uri, 
        ate.original_language,
        ate.extracted_title_en,
        ate.extracted_inventor, 
        ate.extracted_abstract_en,
        ate.filed_date,
        ate.class_international
    FROM `{project_id}.{patent_analysis}.ai_text_extraction` AS ate
    WHERE ate.extracted_title_en is not NULL
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, title="Sample of 5 Records from the `ai_text_extraction` Table")

except Exception as e:
    print(f"❌ FAILED: An error occurred. Error:\n\n{e}")
```

```python
# 1. Multimodal Analysis - only extending ai_text_extraction table with the technical diagrams.

diagram_prompt_text = """
Describe this technical diagram from a patent document. 
What is its primary function and what key components are labeled?
"""

sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.ai_text_extraction` AS (

  WITH figures_with_object_ref AS (
      SELECT
        fig.*, obj.ref
      FROM
        `bigquery-public-data.labeled_patents.figures` AS fig
      JOIN
        `{project_id}.{patent_analysis}.patent_documents_object_table` AS obj
      ON
        fig.gcs_path = obj.uri
    ),
    
    generated_descriptions AS (
      SELECT
        gcs_path,
        ml_generate_text_llm_result AS diagram_description
      FROM
        ML.GENERATE_TEXT(
          MODEL `{project_id}.{patent_analysis}.gemini_vision_analyzer`,
          (
            SELECT
              gcs_path,
              [
                JSON_OBJECT('uri', ref.uri, 'bounding_poly', [
                  STRUCT(x_relative_min AS x, y_relative_min AS y),
                  STRUCT(x_relative_max AS x, y_relative_min AS y),
                  STRUCT(x_relative_max AS x, y_relative_max AS y),
                  STRUCT(x_relative_min AS x, y_relative_max AS y)
                ])
              ] AS contents,
              '''{diagram_prompt_text}''' AS prompt
            FROM
              figures_with_object_ref
          ),
          STRUCT(
            4096 AS max_output_tokens,
            0.2 AS temperature,
            TRUE AS flatten_json_output
          )
        )
    ),

    aggregated_descriptions AS (
      SELECT
        gcs_path,
        ARRAY_AGG(diagram_description IGNORE NULLS) AS diagram_descriptions
      FROM
        generated_descriptions
      GROUP BY
        gcs_path
    )

  SELECT
    T.*,
    S.diagram_descriptions
  FROM
    `{project_id}.{patent_analysis}.ai_text_extraction` AS T
  LEFT JOIN
    aggregated_descriptions AS S
  ON
    T.uri = S.gcs_path
);
"""

print("Attempting to extend the ai text extraction table with the diagram description...")
job = client.query(sql_query)
try:
    job.result()
    print("✅ Success: The `ai_text_extraction` table was extended.")

    print("\nFetching a sample of 5 records from the table:")
    sql_select_sample_query = f"""
    SELECT 

        ate.uri, 
        ate.original_language,
        ate.extracted_title_en,
        ate.extracted_inventor,
        ate.filed_date,
        ate.diagram_descriptions
    
    FROM `{project_id}.{patent_analysis}.ai_text_extraction` AS ate
    WHERE ate.extracted_title_en is not NULL AND ARRAY_LENGTH(ate.diagram_descriptions) > 0
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, title="Sample of 5 Records from the `ai_text_extraction` Table, with diagrams descriptions")

except Exception as e:
    print(f"❌ FAILED: An error occurred. Error:\n\n{e}")
```

```python
# 2. Knowledge Graph - patent_knowledge_graph table.

# Define the schema as a Python variable
schema = """
invention_domain STRING, problem_solved STRING, patent_type STRING, 
components ARRAY<STRUCT<component_name STRING, component_function STRING, connected_to ARRAY<STRING>>>
"""

# The prompt text remains the same
prompt_text = """
From the following patent text, perform these tasks:
1. Determine the high-level technical domain (e.g., 'Telecommunications', 'Medical Devices').
2. Provide a one-sentence summary of the core problem the invention solves.
3. Classify the patent as a 'Method', 'System', 'Apparatus', or a combination.
4. Extract all technical components into a nested list. 
For each component, provide its name, its primary function, and a list of other components it is connected to.

Here is the text:
"""

sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.patent_knowledge_graph` AS (
  SELECT
    t.uri,
    t.invention_domain,
    t.problem_solved,
    t.patent_type,
    t.components
  FROM
    AI.GENERATE_TABLE(
      MODEL `{project_id}.{patent_analysis}.gemini_vision_analyzer`,
      (
        SELECT
          uri,
          CONCAT(
            '''{prompt_text}''',
            '\\n\\n',
            IFNULL(extracted_title_en, ''),
            '\\n\\n',
            IFNULL(extracted_abstract_en, ''),
            '\\n\\nDiagrams:\\n',
            IFNULL(ARRAY_TO_STRING(diagram_descriptions, '\\n'), '')
          ) AS prompt
        FROM
          `{project_id}.{patent_analysis}.ai_text_extraction`
        WHERE
          extracted_abstract_en IS NOT NULL
      ),
      STRUCT(
        '''{schema}''' AS output_schema
      )
    ) AS t
);
"""

print("Attempting to create the patent knowledge graph...")
job = client.query(sql_query)
try:
    job.result()
    print("✅ Success: The `patent_knowledge_graph` table was extended.")

    print("\nFetching a sample of 5 records from the table:")
    sql_select_sample_query = f"""
    SELECT 
    
        pkg.uri,
        pkg.invention_domain,
        pkg.problem_solved,
        pkg.patent_type,
        pkg.components
    
    FROM `{project_id}.{patent_analysis}.patent_knowledge_graph` AS pkg
    WHERE ARRAY_LENGTH(pkg.components) > 0 and pkg.invention_domain is not NULL
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, title="Sample of 5 Records from the `patent_knowledge_graph` Table")

except Exception as e:
    print(f"❌ FAILED: An error occurred. Error:\n\n{e}")
```

# Data Analysis & Quality Validation
---

**What are we doing?**
Before we trust our AI-generated knowledge graph for analysis, we must first validate its quality. This section performs a series of data quality checks to ensure the data is complete, consistent, and reliable.

**Why is this important?**
This is a critical step in any production-grade data pipeline. It builds trust in our data and ensures that the insights we derive in the following sections are based on a solid foundation.

---

## 1. Completeness Check: Null Rates
We'll start by checking for missing data. This single query calculates the total count and the percentage of null values for our key AI-generated columns. A low null rate indicates a successful extraction.

---

## 2. Uniqueness Check: Duplicate Patents
Next, we ensure that each patent document (`uri`) is represented only once in our final table. This query should return zero rows.

---

## 3. Consistency Check: Component Schema
We need to verify that the AI consistently followed our instructions. This query checks if every component in our nested knowledge graph has both a `component_name` and a `component_function`, as required by our prompt.

---

## 4. Outlier Detection: Component Count
Finally, we'll perform a statistical check to find patents that might be outliers. This query identifies patents with an unusually high number of components (e.g., more than 3 standard deviations above the average), which could indicate an extraction error or a uniquely complex invention worthy of a closer look.

---

## 5. Compare Companies' Patents

1. Get Better Data

First, we'll write an SQL query to get the key numbers for each company. Instead of just counting the parts in their patents, we'll count the connections between the parts, which is a much better way to measure how complex their inventions really are.

2. Make the Chart

Then, we'll use this data to create our bubble chart. This chart will compare the companies, showing us who is making the most complex and diverse technology.

---

```python
# 1. This query calculates the null percentage for key columns in the knowledge graph.
sql_completeness_check = f"""
SELECT
  COUNT(*) AS total_rows,
  ROUND(100 * COUNTIF(invention_domain IS NULL) / COUNT(*), 2) AS pct_null_domain,
  ROUND(100 * COUNTIF(problem_solved IS NULL) / COUNT(*), 2) AS pct_null_problem,
  ROUND(100 * COUNTIF(ARRAY_LENGTH(components) IS NULL OR ARRAY_LENGTH(components) = 0) / COUNT(*), 2) AS pct_empty_components
FROM
  `{project_id}.{patent_analysis}.patent_knowledge_graph`;
"""

print("--- Running Completeness Check ---")
try:
    df_completeness = client.query(sql_completeness_check).to_dataframe()
    display_styled_df(df_completeness, "Data Completeness and Null Rates (%)")
except Exception as e:
    print(f"❌ FAILED: The query failed. Error:\n\n{e}")
```

#### Inference
---

The results of our completeness check are highly positive. With null rates of less than 1.2% across all key AI-generated fields, we can confirm that our data extraction process was successful and the resulting knowledge graph is a high-quality, reliable foundation for the analysis that follows.

---

```python
# 2. This query checks for duplicate URIs in the knowledge graph table.
sql_duplicate_check = f"""
SELECT
  uri,
  COUNT(*) AS num_occurrences
FROM
  `{project_id}.{patent_analysis}.patent_knowledge_graph`
GROUP BY
  uri
HAVING
  num_occurrences > 1;
"""

print("--- Running Uniqueness Check ---")
try:
    df_duplicates = client.query(sql_duplicate_check).to_dataframe()
    
    if df_duplicates.empty:
        print("✅ Success: No duplicate patents found.")
    else:
        print("⚠️ Warning: Duplicate patents found! These URIs appear more than once:")
        display_styled_df(df_duplicates, "Duplicate Patent URIs")

except Exception as e:
    print(f"❌ FAILED: The query failed. Error:\n\n{e}")
```

```python
# 3. This query validates the nested component schema for completeness.
sql_schema_check = f"""
SELECT
  COUNT(*) AS total_components,
  COUNTIF(c.component_name IS NULL) AS components_missing_name,
  COUNTIF(c.component_function IS NULL) AS components_missing_function
FROM
  `{project_id}.{patent_analysis}.patent_knowledge_graph` AS t,
  UNNEST(t.components) AS c;
"""

print("--- Running Schema Consistency Check ---")
try:
    df_schema = client.query(sql_schema_check).to_dataframe()
    display_styled_df(df_schema, "Component Schema Consistency")
except Exception as e:
    print(f"❌ FAILED: The query failed. Error:\n\n{e}")
```

```python
# 4. This query finds patents with an anomalous number of components.
sql_outlier_check = f"""
WITH component_stats AS (
  SELECT
    uri,
    ARRAY_LENGTH(components) AS num_components,
    AVG(ARRAY_LENGTH(components)) OVER() AS avg_components,
    STDDEV(ARRAY_LENGTH(components)) OVER() AS stddev_components
  FROM
    `{project_id}.{patent_analysis}.patent_knowledge_graph`
)
SELECT
  uri,
  num_components
FROM
  component_stats
WHERE
  -- A standard statistical definition of an outlier
  num_components > avg_components + (3 * stddev_components);
"""

print("--- Running Outlier Detection ---")
try:
    df_outliers = client.query(sql_outlier_check).to_dataframe()
    
    if df_outliers.empty:
        print("✅ Success: No significant outliers found in component counts.")
    else:
        print("⚠️ Warning: Potential outliers found. These patents have an unusually high number of components:")
        display_styled_df(df_outliers, "Patent Component Count Outliers")

except Exception as e:
    print(f"❌ FAILED: The query failed. Error:\n\n{e}")
```

```python
# Distribution of component counts - Histogram

# --- Step 1: Fetch the component count for ALL patents ---
sql_all_counts = f"""
SELECT
  ARRAY_LENGTH(components) AS num_components
FROM
  `{project_id}.{patent_analysis}.patent_knowledge_graph`
WHERE
  ARRAY_LENGTH(components) > 0;
"""

print("--- Generating Distribution Plot with Outlier List ---")

try:
    df_all_counts = client.query(sql_all_counts).to_dataframe()
    
    # --- Step 2: Create the Histogram Figure ---
    fig = px.histogram(
        df_all_counts,
        x="num_components",
        title="<b>Distribution of Component Counts</b>",
        labels={"num_components": "Number of Components per Patent"}
    )

    # Add vertical lines for each outlier
    for index, row in df_outliers.iterrows():
        fig.add_vline(
            x=row['num_components'],
            line_width=2,
            line_dash="dash",
            line_color="red"
        )

    fig.update_layout(
        xaxis_title="<b>Number of Components</b>",
        yaxis_title="<b>Number of Patents</b>",
        font=dict(family="Arial, sans-serif", size=12),
        width=600 # Set a fixed width for the chart
    )
    
    # --- Step 3: Convert the chart and the list to HTML strings ---
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    outlier_list_html = "<h4>Potential Outliers:</h4><ul style='font-size: 12px; list-style-type: none; padding-left: 0;'>"
    for index, row in df_outliers.iterrows():
        short_name = row['uri'].split('/')[-1]
        outlier_list_html += f"<li style='margin-bottom: 5px;'>- {short_name} ({row['num_components']} components)</li>"
    outlier_list_html += "</ul>"

    # --- Step 4: Combine everything into a single HTML table for side-by-side display ---
    final_html = f"""
    <div style="display: flex; flex-direction: row; align-items: flex-start;">
        <div style="flex: 3;">{chart_html}</div>
        <div style="flex: 1; padding-left: 20px;">{outlier_list_html}</div>
    </div>
    """
    
    # Display the final combined HTML
    display(HTML(final_html))

except Exception as e:
    print(f"❌ FAILED: Could not generate the plot. Error:\n\n{e}")
```

#### Inference
---

This chart reveals that most patents in our dataset are of normal complexity, typically having between 2 and 15 components. The long tail to the right, marked by the red lines, shows that patents with a high number of components are rare outliers.

Crucially, our analysis confirms that these outliers are patents that contain technical diagrams. This proves that the true architectural complexity of an invention is often hidden within its visual data. Our multimodal pipeline is essential because it successfully unlocks this deeper layer of information, providing a much richer and more accurate understanding than a text-only analysis ever could.

---

```python
# 5. Compare Companies' Patents.

# This query creates a summary table for each patent applicant.
sql_connection_density_query = f"""
WITH
  patent_connection_stats AS (
    SELECT
      T1.uri,
      T1.applican,
      T2.invention_domain,
      (
        SELECT SUM(ARRAY_LENGTH(c.connected_to))
        FROM UNNEST(T2.components) AS c
        WHERE c.connected_to IS NOT NULL
      ) AS total_connections
    FROM
      `{project_id}.{patent_analysis}.ai_text_extraction` AS T1
    JOIN
      `{project_id}.{patent_analysis}.patent_knowledge_graph` AS T2
    ON
      T1.uri = T2.uri
    WHERE
      T1.applican IS NOT NULL AND T2.invention_domain IS NOT NULL
  )

SELECT
  applican,
  COUNT(DISTINCT invention_domain) AS innovation_breadth,
  ROUND(AVG(total_connections), 2) AS average_connection_density,
  COUNT(uri) AS total_patents
FROM
  patent_connection_stats
WHERE
  total_connections > 0 -- Exclude patents with no connections to avoid skewing the average.
GROUP BY
  applican
HAVING
  COUNT(uri) > 1 -- Filter for applicants with more than one patent for a cleaner chart.
ORDER BY
  total_patents DESC;
"""

print("--- Calculating Enhanced Portfolio Metrics ---")
try:
    df_summary_enhanced = client.query(sql_connection_density_query).to_dataframe()
    print("✅ Success: Enhanced metrics calculated.")
    display(df_summary_enhanced.head())
except Exception as e:
    print(f"❌ FAILED: The query failed. Error:\n\n{e}")

# Create the Interactive Bubble Chart using the new "connection density" metric.
fig = px.scatter(
    df_summary_enhanced,
    x="innovation_breadth",
    y="average_connection_density",
    size="total_patents",
    color="applican",
    hover_name="applican",
    log_x=True,
    size_max=60,
    title="<b>Strategic Patent Portfolio Analysis: Breadth vs. Connection Density</b>",
    labels={
        "innovation_breadth": "Innovation Breadth (Number of Domains)",
        "average_connection_density": "Average Connection Density (Connections per Patent)"
    }
)

# Customize the layout for a professional look
fig.update_layout(
    showlegend=False,
    xaxis_title="<b>Innovation Breadth ➡️</b> (More Diverse)",
    yaxis_title="<b>Architectural Complexity ⬆️</b> (More Connections)"
)

display(HTML(fig.to_html()))
```

#### Inference
---
This chart is a strategic map of the patent applicants in our dataset. Each bubble represents a single company.

- Position Right ➡️ (Innovation Breadth): Shows companies that are more diverse, patenting across many different technology domains.

- Position Top ⬆️ (Architectural Complexity): Shows companies that build more complex inventions with a higher number of connections between components.

- Bubble Size ⚪ (Portfolio Size): Shows who is most prolific, with larger bubbles representing more total patents.

This allows us to instantly identify different innovation strategies, such as a large, diverse innovator in the top-right versus a specialized, deep-tech player in the top-left.

---

# Patent Search Engine Preps

---

## - What did we build?
A powerful semantic search engine that finds specific technical components based on a natural language description of their function.

---

## - Why is this important?
- Standard search finds keywords. This search finds meaning.
- By combining two different vector embeddings, the engine understands patent's components and the technical context in which it operates.
- This allows an engineer to find a "valve for precise fluid delivery" and get results from relevant medical patents, not car engine patents.

---

## - How did we do it?
The process involves three key stages, all performed within BigQuery:

1. **Dual Embeddings**:
   - We first generate two separate vector embeddings:
     - One for the high-level patent context (title, abstract, domain, diagrams)
     - Another for the specific component's function

2. **Vector Combination**:
    - We create a custom User-Defined Function (UDF) to perform a weighted average of our two embeddings.
    - We assign 70% weight to the specific component function and 30% to the broader patent context.
    - This creates a final vector that prioritizes the component's specific role, reducing noise from the broader topic and improving search relevance.

3. **Semantic Search**:
   - Finally, we use the `VECTOR_SEARCH` function to compare a user's query against these combined vectors.
   - Returns the most similar components from the entire dataset.

---

```python
# This query creates a flat table of all components from all patents.
sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.patent_components_flat` AS (
  SELECT
    t.uri,
    t.invention_domain,
    c.component_name,
    c.component_function,
    c.connected_to
  FROM
    `{project_id}.{patent_analysis}.patent_knowledge_graph` AS t,
    UNNEST(t.components) AS c
  WHERE
    c.component_function IS NOT NULL
    AND c.component_name IS NOT NULL
);
"""

print("Attempting to create the flattened components table...")
job = client.query(sql_query)
try:
    job.result()
    print("✅ Success: The `patent_components_flat` table was created.")

    print("\nFetching a sample of 5 records from the new table:")
    sql_select_sample_query = f"""
    SELECT * FROM `{project_id}.{patent_analysis}.patent_components_flat` 
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, "Patent Components Flattened")

except Exception as e:
    print(f"❌ FAILED: An error occurred. Error:\n\n{e}")
```

```python
# This query creates a single context vector for each patent, reading from ai_text_extraction table.
sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.patent_context_embeddings` AS (
  SELECT
    t.uri,
    t.ml_generate_embedding_result AS patent_context_vector
  FROM
    ML.GENERATE_EMBEDDING(
      MODEL `{project_id}.{patent_analysis}.embedding_model`,
      (
        SELECT
          uri,
          CONCAT(
            'Represent this technical patent for semantic search: \\n\\n', 
            'Patent Title: ', IFNULL(extracted_title_en, ''), '\\n\\n',
            'Applicant: ', IFNULL(applican, ''), '\\n\\n',
            'International Classification: ', IFNULL(class_international, ''), '\\n\\n',
            'Abstract: ', IFNULL(extracted_abstract_en, ''), '\\n\\n',
            'Diagram Descriptions: ', IFNULL(ARRAY_TO_STRING(diagram_descriptions, '\\n'), '')
          ) AS content
        FROM
          `{project_id}.{patent_analysis}.ai_text_extraction`
        WHERE
          extracted_title_en IS NOT NULL
      )
    ) AS t
);
"""

print("Attempting to create the patent context embeddings table...")
job = client.query(sql_query)
try:
    job.result() 
    print("✅ Success: The `patent_context_embeddings` table was created.")

    print("\nFetching a sample of 5 records from the new table:")
    sql_select_sample_query = f"""
    SELECT 
        uri, 
        ARRAY_LENGTH(patent_context_vector) as vector_dimensions 
    FROM `{project_id}.{patent_analysis}.patent_context_embeddings` 
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, "Patent Context Embedding Sample")

except Exception as e:
    print(f"❌ FAILED: An error occurred. Error:\n\n{e}")
```

```python
# This query creates a single specific function vector for each individual component.
sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.component_function_embeddings` AS (
  SELECT
    t.uri,
    t.component_name,
    t.ml_generate_embedding_result AS component_function_vector
  FROM
    ML.GENERATE_EMBEDDING(
      MODEL `{project_id}.{patent_analysis}.embedding_model`,
      (
        SELECT
          uri,
          component_name,
          CONCAT(
            'Represent this technical patent for semantic search: \\n\\n',
            'A component named "', component_name, '" whose function is to ', component_function
          ) AS content
        FROM
          `{project_id}.{patent_analysis}.patent_components_flat`
      )
    ) AS t
);
"""

print("Attempting to create the component function embeddings table...")
job = client.query(sql_query)
try:
    job.result()
    print("✅ Success: The `component_function_embeddings` table was created.")

    print("\nFetching a sample of 5 records from the new table:")
    sql_select_sample_query = f"""
    SELECT 
        uri, 
        component_name,
        ARRAY_LENGTH(component_function_vector) as vector_dimensions 
    FROM `{project_id}.{patent_analysis}.component_function_embeddings` 
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, "Component Function Embedding Sample")

except Exception as e:
    print(f"❌ FAILED: An error occurred. Error:\n\n{e}")
```

```python
# Normalization

def normalize_and_save_vectors(
    table_id: str,
    vector_column: str,
    client: bigquery.Client
):
    """
   Normalizes a vector column in a BigQuery table in-place by replacing
    the table with its normalized version.

    Args:
        table_id: The full ID of the table to update (e.g., "project.dataset.table").
        vector_column: The name of the column containing the vectors to normalize.
        client: An authenticated BigQuery client object.
    """


    # This SQL query selects all original columns and replaces the vector
    # column with its normalized version.
    sql_query = f"""
    CREATE OR REPLACE TABLE `{table_id}` AS (
      SELECT
        * EXCEPT({vector_column}),
        `{client.project}.{patent_analysis}.L2_NORMALIZE`({vector_column}) AS {vector_column}
      FROM
        `{table_id}`
    );
    """

    try:
        # Execute the query.
        job = client.query(sql_query)
        job.result()
    except Exception as e:
        print(f"❌ FAILED: An error occurred during normalization. Error:\n\n{e}")


# 1. Normalize the patent context embeddings.
print("--- Normalizing Patent Context Vectors ---")
normalize_and_save_vectors(
   table_id=f"{project_id}.{patent_analysis}.patent_context_embeddings",
   vector_column="patent_context_vector",
   client=client
)

# 2. Normalize the component function embeddings.
print("\n--- Normalizing Component Function Vectors ---")
normalize_and_save_vectors(
   table_id=f"{project_id}.{patent_analysis}.component_function_embeddings",
   vector_column="component_function_vector",
   client=client
)

print("\n--- Fetching a Diverse Sample of 5 Unique Patents ---")

# This query uses QUALIFY to get one component from 5 different patents.
sql_select_sample = f"""
SELECT
    uri,
    component_name,
    ARRAY_LENGTH(component_function_vector) as vector_dimensions
FROM
    `{project_id}.{patent_analysis}.component_function_embeddings`
QUALIFY
    ROW_NUMBER() OVER(PARTITION BY uri ORDER BY RAND()) = 1
LIMIT 5;
"""

try:
    df_sample = client.query(sql_select_sample).to_dataframe()
    display_styled_df(df_sample, "Normalized Embedding Sample")
except Exception as e:
    print(f"❌ FAILED to fetch a diverse sample. Error:\n\n{e}")
```

```python
# This query rebuilds the search index using the UDF - weighted average function.
sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.component_search_index` AS (
  SELECT
    flat.uri,
    flat.component_name,
    flat.component_function,
    -- Call our new UDF with the desired weights.
    `{project_id}.{patent_analysis}.VECTOR_WEIGHTED_AVG`(
      func.component_function_vector, 0.7, -- 70% weight to the function
      ctx.patent_context_vector, 0.3      -- 30% weight to the context
    ) AS combined_vector
  FROM
    `{project_id}.{patent_analysis}.patent_components_flat` AS flat
  JOIN
    `{project_id}.{patent_analysis}.patent_context_embeddings` AS ctx
  ON
    flat.uri = ctx.uri
  JOIN
    `{project_id}.{patent_analysis}.component_function_embeddings` AS func
  ON
    flat.uri = func.uri AND flat.component_name = func.component_name
);
"""

print("Attempting to create the final component search index table...")
job = client.query(sql_query)
try:
    job.result()
    print("✅ Success: The `component_search_index` table was created.")

    print("\nFetching a diverse sample of 5 records from the new table:")
    sql_select_sample_query = f"""
    SELECT
        uri,
        component_name,
        ARRAY_LENGTH(combined_vector) as vector_dimensions
    FROM
        `{project_id}.{patent_analysis}.component_search_index`
    QUALIFY
        ROW_NUMBER() OVER(PARTITION BY uri ORDER BY RAND()) = 1
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, "Final Component Searching Sample")

except Exception as e:
    print(f"❌ FAILED: An error occurred. Error:\n\n{e}")
```

# Demo: Search Engine Gate

---

### **Next Steps: Explore the Live Application**

The data processing pipeline is now complete. The `patent_knowledge_graph` and `component_search_index` tables have been created in BigQuery and are ready for analysis.

To experience the full power of this project, please visit our live, interactive Streamlit application. It provides a polished user interface for the semantic search engine and the strategic analysis dashboards.

-   **Live Demo:** [**https://patent-search-analytics.streamlit.app/**](https://patent-search-analytics.streamlit.app/)

---

# **Business Impact & ROI: A Quantitative Analysis**

---

The true impact of this platform is best understood by contrasting the automated AI pipeline with the manual reality it replaces. The semantic search and analysis that took minutes to execute would require an expert hundreds of hours of manual reading to achieve the same results.

To provide a clear and defensible measure of the value created, we based our analysis on the following conservative assumptions:

-   **Manual Analysis Time:** **1.5 hours** per patent for a skilled engineer to read, comprehend, and document the key components and their interconnections.
-   **Blended Expert Cost Rate:** **$150 per hour**, a standard corporate rate accounting for salary, benefits, and overhead.

Based on these inputs, the ROI for analyzing the 403-patent corpus is transformative.

### **Financial & Efficiency Metrics**

| Metric | Manual Process | BigQuery AI Pipeline | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Time** | ~605 Hours | ~15 Minutes | **>99% Reduction** |
| **Total Cost** | ~$90,750 | ~$20 | **>4,500x ROI** |
| **Analysis Throughput**| ~0.6 patents/hr | ~1,600 patents/hr | **2,400x Increase** |

---

### **Strategic Value Unlocked**

Beyond the significant cost and time savings, the platform unlocks critical strategic advantages by transforming a static, high-cost data project into a dynamic, low-cost intelligence asset. This enables an organization to:

* **♟️ Accelerate Innovation:** Move from months of research to minutes of interactive discovery, dramatically shortening product development cycles.
* **🛡️ Mitigate Risk:** Avoid patent infringement and redundant R&D by leveraging a context-aware semantic search.
* **🧠 Democratize Knowledge:** Empower the entire R&D team with direct access to deep technical insights, creating a permanent and queryable "corporate brain."

# **Conclusion: From Painful to Interactive**

#### This is a transformation tool. It turns a static, six-figure, multi-month data entry project into a **$20, 15-minute process** and empowers the entire R&D team with an interactive search capability that was previously excruciating and painful to make.
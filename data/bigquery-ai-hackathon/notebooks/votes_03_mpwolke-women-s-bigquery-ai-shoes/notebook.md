# Women's BigQuery AI? Shoes!!

- **Author:** Marília Prata
- **Votes:** 48
- **Ref:** mpwolke/women-s-bigquery-ai-shoes
- **URL:** https://www.kaggle.com/code/mpwolke/women-s-bigquery-ai-shoes
- **Last run:** 2025-08-13 03:42:31.110000

---

Published on August 12, 2025. By Prata, Marília (mpwolke)

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.offline as py

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

## Competition Citation:

@misc{bigquery-ai-hackathon,

    author = {Wei Hsia and Jing Jing Long and Rachael Deacon-Smith and Gabe Weiss and Gautam Gupta and Omid Fatemieh and Yves-Laurent Kom Samo and Thibaud Hottelier and Ivan Santa Maria Filho and Jiaxun Wu},
    
    title = {BigQuery AI - Building the Future of Data},
    
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/bigquery-ai-hackathon}},
    note = {Kaggle}
}

### Welcome to the BigQuery AI Hackathon

This is a Hackathon with **no provided dataset**.

https://www.kaggle.com/competitions/bigquery-ai-hackathon/data?select=README

## Import Bigquery

```python
from google.cloud import bigquery

client = bigquery.Client()


# List the tables in geo_openstreetmap dataset which resides in bigquery-public-data project:
dataset = client.get_dataset('bigquery-public-data.geo_openstreetmap')
tables = list(client.list_tables(dataset))
print([table.table_id for table in tables])
```

```python
#By Anna Epishova https://www.kaggle.com/annaepishova/starter-geo-openstreetmap-bigquery-dataset

sql = '''
SELECT nodes.*
FROM `bigquery-public-data.geo_openstreetmap.planet_nodes` AS nodes
JOIN UNNEST(all_tags) AS tags
WHERE tags.key = 'amenity'
  AND tags.value IN ('hospital',
    'clinic',
    'doctors')
LIMIT 10
'''
# Set up the query
query_job = client.query(sql)

# Make an API request  to run the query and return a pandas DataFrame
df = query_job.to_dataframe()
df.head(5)
```

```python
sql = '''
SELECT nodes.*
FROM `bigquery-public-data.geo_openstreetmap.planet_nodes` AS nodes
JOIN UNNEST(all_tags) AS tags
WHERE tags.key = 'amenity'
  AND tags.value IN ('stores',
    'clothes',
    'shoes')
LIMIT 10
'''
# Set up the query
query_job = client.query(sql)

# Make an API request  to run the query and return a pandas DataFrame
df = query_job.to_dataframe()
df.head()
```

## Shoes on fourth column and other columns

Longitude 8.2768427 Latitude 47.3489955  That's in **Switzerland**

47°20'56.4"N 8°16'36.6"E:  Wohlen, Switzerland

```python
#Fifth row, fourth column 

df.iloc[1,7]
```

```python
df['all_tags'].value_counts()
```

## 小脚印童鞋: Small footprint children's shoes

Longitude 117.753525300 Latitude 39.886507300

Google Maps: 39°53'11.4"N 117°45'12.7"E  Yutian County, Hebei, Tangshan, Hebei, **China**.

```python
df.iloc[5,7]
```

## 新脚印鞋店: New Footprints Shoe Store

Longitude 117.7536421 Latidude 39.8863103

According to Google Maps: 39°53'10.7"N 117°45'13.1"E  is in Yutian County, Hebei, Tangshan, Hebei, **China**.

```python
df.iloc[6,7]
```

```python
df.head(7)
```

## I wanted Milan since Manolo's and Louboutin's are made in Italy

"Manolo Blahnik shoes are made in Italy. The brand has a strong connection to the region, **particularly near Milan**, due to the Italian craftsmanship and sense of beauty that the designer admires. While the shoes are crafted in Italy, Manolo Blahnik himself is Spanish, and the company is headquartered in London." 

"Christian Louboutin shoes are primarily crafted in Europe, with a strong emphasis on Italy and Spain. Specifically, the Louboutin factory outside **Milan** is a key location for the manufacturing of their footwear."

### Unfortunately, my locations are in China 😂 and Switzerland.

```python
speeds_query = """
               WITH milan AS (
               SELECT ST_MAKEPOLYGON(ST_MAKELINE(
               [ST_GEOGPOINT(8.2768427 47.3489955),ST_GEOGPOINT(117.7535253 39.8865073),
               ST_GEOGPOINT(117.7536421 39.8863103)
               ]
               )) AS boundingbox
               )
               """
```

## Define to run query

```python
def run_query(shoes_query):
    return pd.read_sql_query(shoes_query, df) #Original was db
```

## Damm Cursor: AttributeError 'DataFrame' object has no attribute 'cursor'

-> 2672         cur = self.con.cursor()

-> 2738         cursor = self.execute(sql, params)

```python
shoes_query = '''
SELECT hist.*
FROM `bigquery-public-data.geo_openstreetmap.history_nodes` AS hist
INNER JOIN UNNEST(all_tags) AS tags
INNER JOIN milan on ST_INTERSECTS(milan.boundingbox, hist.geometry)
WHERE tags.key = 'nice'
  AND tags.value IN ('stores',
    'clothes',
    'shoes')
  AND hist.id NOT IN (
    SELECT nodes.id
    FROM `bigquery-public-data.geo_openstreetmap.planet_nodes` AS nodes
    INNER JOIN UNNEST(all_tags) AS tags
    INNER JOIN milan on ST_INTERSECTS(milan.boundingbox, nodes.geometry)
    WHERE tags.key = 'nice'
      AND tags.value IN ('stores',
        'closes',
        'shoes')
)
'''
run_query(shoes_query)
```

## The importance of shoes to women. "WomMansplaining" the Notebook title

"Shoes are very important to women, fulfilling both practical and stylistic needs. Shoes significantly impact a woman's outfit and overall appearance, influencing style and mood."

No matter they are uncomfortable, **it's all about Style**.

## Draft Session (1h:57m) 

### Forget the Manolo Blahnik or the Louboutin, it's just Blah code.

After 2 locations in China and one in Switzerland, just a Cursor AttributeError and No Milan.

![](https://i.pinimg.com/736x/8b/be/8c/8bbe8c008739e52b8eb01af098e9205b.jpg)Pinterest

#Acknowledgements:

Anna Epishova https://www.kaggle.com/annaepishova/starter-geo-openstreetmap-bigquery-dataset

DanB https://www.kaggle.com/code/dansbecker/select-from-where

mpwolke https://www.kaggle.com/code/mpwolke/openstreemap-bquery
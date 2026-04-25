# Multimodal Data Processing with BigQuery AI

- **Author:** Ernst Ashurov
- **Votes:** 33
- **Ref:** vet231199/multimodal-data-processing-with-bigquery-ai
- **URL:** https://www.kaggle.com/code/vet231199/multimodal-data-processing-with-bigquery-ai
- **Last run:** 2025-08-12 18:01:52.430000

---

<div style="text-align:center">
    <span style="background: linear-gradient(to right, darkorange, darkcyan);
           -webkit-background-clip: text;
           -webkit-text-fill-color: transparent;
           font-size: 28px;
           font-weight: bold;
           display: inline-block;">
Multimodal Data Processing with BigQuery AI    </span>
</div>

### <div style="color:white;background-color:darkcyan;padding:1.2%;border-radius:12px 12px;font-size:1.1em;text-align:center"> Creating Embeddings for Text and Images.</div> 

![](https://www.kaggle.com/competitions/110281/images/header)

```python
import openai
from google.cloud import bigquery
import pandas as pd


openai.api_key = 'ваш_openai_api_key'

# Инициализация клиента BigQuery
client = bigquery.Client()

# Предположим, у вас есть список текстов
texts = ['пример текста 1', 'пример текста 2', 'пример текста 3']

embeddings = []
for text in texts:
    response = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002'
    )
    embeddings.append(response['data'][0]['embedding'])

# Создаем DataFrame для загрузки в BigQuery
df = pd.DataFrame({
    'text': texts,
    'embedding': embeddings
})

# Загружаем в BigQuery
table_id = 'your_project.your_dataset.embeddings'

job = client.load_table_from_dataframe(df, table_id)
job.result()  # дождаться завершения

print("Данные загружены успешно.")
```

# <span style="color:darkcyan;">၊၊||၊ Creating Vector Indexes for Fast Searching | </span>
<p style="border-bottom: 35px solid darkorange"></p>
<p style="border-bottom: 5px solid darkcyan"></p>

```python
-- Создаем индекс для текстовых эмбеддингов
CREATE OR REPLACE MODEL my_dataset.text_vector_index
OPTIONS(
  model_type='vector',
  input_label_cols=['text_embedding']
) AS
SELECT
  id,
  text_embedding
FROM
  my_dataset.text_embeddings;

-- Аналогично для изображений (если поддерживается)
-- В зависимости от версии BigQuery, создание индекс может иметь свои особенности
```

# <span style="color:darkcyan;">၊၊||၊ Search for similar documents by request | </span>
<p style="border-bottom: 35px solid darkorange"></p>
<p style="border-bottom: 5px solid darkcyan"></p>

```python
-- Предположим, у вас есть текстовый запрос
DECLARE query_text STRING DEFAULT 'пример запроса';

-- Генерируем эмбеддинг для запроса
WITH query_embedding AS (
  SELECT
    ML.GENERATE_EMBEDDING(
      query_text,
      MODEL 'embedder/text'
    ) AS embedding
)

-- Находим наиболее похожие документы
SELECT
  d.id,
  d.text,
  ML.SIMILARITY(d.text_embedding, q.embedding) AS similarity_score
FROM
  my_dataset.text_embeddings AS d,
  query_embedding AS q
ORDER BY
  similarity_score DESC
LIMIT 10;
```

# <span style="color:darkcyan;">၊၊||၊ Generate a short description for the selected document | </span>
<p style="border-bottom: 35px solid darkorange"></p>
<p style="border-bottom: 5px solid darkcyan"></p>

```python
-- Предположим, что есть выбранный id
DECLARE selected_id STRING DEFAULT 'id_пример';

-- Получаем текст
WITH selected_text AS (
  SELECT text FROM my_dataset.multimodal_data WHERE id = selected_id
)

-- Генерируем резюме или описание
SELECT
  AI.GENERATE_TEXT(
    prompt => CONCAT('Создайте краткое описание для следующего текста:\n', (SELECT text FROM selected_text)),
    temperature => 0.7,
    max_tokens => 100
  ) AS summary;
```

Summary:
* You can combine these queries to search for similar multimodal data and automatically generate their description.
* A full prototype would require automation of these steps using Python or Workflow scripts, as well as data preparation.
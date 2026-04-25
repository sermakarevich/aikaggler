# 🏆 WSDM 🙂 Sentiment Analysis 📊 EDA

- **Author:** Taimour Nazar
- **Votes:** 161
- **Ref:** taimour/wsdm-sentiment-analysis-eda
- **URL:** https://www.kaggle.com/code/taimour/wsdm-sentiment-analysis-eda
- **Last run:** 2024-12-06 06:56:00.367000

---

# <span style="background-color:#d4fad6;color:black;padding:10px;border-radius:40px;">🎒 Import Libraries</span>

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lightgbm import early_stopping,log_evaluation,LGBMClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from textblob import TextBlob #for sentiment analysis
```

# <span style="background-color:#d4fad6;color:black;padding:10px;border-radius:40px;">⬆️ Load Data</span>

```python
path = "/kaggle/input/wsdm-cup-multilingual-chatbot-arena/"
train = pd.read_parquet(path+"train.parquet")
test = pd.read_parquet(path+"test.parquet")
```

# <span style="background-color:#d4fad6;color:black;padding:10px;border-radius:40px;">🔎 View Data</span>

```python
train.head(4)
```

```python
test.head(2)
```

# <span style="background-color:#d4fad6;color:black;padding:10px;border-radius:40px;">🙂 Sentiment Analysis</span>

```python
def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

# Apply sentiment analysis to each column
train['prompt_sentiment'] = train['prompt'].apply(sentiment_analysis)
train['response_a_sentiment'] = train['response_a'].apply(sentiment_analysis)
train['response_b_sentiment'] = train['response_b'].apply(sentiment_analysis)
test['prompt_sentiment'] = test['prompt'].apply(sentiment_analysis)
test['response_a_sentiment'] = test['response_a'].apply(sentiment_analysis)
test['response_b_sentiment'] = test['response_b'].apply(sentiment_analysis)
```

<div style="background-color:white;color:black;padding:20px;border:5px solid blue;border-radius:20px;">
We have three important text columns i.e prompt, response a and response b for this competition. It will be a good idea to do sentiment analysis on these columns. TextBlob is used above to perform sentiment analysis on both train and test data.
</div>

# <span style="background-color:#d4fad6;color:black;padding:10px;border-radius:40px;">⚙️ Feature Engineering</span>

```python
# Here I compute some features
def compute_feats(df):
    for col in ["response_a","response_b","prompt"]:
        # response lenght is a key factor when choosing between two responses
        df[f"{col}_len"]=df[f"{col}"].str.len()

        # Some characters counting features 
        df[f"{col}_spaces"]=df[f"{col}"].str.count("\s")
        df[f"{col}_punct"]=df[f"{col}"].str.count(",|\.|!")
        df[f"{col}_question_mark"]=df[f"{col}"].str.count("\?")
        df[f"{col}_quot"]=df[f"{col}"].str.count("'|\"")
        df[f"{col}_formatting_chars"]=df[f"{col}"].str.count("\*|\_")
        df[f"{col}_math_chars"]=df[f"{col}"].str.count("\-|\+|\=")
        df[f"{col}_curly_open"]=df[f"{col}"].str.count("\{")
        df[f"{col}_curly_close"]=df[f"{col}"].str.count("}")
        df[f"{col}_round_open"]=df[f"{col}"].str.count("\(")
        df[f"{col}_round_close"]=df[f"{col}"].str.count("\)")
        df[f"{col}_special_chars"]=df[f"{col}"].str.count("\W")
        df[f"{col}_digits"]=df[f"{col}"].str.count("\d")>0
        df[f"{col}_lower"]=df[f"{col}"].str.count("[a-z]").astype("float32")/df[f"{col}_len"]
        df[f"{col}_upper"]=df[f"{col}"].str.count("[A-Z]").astype("float32")/df[f"{col}_len"]
        df[f"{col}_chinese"]=df[f"{col}"].str.count(r'[\u4e00-\u9fff]+').astype("float32")/df[f"{col}_len"]

        # Feature that show how balanced are curly and round brackets
        df[f"{col}_round_balance"]=df[f"{col}_round_open"]-df[f"{col}_round_close"]
        df[f"{col}_curly_balance"]=df[f"{col}_curly_open"]-df[f"{col}_curly_close"]

        # Feature that tells if the string json is present somewhere (e.g. asking a json response or similar)
        # This for example could be expanded also to yaml, but analyses on train set are required to see if enough data is present for this to be really useful
        df[f"{col}_json"]=df[f"{col}"].str.lower().str.count("json")
    return df
    
train=compute_feats(train)
test=compute_feats(test)
```

# <span style="background-color:#d4fad6;color:black;padding:10px;border-radius:40px;">📊 Exploratory Data Analysis</span>

```python
# DataFrame 'train' with the column 'winner'
vc = train['winner'].value_counts()  # Count occurrences of each value

# Create a bar chart
plt.bar(vc.index, vc.values)  # Use index for category labels
# Customize the bar chart for better visualization
plt.xlabel('Models')
plt.ylabel('Count')
plt.title('Winning Models Count')
plt.xticks(ha='right')  # Rotate category labels for readability
plt.tight_layout()  # Adjust spacing to prevent overlapping elements

plt.show()

vc
```

```python
vc = train['winner'].value_counts()

# Plot the pie chart
plt.pie(vc.values, labels=vc.index, autopct="%1.1f%%")  # Add percentages to pie slices
plt.title('Winner Models')
plt.show()
```

<div style="background-color:white;color:black;padding:20px;border:5px solid blue;border-radius:20px;">
It shows that our dataset is well balanced. We see model_a and model_b are almost present in same number for target variable winner. 
</div>

```python
# DataFrame 'train' with the column 'model_a'
vc = train['model_a'].value_counts()  # Count occurrences of each value
# Increase plot size (adjust width and height as needed)
plt.figure(figsize=(12, 6))  # Set width to 12, height to 6
# Create a bar chart
plt.bar(vc.index, vc.values)  # Use index for category labels
# Customize the bar chart for better visualization
plt.xlabel('Model_A')
plt.ylabel('Count')
plt.title('Models Used as Model A')
plt.xticks(rotation=90, ha='right')  # Rotate category labels for readability
plt.tight_layout()  # Adjust spacing to prevent overlapping elements

plt.show()
```

<div style="background-color:white;color:black;padding:20px;border:5px solid blue;border-radius:20px;">
Here we can see all the models that have been used as model A. The bar chart makes it clear that which models were used the most and which were used the least.</div>

```python
# DataFrame 'train' with the column 'model_b'
vc = train['model_b'].value_counts()  # Count occurrences of each value
# Increase plot size (adjust width and height as needed)
plt.figure(figsize=(12, 6))  # Set width to 12, height to 6
# Create a bar chart
plt.bar(vc.index, vc.values)  # Use index for category labels
# Customize the bar chart for better visualization
plt.xlabel('Model_B')
plt.ylabel('Count')
plt.title('Models Used as Model B')
plt.xticks(rotation=90, ha='right')  # Rotate category labels for readability
plt.tight_layout()  # Adjust spacing to prevent overlapping elements

plt.show()
```

<div style="background-color:white;color:black;padding:20px;border:5px solid blue;border-radius:20px;">Here we can see all the models that have been used as model B. The bar chart makes it clear that which models were used the most and which were used the least.</div>

```python
# DataFrame 'train' with the column 'language'
vc = train['language'].value_counts()  # Count occurrences of each value
# Increase plot size (adjust width and height as needed)
plt.figure(figsize=(17, 8))  # Set width to 17, height to 8
# Create a bar chart
plt.bar(vc.index, vc.values)  # Use index for category labels
# Customize the bar chart for better visualization
plt.xlabel('Languages')
plt.ylabel('Count')
plt.title('Languages used in Chat')
plt.xticks(rotation=90, ha='right', fontweight='bold')  # Rotate category labels for readability
plt.tight_layout()  # Adjust spacing to prevent overlapping elements

plt.show()

vc
```

<div style="background-color:white;color:black;padding:20px;border:5px solid blue;border-radius:20px;">We can see that our dataset mostly contains text in English language. While text in other languages is significatly lower as compared to English.</div>

```python
# DataFrame 'train' with the column 'winner'
vc = train['prompt_sentiment'].value_counts()  # Count occurrences of each value

# Create a bar chart
plt.bar(vc.index, vc.values)  # Use index for category labels
# Customize the bar chart for better visualization
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Prompt Sentiment')
plt.xticks(ha='right')  # Rotate category labels for readability
plt.tight_layout()  # Adjust spacing to prevent overlapping elements

plt.show()
```

```python
# DataFrame 'train' with the column 'winner'
vc = train['response_a_sentiment'].value_counts()  # Count occurrences of each value

# Create a bar chart
plt.bar(vc.index, vc.values)  # Use index for category labels
# Customize the bar chart for better visualization
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Response A Sentiment')
plt.xticks(ha='right')  # Rotate category labels for readability
plt.tight_layout()  # Adjust spacing to prevent overlapping elements

plt.show()
```

```python
# DataFrame 'train' with the column 'winner'
vc = train['response_b_sentiment'].value_counts()  # Count occurrences of each value

# Create a bar chart
plt.bar(vc.index, vc.values)  # Use index for category labels
# Customize the bar chart for better visualization
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Response B Sentiment')
plt.xticks(ha='right')  # Rotate category labels for readability
plt.tight_layout()  # Adjust spacing to prevent overlapping elements

plt.show()
```

<div style="background-color:white;color:black;padding:20px;border:5px solid blue;border-radius:20px;">It is clear from the bar charts above that setiment of prompt, response of model A and model B are neutral. This is very important to ensure that we don't have a bias here. If the sentiment was positive or if the sentiment was negative then it could have led to a bias in selecting the winners. Majority of the text is neutral which indicates that this data set is prepared very carefully. This neutrality is highly critical for ensuring that dataset is well suited for a competition like this one.</div>

# <span style="background-color:#d4fad6;color:black;padding:10px;border-radius:40px;">⭐ TF IDF Vectorizer</span>

```python
vectorizer_char = TfidfVectorizer(sublinear_tf=True, analyzer='char', ngram_range=(1,2), max_features=50000)
vectorizer_word = TfidfVectorizer(sublinear_tf=True, analyzer='word', min_df=3)
preprocessor = ColumnTransformer(
    transformers=[
        ('prompt_feats', FeatureUnion([
            ('prompt_char', vectorizer_char),
            ('prompt_word', vectorizer_word)
        ]), 'prompt'),
        ('response_a_feats', FeatureUnion([
            ('response_a_char', vectorizer_char),
            ('response_a_word', vectorizer_word)
        ]), 'response_a'),
        ('response_b_feats', FeatureUnion([
            ('response_b_char', vectorizer_char),
            ('response_b_word', vectorizer_word)
        ]), 'response_b')
    ]
)
train_feats = preprocessor.fit_transform(train[["response_a","response_b","prompt"]])
test_feats = preprocessor.transform(test[["response_a","response_b","prompt"]])
```

# <span style="background-color:#d4fad6;color:black;padding:10px;border-radius:40px;">🚄 Train Model</span>

```python
feats=list(train.columns)[8:]
train["winner"]=(train["winner"]=="model_a").astype("int")
X_train=train[feats]
y_train=train["winner"]
```

```python
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Create the model with early stopping
model = LGBMClassifier(n_estimators=1000,  # Set a large number for early stopping
                        learning_rate=0.1,
                        early_stopping_rounds=15)  # Stop if no improvement in 15 rounds

# Train the model
history = model.fit(X_train, y_train,eval_set=[(X_val, y_val)], eval_metric='binary_logloss')
```

# <span style="background-color:#d4fad6;color:black;padding:10px;border-radius:40px;">🌠 Prediction</span>

```python
X_test=test[feats]
test["winner"]=model.predict(X_test)
```

# <span style="background-color:#d4fad6;color:black;padding:10px;border-radius:40px;">📁 Submission</span>

```python
test["winner"]=test["winner"].apply(lambda x: "model_a" if x==1 else "model_b")
sub=test[["id","winner"]]
sub.head()
```

```python
sub.to_csv("submission.csv",index=False)
```
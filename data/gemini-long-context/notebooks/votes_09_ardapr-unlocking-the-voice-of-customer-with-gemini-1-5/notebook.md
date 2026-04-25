# Unlocking the Voice of Customer with Gemini 1.5

- **Author:** Arda Putra Ryandika
- **Votes:** 56
- **Ref:** ardapr/unlocking-the-voice-of-customer-with-gemini-1-5
- **URL:** https://www.kaggle.com/code/ardapr/unlocking-the-voice-of-customer-with-gemini-1-5
- **Last run:** 2024-12-01 14:56:51.427000

---

![Background.jpg](attachment:5b16e8b2-fed0-4216-917d-bd2013e76f94.jpg)

# **Gemini Long Context - AI4Indonesia - Gemini AI Voice of Customer**
## AI4Indonesia is proud to announce Voice of Customer powered by Gemini 1.5, watch and enjoy this video!

```python
from IPython.display import YouTubeVideo, HTML

# Display a centered YouTube video
video_id = 'ByzGiI1GiHA?si=Fd7Jkuy9BAd3aScB'
HTML(f"""
<div style="display: flex; justify-content: center;">
    <iframe width="1024" height="576" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>
</div>
""")
```

# **Now, why we start this initiative, it's based on our pain points..**

# **1. Background**
## **1.1 The Problem: The Hidden Struggles of Customer Support Agents**
Every Monday, one of our team sits down, bracing for the week ahead. **Over 100,000 complaints pour in every month**, with 20,000 to 30,000 arriving each week. These complaints are not just numbers—they’re customers frustrated with dropped calls, billing errors, or service disruptions. Each one expects a resolution, but the current process makes it almost impossible to meet their expectations.
**The team, made up of just four people**, is forced to analyze **only 100 cases per week each**, addressing less than 5% of the incoming volume. To make sense of the data, **they rely on sampling—looking at a small subset of complaints to identify trends**. But sampling often **leads to generalizations and misinterpretations**, missing critical patterns hidden in the broader data.

## **1.2 The Key Challenge: Limited Context, Flawed Insights**
Existing tools and technologies are not equipped to handle the scale of data these agents face. AI models with short context windows (typically 32,000–128,000 tokens) can only process a fraction of the available information at a time. As a result:
1. *Key connections are missed*, as models cannot grasp the full scope of complaints in context.
2. *Analysis is fragmented*, requiring additional effort to combine outputs.
3. *Decisions are delayed*, leading to suboptimal resolutions and dissatisfied customers.
The root issue remains: existing technology cannot process large, complex datasets in their entirety.

## **1.3 The Key Solution: Why Gemini 1.5 is the Game-Changer?**
**Gemini 1.5 redefines AI capabilities with its 2-million-token context window**, enabling it to process vast datasets—like 20,000 weekly complaints—in one pass, unlike traditional models limited to 32,000–128,000 tokens. This **eliminates the need for workarounds like vector databases and RAG**, offering direct **in-context retrieval for accurate insights**. With Gemini, agents can **analyze entire datasets effortlessly**, identify root causes faster, and make decisions with precision, **transforming customer support from reactive to proactive**.

![Background-1_new.jpg](attachment:e2e5977c-b752-4056-9f76-3cfa00e676cd.jpg)

# **2. Framework Study**
To conduct this research effectively, **we prepare a comprehensive methodology that outlines the steps necessary for utilizing Gemini's long context feature in handling customer service complaints**. This structured approach ensures that our investigation is systematic, rigorous, and capable of yielding meaningful insights.

## **2.1 Objective**
The research begins with a clear design that defines the objectives and scope of the study. **We aim to explore **how advanced Gemini 1.5 capabilities can enhance the analysis and resolution of customer complaints**, ultimately improving customer satisfaction and operational efficiency.

## **2.2 Environmental Setup**
To facilitate our research, we establish an appropriate environment by identifying and installing essential libraries. This includes data manipulation tools like Pandas and NumPy, as well as the Google Cloud library for accessing Gemini’s API. Additionally, we obtain an API key from Google Cloud Console, which is crucial for leveraging the generative AI functionalities offered by Gemini.

## **2.3 Data Preparation and Cleansing**
The next step involves gathering a relevant dataset that contains historical customer service complaints. This dataset typically includes key fields such as customer complaint. The quality and comprehensiveness of this data are critical for the success of our analysis. We will be utilizing the *"customer-service-chat-data-30k-rows"* dataset as a primary example **https://www.kaggle.com/datasets/aimack/customer-service-chat-data-30k-rows**. This dataset contains 30,000 rows of customer service chat interactions, which can be used for various natural language processing (NLP) analyses.

Data cleaning is a vital step in our methodology. We address **missing values through strategies such as imputation or removal, ensuring that our dataset is robust and reliable for analysis**. Furthermore, we eliminate unnecessary columns that do not contribute significantly to our research objectives, streamlining our data for more effective analysis.

## **2.4 Exploratory Data Analysis (EDA)**
With a clean dataset in hand, we **conduct exploratory data analysis to uncover patterns and trends** within the complaints data. This involves calculating summary statistics and creating visualizations to identify relationships among variables and gain insights.

## **2.5 Implementation of Gemini**
The core of our research lies in **implementing Gemini's long context feature**. We initialize the API client and utilize its capabilities to analyze complaint texts comprehensively. By generating contextual responses or insights based on the complaint data, we can assess how effectively AI can support customer service operations.

# **3. Implementation**
In this section, we will begin to explore the practical application of our research methodology by implementing the use case of leveraging Gemini's long context feature to enhance the handling of customer service complaints. This implementation aims to demonstrate how advanced AI capabilities can be integrated into existing customer service frameworks to improve response accuracy, efficiency, and overall customer satisfaction.

## **3.1 Explaning the Libraries**
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
In this implementation, we use several key libraries to facilitate data analysis and AI integration:
<ol>
  <li><strong>Pandas</strong>: <em>For data manipulation and analysis using DataFrames.</em></li>
  <li><strong>NumPy</strong>: <em>For numerical computations on arrays and matrices.</em></li>
  <li><strong>Matplotlib</strong>: <em>For creating graphs and visualization</em></li>
  <li><strong>Json</strong>: <em>For creating and parsing json file format</em></li>
  <li><strong>Datetime</strong>: <em>For handling dates and times in data analysis.</em></li>
  <li><strong>Google Generative AI</strong>: <em>To access Google’s AI capabilities for analyzing text and generating responses.</em></li>
  <li><strong>Kaggle Secrets Client</strong>: <em>For securely accessing API keys and credentials.</em></li>
</ol>
</div>

```python
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# Setting
pd.set_option('display.max_colwidth', None)
```

## **3.2 Explanation of Fetching the Gemini API Key**
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
In this section, we demonstrate how to securely retrieve the Gemini API key using `UserSecretsClient` from Kaggle Secrets. This approach ensures sensitive information like API keys are managed properly and avoids hard-coding them into our codebase.

We initialize `UserSecretsClient`, which enables access to stored secrets in Kaggle's environment. Then, we use its `get_secret` method to fetch the specific 'gemini_api_key'. Finally, we configure Google Generative AI (`genai`) with this retrieved key, enabling authentication and requests to the Gemini API effectivel

This secure retrieval process enhances security by keeping credentials hidden within secret storage rather than being exposed directly in our application code.

</div>

```python
# Fetch Gemini API Key through Secrets
user_secrets = UserSecretsClient()
gemini_api_key = user_secrets.get_secret("gemini_api_key")
genai.configure(api_key=gemini_api_key)
```

## **3.3 Explanation of Reading the Excel File**
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
    
In this section, we specify the file path to an Excel file containing customer service chat data, using the line `file_path = "/kaggle/input/customer-service-chat-data-30k-rows/Chat_Team_CaseStudy FINAL.xlsx"` to define its location within the Kaggle environment; then, we read the file into a Pandas DataFrame named `df` with `df = pd.read_excel(file_path)`, which allows us to efficiently manipulate and analyze the data; finally, we use `df.head(5)` to display the first five rows of the DataFrame, providing a quick overview of the dataset's structure, including column names andata types. 

This datasetet is about 30k customer interaction data with Airline Customer Representatives and we are going to focus more on the 'Text' column which contains the text data on what the customer said first when they initiated the interaction.

</div>

```python
# Read the Excel file
file_path = "/kaggle/input/customer-service-chat-data-30k-rows/Chat_Team_CaseStudy FINAL.xlsx"
df = pd.read_excel(file_path)

df.head(5)
```

### Focus on the *Text* Column
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
We will focus on the 'Text' column because it contains initial customer messages and we want to analyze these interactions further.
</div>

```python
# Focus on 'Text' column
df[['Text']].head(10)
```

## **3.4 Explanation of Filtering and Preparing Text Data**
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
    
This code snippet focuses on filtering non-NaN values from the 'Text' column of the DataFrame and preparing the text data for further analysis.
</div>

```python
# Filter non-NaN values in the 'Text' column and take the first 10k rows to process
non_nan_texts = df['Text'].head(10000).dropna()

# Iterate and append with prefixes
output = ""
for i, text in enumerate(non_nan_texts, start=1):
    output += f"{text} |"

# Show the first 2000 words
output[:2000]
```

```python
# Show the total interaction
total_interaction = i

print(f"There are in total {total_interaction} non-null interactions")
```

## **3.5 Exploratory Data Analysis (EDA)**
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
The objective of this code is to process time-related data for analysis by converting time durations into usable formats. This is particularly useful when analyzing metrics like chat durations, response times, and customer wait times in seconds.
    
<strong>Key Steps:</strong>
<ol>
  <li><strong>Convert Time Strings to Datetime Objects:</strong>
    <ul>
      <li>The code converts time data from string format (e.g., `'HH:MM:SS'`) into datetime objects.</li>
      <li>This conversion simplifies calculations and ensures consistency in handling time values.</li>
      <li>It uses the <code>pd.to_datetime</code> function with specified formatting and error handling (<code>errors='coerce'</code>).</li>
    </ul>
  </li>
  
  <li><strong>Calculate Total Duration in Seconds:</strong>
    <ul>
      <li>The <code>Time Chat Duration</code> column is further processed to compute the total time in seconds.</li>
      <li>The calculation extracts hours, minutes, and seconds, converting them into a single numerical value for each entry.</li>
      <li>This enables easier aggregation, comparison, and numerical analysis of time data.</li>
    </ul>
  </li>
</ol>

<strong>Purpose</strong>

This preprocessing step is essential in contexts like:
- Evaluating service efficiency, such as chat response times or customer wait durations.
- Normalizing time data to a single, comparable format for advanced analysis or visualization.
</div>

```python
#convert to date time
df['Time Chat Duration'] = pd.to_datetime(df['Chat Duration'], format='%H:%M:%S', errors='coerce')
df['Time Response Time of Agent'] = pd.to_datetime(df['Response Time of Agent'], format='%H:%M:%S', errors='coerce')
df['Time Response time of Visitor'] = pd.to_datetime(df['Response time of Visitor'], format='%H:%M:%S', errors='coerce')
df['Time Customer Wait Time'] = pd.to_datetime(df['Customer Wait Time'], format='%H:%M:%S', errors='coerce')

#convert to second
df['Second Chat Duration'] = (df['Time Chat Duration'].dt.hour*3600)+(df['Time Chat Duration'].dt.minute*60)+(df['Time Chat Duration'].dt.second)
```

### Visualizing Chat Duration
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code analyzes and visualizes the distribution of chat durations in seconds.

<ol>
  <li><strong>KDE Plot:</strong>
    <ul>
      <li>Uses <code>seaborn</code> to create a Kernel Density Estimate (KDE) plot for <code>Second Chat Duration</code>.</li>
      <li>Highlights the density and spread of chat durations, showing common ranges.</li>
    </ul>
  </li>

  <li><strong>Statistical Summary:</strong>
    <ul>
      <li>The <code>describe()</code> method provides key statistics (e.g., mean, min, max, percentiles) for <code>Second Chat Duration</code>.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
These steps help identify patterns, typical durations, and outliers to improve service efficiency.
</div>

```python
#visualize the distribution of Chat Duration

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.kdeplot(data=df, x='Second Chat Duration', fill=True)
plt.xlabel('Chat Duration in Second')
plt.ylabel('Density')
plt.title('KDE Chat Duration in Second')
plt.show()

df['Second Chat Duration'].describe()
```

### Visualizing Customer Comment Distribution
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code analyzes and visualizes the distribution of customer comments:

<ol>
  <li><strong>Count Comments:</strong>
    <ul>
      <li>Filters out empty comments and counts occurrences of each unique value in the <code>Customer Comment</code> column.</li>
    </ul>
  </li>

  <li><strong>Pie Chart Visualization:</strong>
    <ul>
      <li>Displays the percentage distribution of different comment resolutions using a pie chart.</li>
      <li>Adds a title and removes the y-axis label for a cleaner look.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The visualization highlights the proportion of different customer comments, providing insights into resolution trends.
</div>

```python
#visualize the distribution of Customer Comment
com_count = df[df['Customer Comment']!= ' ']['Customer Comment'].value_counts()

# Create a bar chart
plt.figure(figsize=(12, 5))

# Create a pie chart
com_count.plot.pie(autopct='%1.1f%%', startangle=140)
plt.title('Number of Comment Resolution  Distribution')
plt.ylabel('')  # Remove y-axis label
plt.show()

com_count
```

### Visualizing Customer Rating Distribution
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code analyzes and visualizes customer ratings:

<ol>
  <li><strong>Count Ratings:</strong>
    <ul>
      <li>Filters out empty ratings and counts occurrences of each unique rating value.</li>
    </ul>
  </li>

  <li><strong>Bar Chart Visualization:</strong>
    <ul>
      <li>Displays the number of interactions for each rating using a bar chart.</li>
      <li>Includes a title, and axis labels for clarity.</li>
    </ul>
  </li>

  <li><strong>Statistical Summary:</strong>
    <ul>
      <li>Provides a summary of the <code>Customer Rating</code> column, including count, mean, and other key statistics.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
This analysis helps understand customer satisfaction trends by visualizing rating distributions and summarizing their key statistics.
</div>

```python
#visualize the distribution of Rating
rat_count = df[df['Customer Rating']!= ' ']['Customer Rating'].value_counts()
rating_only = df[df['Customer Rating']!= ' ']['Customer Rating']

# Create a bar chart
plt.figure(figsize=(12, 5))
rat_count.plot.bar()
plt.title('Number of Rating by Customer')
plt.xlabel('Rating')
plt.ylabel('Count Interaction')
plt.show()

rating_only.describe()
```

## **3.6 Gemini Context Caching**
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code establishes a caching mechanism for customer interaction analysis to enhance efficiency and reduce redundant processing.

<ol>
  <li><strong>Caching Customer Interaction Data:</strong>
    <ul>
      <li>The <code>genai.caching.CachedContent.create()</code> method is used to store the context of customer interactions in memory for reuse, reducing the need to repeatedly process the same data.</li>
      <li>The model <code>"gemini-1.5-pro-002"</code> is selected to perform the analysis of the interactions, ensuring consistent and high-quality insights.</li>
    </ul>
  </li>

  <li><strong>System Instruction:</strong>
    <ul>
      <li>The system instruction defines the AI’s role as an analyzer of customer interactions, emphasizing the need for concise, professional, and contextually relevant responses.</li>
      <li>The guidelines ensure that the AI’s output is focused solely on extracting insights from the provided data without any extraneous commentary.</li>
    </ul>
  </li>

  <li><strong>TTL (Time to Live):</strong>
    <ul>
      <li>The cache has a time limit (<code>ttl=datetime.timedelta(minutes=60)</code>), which means the stored data will remain valid for 60 minutes before it is refreshed or discarded.</li>
    </ul>
  </li>

  <li><strong>Display Name:</strong>
    <ul>
      <li>The cached content is given a name (<code>"interactions-cache"</code>) for easy identification and management in the system.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose of caching is to optimize performance by storing the analyzed content temporarily. This <strong>reduces the computational load by allowing repeated access to the analysis context without needing to recompute it</strong>, making the system faster and more efficient when processing similar customer interactions within a short time frame.
</div>

```python
# Context Caching for all Interactions

cached_content = genai.caching.CachedContent.create(
    model="gemini-1.5-pro-002",
    system_instruction="""
        You are an AI Customer Interaction Analyzer designed to assist analysts in extracting insights from customer interactions. 
        Your responses should be concise, well-structured, and relevant to the context of the provided data.  
        You must adhere to the following guidelines:  
            1. Focus solely on analyzing and providing insights from the customer interactions.  
            2. Format your answers in a clear and professional manner, suitable for analytical purposes.  
            3. Avoid including any information or commentary outside the scope of the provided data.  
        """,
    contents=[output],
    ttl=datetime.timedelta(minutes=60),
    display_name="interactions-cache",
)
```

### **3.6.1 Build the In-Context Chatbot using Cached Content**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code initializes the generative AI model using previously cached data and starts an interactive chat session.

<ol>
  <li><strong>Initialize the Model:</strong>
    <ul>
      <li>The <code>genai.GenerativeModel</code> is initialized with the model name <code>"gemini-1.5-pro-002"</code>, designed for analyzing customer interactions.</li>
      <li>The <code>from_cached_content()</code> method is called to load the cached content (<code>cached_content</code>), ensuring that the model has the relevant context without needing to reprocess the data.</li>
    </ul>
  </li>

  <li><strong>Start an Interactive Chat:</strong>
    <ul>
      <li>The <code>start_chat()</code> method initiates a new chat session, allowing users to query and generate insights based on the cached context.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The objective of this process is to <strong>enable efficient interaction with the AI model using cached context</strong>, improving performance by reducing the need for repetitive data processing. This setup allows for faster, more relevant responses during the chat session, tailored to the given interaction context.
</div>

```python
# Initialize the Model from the Cached Content
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",
).from_cached_content(cached_content=cached_content)

# Start asking Gemini
model_chat = model.start_chat()
```

## **3.7 Use Case 1 : Build the In-Context Analysis and Categorization**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code leverages Gemini’s AI model to instantly categorize customer interactions based on predefined criteria and provide an organized summary.

<ol>
  <li><strong>Creating Categorization Prompt:</strong>
    <ul>
      <li>A prompt is generated to guide the AI in categorizing customer interactions, provided as a series of text separated by a delimiter ("|").</li>
      <li>The AI is tasked to assign each interaction to a specific category or label them as "Others" when the content is unclear.</li>
    </ul>
  </li>

  <li><strong>Interaction Analysis:</strong>
    <ul>
      <li>The AI processes the interactions, calculates the total count, and computes the percentage for each category.</li>
      <li>The results are sorted in descending order by percentage, allowing for easy identification of the most common categories.</li>
    </ul>
  </li>

  <li><strong>Presenting Results in JSON Format:</strong>
    <ul>
      <li>The categorized results are returned in a JSON format, featuring a table with columns for Category Name, Category Definition, Count, and Percentage.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose of this process is to <strong>categorize and analyze</strong> customer interactions efficiently, providing a clear breakdown of interaction types and their distribution, helping identify trends and improving service quality.
</div>

```python
# Get the breakdown of Interactions instantly with the help of Gemini

categorization_prompt = f'''
The text provided contains a total of {total_interaction} customer-initiated interactions with an airline customer service representative, separated by "|".  

Your task is to:  
1. Categorize each interaction based on its content. Dont create ambiguous categories or categories that can easily overlap with each other. Use "Others" for interactions that do not fit into any specific category. Create max 8 categories.
2. Count the total interactions for each category and calculate their percentage, ensuring the sum of percentages equals 100%. It has to be ACCURATE.
3. Present the results in JSON format as a table with the following columns: (The column name have to be exact)
    a. Category Name: Name of the category.
    b. Category Definition: A clear but sharp description of the category to ensure that no overlap happens
    c. Count: Number of interactions in this category. This need to be accurate.
    d. Percentage: Percentage of interactions in this category (rounded 1 decimal). This need to be accurate.
4. Sort the JSON table by the Percentage column in descending order and name the JSON object categorization.

You need to be accurate
'''


categorization_response = model_chat.send_message(
    categorization_prompt,
    generation_config={"response_mime_type": "application/json", 
                       'temperature': 0,
                       'top_k': 1,
                       'top_p': 1,
                       'candidate_count': 1,
                       'max_output_tokens': 1500,
                       'stop_sequences': ["STOP!"]}
)

categorization_response.text
```

### **3.7.1 Categorizing and Structuring Interaction Data**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code extracts and structures the categorization data into a DataFrame for further analysis.

<ol>
  <li><strong>Parse JSON Response:</strong>
    <ul>
      <li>The response from the categorization process is in JSON format. The <code>json.loads()</code> function is used to load the response text into a Python dictionary.</li>
    </ul>
  </li>

  <li><strong>Create DataFrame:</strong>
    <ul>
      <li>A pandas DataFrame (<code>categorization_df</code>) is created using the parsed JSON data, specifically extracting the <code>categorization</code> section of the response.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose of this step is to <strong>transform the categorization results into a structured format</strong> (DataFrame), enabling easier analysis and manipulation of the categorized interaction data.
</div>

```python
categorization_df = pd.DataFrame(json.loads(categorization_response.text)['categorization'])

categorization_df
```

### **3.7.2 Visualizing Categorization Counts**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code visualizes the distribution of category counts using a horizontal bar chart.

<ol>
  <li><strong>Create Bar Chart:</strong>
    <ul>
      <li>The <code>barh()</code> function is used to create a horizontal bar chart, representing the count of each category from the DataFrame.</li>
      <li>The bars are styled with a Google-like blue color for better visual appeal.</li>
    </ul>
  </li>

  <li><strong>Set Title and Labels:</strong>
    <ul>
      <li>The chart's title and axis labels are set with a bold, clean font for clarity.</li>
    </ul>
  </li>

  <li><strong>Style the Chart:</strong>
    <ul>
      <li>The axis tick labels are adjusted for better readability, using the Roboto font and appropriate size.</li>
    </ul>
  </li>

  <li><strong>Display Count Labels:</strong>
    <ul>
      <li>Count values are displayed on top of each bar for clarity using <code>plt.text()</code>.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose of this visualization is to clearly represent the count distribution of categories, making it easy to compare their frequencies in a visually appealing format.
</div>

```python
import matplotlib.pyplot as plt

# Create a bar chart
plt.figure(figsize=(10, 6))
bars = plt.barh(
    categorization_df['Category Name'], 
    categorization_df['Count'].astype('int'), 
    color='#4285F4')  # Google-like blue color

# Set the title and labels
plt.title('Category Count Distribution', fontsize=16, fontweight='bold', family='Roboto')
plt.xlabel('Count', fontsize=14, fontweight='bold', family='Roboto')
plt.ylabel('Category Name', fontsize=14, fontweight='bold', family='Roboto')

# Style the chart with Google-like aesthetics
plt.gca().tick_params(axis='y', labelsize=12)
plt.gca().tick_params(axis='x', labelsize=12)

# Display the count labels on the bars
for bar in bars:
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', 
             va='center', fontsize=12, fontweight='bold', family='Roboto')

# Show the plot
plt.tight_layout()
plt.show()
```

## **3.8 Use Case 2 : Conduct More Detailed Analysis for Specific Issue**
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code sends a prompt to generate a detailed breakdown of the "Baggage Issue" category into subcategories.

<ol>
  <li><strong>Define the Prompt:</strong>
    <ul>
      <li>The prompt specifies the task: breaking down the "Baggage Issue" category into subcategories based on themes from customer interactions.</li>
      <li>It also instructs the model to calculate counts and percentages for each subcategory, with results sorted by percentage.</li>
    </ul>
  </li>

  <li><strong>Send Request to Model:</strong>
    <ul>
      <li>The prompt is sent to the model using <code>model_chat.send_message()</code>, with the response expected in JSON format.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose is to <strong>further analyze the "Baggage Issue" category</strong> by breaking it down into more specific subcategories, enabling a deeper understanding of customer concerns and patterns.
</div>

```python
# Ask a more detailed breakdown to specific issue

detailed_breakdown_prompt = '''
The previous analysis identified "Baggage Issue" as one of the categories. The text provided contains details about interactions classified under this category.  

Your task is to:  
1. Further break down the "Baggage Issue" category into subcategories based on specific themes or issues mentioned in the interactions.
2. Calculate the total count and percentage of interactions for each subcategory.  
3. Present the results in descending order of percentage (highest to lowest).  

The output should be in JSON format of a table ith 'detailed_breakdown' as the name and these columns:
    1. Category
    2. Sub Category
    3. Sub Category Definition, 
    4. Count
    5. Percent
'''

detailed_breakdown_response = model_chat.send_message(
    detailed_breakdown_prompt,
    generation_config={"response_mime_type": "application/json", 
                       'temperature': 0,
                       'top_k': 1,
                       'top_p': 1,
                       'candidate_count': 1,
                       'max_output_tokens': 1500,
                       'stop_sequences': ["STOP!"]}
)

detailed_breakdown_response.text
```

### **3.81. Structuring the Detailed Breakdown Data**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code extracts and structures the detailed breakdown data into a DataFrame for further analysis.

<ol>
  <li><strong>Parse JSON Response:</strong>
    <ul>
      <li>The response from the detailed breakdown process is in JSON format. The <code>json.loads()</code> function is used to parse the response text into a Python dictionary.</li>
    </ul>
  </li>

  <li><strong>Create DataFrame:</strong>
    <ul>
      <li>A pandas DataFrame (<code>detailed_breakdown_df</code>) is created using the parsed JSON data, specifically extracting the <code>detailed_breakdown</code> section of the response.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose of this step is to <strong>structure the detailed breakdown results into a DataFrame</strong>, making it easier to analyze and manipulate the subcategory data for further insights.
</div>

```python
detailed_breakdown_df = pd.DataFrame(json.loads(detailed_breakdown_response.text)['detailed_breakdown'])

detailed_breakdown_df
```

### **3.8.2 Visualizing Sub Categorization Counts**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code visualizes the distribution of category counts using a horizontal bar chart.

<ol>
  <li><strong>Create Bar Chart:</strong>
    <ul>
      <li>The <code>barh()</code> function is used to create a horizontal bar chart, representing the count of each category from the DataFrame.</li>
      <li>The bars are styled with a Google-like blue color for better visual appeal.</li>
    </ul>
  </li>

  <li><strong>Set Title and Labels:</strong>
    <ul>
      <li>The chart's title and axis labels are set with a bold, clean font for clarity.</li>
    </ul>
  </li>

  <li><strong>Style the Chart:</strong>
    <ul>
      <li>The axis tick labels are adjusted for better readability, using the Roboto font and appropriate size.</li>
    </ul>
  </li>

  <li><strong>Display Count Labels:</strong>
    <ul>
      <li>Count values are displayed on top of each bar for clarity using <code>plt.text()</code>.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose of this visualization is to clearly represent the count distribution of sub categories, making it easy to compare their frequencies in a visually appealing format.
</div>

```python
import matplotlib.pyplot as plt
import numpy as np

# Create a bar chart
plt.figure(figsize=(12, 7))

# Adjust the bar colors with a gradient effect
bars = plt.barh(
    detailed_breakdown_df['Sub Category'], 
    detailed_breakdown_df['Count'].astype('int'), 
    color=plt.cm.Blues(np.linspace(0.3, 0.7, len(detailed_breakdown_df)))  # Gradient effect
)

# Set the title and labels
plt.title('Sub Category Count Distribution', fontsize=18, fontweight='bold', family='Arial')
plt.xlabel('Count', fontsize=16, fontweight='bold', family='Arial')
plt.ylabel('Category Name', fontsize=16, fontweight='bold', family='Arial')

# Style the chart
plt.gca().tick_params(axis='y', labelsize=12)
plt.gca().tick_params(axis='x', labelsize=12)

# Add grid lines for better readability
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Display the count labels on the bars
for bar in bars:
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', 
             va='center', fontsize=12, fontweight='bold', family='Arial', color='black')

# Set the background color for the plot
plt.gca().set_facecolor('#f2f2f2')

# Show the plot
plt.tight_layout()
plt.show()
```

## **3.9 Use Case 3: Create Recommendations to Mitigate the Issues**
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code sends a prompt asking for actionable recommendations to address baggage issues identified in customer interactions.

<ol>
  <li><strong>Define the Prompt:</strong>
    <ul>
      <li>The prompt requests actionable recommendations for the airline on how to address common baggage issues based on customer feedback.</li>
    </ul>
  </li>

  <li><strong>Send Request to Model:</strong>
    <ul>
      <li>The prompt is sent to the AI model using <code>model_chat.send_message()</code>, expecting a response with actionable insights.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose is to <strong>gather actionable recommendations</strong> that can help the airline improve their service by addressing the most common baggage-related issues raised by customers.
</div>

```python
# Ask a more detailed breakdown to specific issue

recommendation_prompt = '''
Could you provide actionable recommendations to the airline regarding what they can do to tackle most of the baggage issues?
'''

recommendation_response = model_chat.send_message(recommendation_prompt)

recommendation_response.text
```

<div style="border: 2px solid #000; padding: 10px; margin: 10px;">

# Actionable Recommendations to Address Baggage Issues  

Based on the provided data, here's a breakdown of actionable recommendations to address the baggage issues, focusing on the most prevalent subcategories:  

## **1. Baggage Allowance/Fees**  

- **Increase Clarity and Transparency:**  
  Ensure baggage policies are easily accessible and clearly stated on the website, during booking, and in pre-trip communications. Use visual aids to illustrate size and weight restrictions. Offer a baggage calculator tool online.  

- **Competitive Pricing:**  
  Analyze competitors' baggage fees and adjust pricing strategies to be more competitive. Consider offering discounts for pre-paying baggage fees online or for frequent flyers. Explore bundled baggage options.  

- **Flexible Baggage Options:**  
  Introduce more flexible baggage options to cater to diverse traveler needs, such as allowing multiple smaller bags instead of one large bag within the weight limit. Consider offering weight-based pricing for checked baggage, as some travelers are willing to pay for slight increases.  

## **2. Lost/Delayed Baggage**  

- **Improved Tracking and Communication:**  
  Invest in robust baggage tracking technology and provide real-time updates to passengers via SMS, email, and mobile app notifications. Implement proactive communication if baggage is delayed, providing estimated delivery times and clear instructions for claims.  

- **Streamlined Claims Process:**  
  Simplify the baggage claim process by making forms readily available online and through the mobile app. Offer expedited processing for urgent claims. Provide clear communication throughout the claim process, including regular updates on the claim status. Consider offering interim expense reimbursement for essential items when baggage is significantly delayed.  

- **Partnerships with Courier Services:**  
  Partner with reliable courier services for efficient delivery of delayed baggage directly to passengers, reducing inconvenience and wait times. Offer options for alternative delivery locations.  

## **3. Damaged Baggage **  

- **More Durable Baggage Handling:**  
  Review baggage handling procedures and equipment to minimize damage. Invest in training programs for baggage handlers to emphasize careful handling. Implement quality control measures to identify areas for improvement.  

- **Expedited Repair/Replacement Options:**  
  Partner with luggage repair services or offer convenient replacement options for damaged bags. Provide pre-paid shipping labels for sending damaged bags for repair or replacement.  

- **Simplified Claims for Damaged Items:**  
  Simplify the claims process for damaged items by allowing passengers to submit claims online with photos as supporting documentation. Offer fair and timely compensation for damaged belongings.  

## **4. Baggage Policy/Restrictions**  

- **Multilingual Support:**  
  Offer baggage policy information in multiple languages to cater to international travelers. Ensure customer service representatives are equipped to handle inquiries in various languages.  

- **Proactive Information at Check-in:**  
  Remind passengers of baggage policies and restrictions at check-in, both online and at the airport, to avoid surprises and unexpected fees. Highlight any specific regulations for connecting flights or international destinations.  

## **5. Baggage Transfer/Interlining**  

- **Stronger Interline Agreements:**  
  Strengthen interline agreements with partner airlines to ensure seamless baggage transfer for connecting flights. Provide clear instructions and information to passengers about baggage transfer procedures when booking multi-airline itineraries.  

- **Improved Baggage Tagging:**  
  Implement clear and consistent baggage tagging procedures to minimize the risk of misrouting or lost baggage during transfers. Utilize technology to track baggage throughout the transfer process.  

By focusing on these key areas, the airline can proactively address the majority of baggage-related issues, enhance customer satisfaction, and improve its overall operational efficiency.

## **3.10 Mock-up Solution**
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
In our commitment to enhancing user experience, we have developed an innovative platform designed to visualize data seamlessly. This platform not only transforms complex data into clear, interactive visuals but also ensures ease of use for all users. By integrating intuitive design and advanced technology, we empower users to effortlessly navigate through insights, making data-driven decisions more accessible than ever. Whether you're analyzing trends, identifying key metrics, or exploring detailed findings, our platform is your go-to tool for turning data into actionable knowledge.
</div>

![mockup_all_new.jpg](attachment:3a3ad0ff-e109-443c-b3f0-032b85dc57a1.jpg)

# **4. Validation Analysis and Discussion**

### **Validation Analysis: Can Gemini 1.5 Understand Long Context in Customer Complaints?**

### The Challenge

We set out to test if **Gemini 1.5** could accurately categorize customer complaints, handling long, complex contexts. While our system was designed to categorize complaints into categories, sub-categories, and events with recommendations, the key question remained: **Is this correct?**

### Our Approach

To validate **Gemini 1.5’s** performance, we followed a structured process:

1. **Select a Baseline**  
   We chose the **top 500 customer complaints** as our sample to ensure a manageable, representative dataset.

2. **Optimize with Caching**  
   We implemented caching to reduce costs and enhance performance by storing analyzed content temporarily.

3. **Categorization with Gemini**  
   We tasked **Gemini 1.5** with creating the categorization schema—identifying categories, sub-categories, and events from the complaints.

4. **Prediction & Comparison**  
   **Gemini 1.5** predicted the categories for all 500 complaints, which we then compared against a manually curated baseline (Use Case 1).

### The Question: Was It Correct?

The results were promising—**Gemini 1.5** showed great accuracy, but there were minor discrepancies, such as misclassified sub-categories or over-generalized events. However, overall, the moreach even higher accuracy.

## **4.1 Select a Baseline: Filter Non-NaN Values in the 'Text' Column**|

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code filters out NaN values from the first 500 entries in the 'Text' column and counts the remaining non-null values.

<ol>
  <li><strong>Filter Non-NaN Values:</strong>
    <ul>
      <li>The <code>dropna()</code> method removes any NaN (missing) values from the first 500 rows in the 'Text' column of the DataFrame.</li>
    </ul>
  </li>

  <li><strong>Count Remaining Values:</strong>
    <ul>
      <li>The <code>count()</code> method is used to determine how many non-null values remain in the filtered data.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose of this step is to <strong>filter and count the valid text entries</strong> in the dataset, ensuring only non-missing data is considered for analysis.
</div>

```python
# Filter non-NaN values in the 'Text' column
validation_texts = df['Text'].head(500).dropna()

validation_texts.count()
```

## **4.2 Optimize Caching: Iterating and Caching Validation Interactions**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code processes the validation texts and prepares them for AI analysis by iterating through the data, appending it with specific prefixes, and caching the result.

<ol>
  <li><strong>Iterate and Append with Prefixes:</strong>
    <ul>
      <li>The code loops through the first 500 non-null values of the 'Text' column and appends each entry with a separator ("|"), forming a continuous string of validation texts.</li>
    </ul>
  </li>

  <li><strong>Create Context Cache for Validation Interactions:</strong>
    <ul>
      <li>The <code>genai.caching.CachedContent.create()</code> method stores the validation text as a cached context. This ensures the AI model can access and process the data efficiently without reprocessing it each time.</li>
    </ul>
  </li>

  <li><strong>Initialize the Model from Cached Content:</strong>
    <ul>
      <li>The <code>genai.GenerativeModel</code> is initialized using the cached validation content, allowing the AI model to generate responses based on the stored interactions.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose is to <strong>prepare and cache the validation interaction data</strong> to enable fast and efficient analysis by the AI model, reducing processing time and ensuring relevant insights are generated from the stored content.
</div>

```python
# Iterate and append with prefixes
output = ""
for i, text in enumerate(validation_texts, start=1):
    output += f"{text} |"

total_interaction_val = i

# Context Caching for validation Interactions
cached_content_val = genai.caching.CachedContent.create(
    model="gemini-1.5-pro-002",
    system_instruction="""
        You are a very smart and meticulous AI Customer Interaction Analyzer designed to assist analysts in extracting insights from customer interactions. 
        Your responses should be concise, well-structured, and relevant to the context of the provided data.  
        You must adhere to the following guidelines:  
            1. Focus solely on analyzing and providing insights from the customer interactions.  
            2. Format your answers in a clear and professional manner, suitable for analytical purposes.  
            3. Avoid including any information or commentary outside the scope of the provided data.  
            4. You have to be accurate when categorizing and counting the interactions.
        """,
    contents=[output],
    ttl=datetime.timedelta(minutes=60),
    display_name="interactions-cache",
)

# Initialize the Model from the Cached Content
model_val = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",
).from_cached_content(cached_content=cached_content_val)
```

## **4.3 Categorize using Gemini: Breakdown of Interactions**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code leverages the Gemini AI model to categorize customer interactions into different categories and provide a detailed breakdown of those interactions.

<ol>
  <li><strong>Generate Categorization Prompt:</strong>
    <ul>
      <li>A categorization prompt is created dynamically using the total number of customer interactions. The task is to categorize each interaction based on its content, ensuring all categories are accurate and percentages add up to 100%.</li>
    </ul>
  </li>

  <li><strong>Generate Categorization Response:</strong>
    <ul>
      <li>The <code>generate_content()</code> method is used to send the prompt to the AI model and receive a categorized breakdown of the interactions in JSON format.</li>
    </ul>
  </li>

  <li><strong>Load Categorization Data into DataFrame:</strong>
    <ul>
      <li>The response is converted from JSON into a pandas DataFrame for easy manipulation and analysis. The categorization table contains the category name, definition, count, and percentage for each category.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The objective is to <strong>categorize and quantify the interactions</strong> accurately, providing a structured breakdown to help identify trends and areas of focus for further analysis.
</div>

```python
# Get the breakdown of Interactions instantly with the help of Gemini

categorization_prompt = f'''
The text provided contains a total of {total_interaction_val} customer-initiated interactions with an airline customer service representative, separated by "|".  

Your task is to:  
1. Categorize each interaction based on its content. Dont create ambiguous categories or categories that can easily overlap with each other. Use "Others" for interactions that do not fit into any specific category. Create max 8 categories.
2. Count the total interactions for each category and calculate their percentage, ensuring the sum of percentages equals 100%. It has to be ACCURATE.
3. Present the results in JSON format as a table with the following columns: (The column name have to be exact)
    a. Category Name: Name of the category.
    b. Category Definition: A clear but sharp description of the category to ensure that no overlap happens
    c. Count: Number of interactions in this category. This need to be accurate.
    d. Percentage: Percentage of interactions in this category (rounded 1 decimal). This need to be accurate.
4. Sort the JSON table by the Percentage column in descending order and name the JSON object categorization.

You need to be accurate
'''

categorization_val_response = model_val.generate_content(
    categorization_prompt,
    generation_config={"response_mime_type": "application/json", 
                       'temperature': 0,
                       'top_k': 1,
                       'top_p': 1,
                       'candidate_count': 1,
                       'max_output_tokens': 1500,
                       'stop_sequences': ["STOP!"]}
)

categorization_val_df = pd.DataFrame(json.loads(categorization_val_response.text)['categorization'])

categorization_val_df
```

## **4.4 Extracting and Displaying Categorization List**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code extracts the relevant columns from the categorized DataFrame and presents them in a readable string format.

<ol>
  <li><strong>Extract Relevant Columns:</strong>
    <ul>
      <li>The code selects the 'Category Name' and 'Category Definition' columns from the categorized DataFrame (<code>categorization_val_df</code>) for further display.</li>
    </ul>
  </li>

  <li><strong>Convert to String Format:</strong>
    <ul>
      <li>The selected columns are converted into a string format using the <code>to_string()</code> method, making it easier to visualize the data without the index.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose of this step is to <strong>present a clean and readable summary</strong> of the categories and their definitions, without the extra DataFrame index, for easier analysis and reporting.
</div>

```python
categorization_val_df = pd.DataFrame(json.loads(categorization_val_response.text)['categorization'])

categorization_list = categorization_val_df[['Category Name', 'Category Definition']].to_dict(orient='tight',index=False)

categorization_list
```

## **4.5 Generating Model Predictions without Context**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code uses a generative model to categorize interactions based on a pre-defined categorization list, without relying on cached context.

<ol>
  <li><strong>Initialize the Model:</strong>
    <ul>
      <li>The generative model is initialized using the model name <code>"gemini-1.5-flash-002"</code>, without any prior context.</li>
    </ul>
  </li>

  <li><strong>Generate Category Predictions:</strong>
    <ul>
      <li>The model processes each interaction in <code>validation_texts</code> to predict a category.</li>
      <li>The prompt asks the model to match each interaction to the most appropriate category from the provided categorization list, returning the result in JSON format under the key 'category'.</li>
    </ul>
  </li>

  <li><strong>Store Results:</strong>
    <ul>
      <li>The results (interaction and corresponding category) are stored in the <code>validation</code> dictionary, which is later converted into a DataFrame for easier viewing and analysis.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose of this process is to <strong>categorize customer interactions</strong> dynamically using a generative AI model, allowing for predictions without relying on cached data, and store the results in a structured format for further analysis.
</div>

```python
# Generate a Model without Context
model_raw = genai.GenerativeModel(
    model_name="gemini-1.5-flash-002",
)

validation = {'interaction': [], 'category': []}

for interaction in validation_texts:
    category_response = model_raw.generate_content(
        contents=[
            "You are an AI trained to categorize interactions based on the provided category names and definitions. Given the category name and its detailed definition, your task is to accurately classify interactions into the correct category. Below are the categories and their definitions. When provided with an interaction, you should categorize it based on the closest match to the definitions given and focus on the bigger picture rather than object",
            f"This is the interaction: {interaction}", 
            f"Based on the Category Name and definition provided in this list {categorization_list}, provide the most suitable category based on the interaction. Only provide 1 with higher priority given to earlier categories if there's overlap"
            ,"Provide the output in json called 'category' "
        ],
        generation_config={"response_mime_type": "application/json",
                           'temperature': 0,
                           'top_k': 1,
                           'top_p': 1,
                           'candidate_count': 1,
                           'max_output_tokens': 200,
                           'stop_sequences': ["STOP!"]}
    )
    validation['interaction'].append(interaction)
    validation['category'].append(json.loads(category_response.text)['category'])

df_validation = pd.DataFrame(validation)

df_validation.head(10)
```

## **4.6 Calculate Category Percentage Distribution**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code calculates and formats the percentage distribution of categories in the <code>df_validation</code> DataFrame.

<ol>
  <li><strong>Calculate Percentage Value Counts:</strong>
    <ul>
      <li>The <code>value_counts(normalize=True)</code> method is used on the <code>'category'</code> column to calculate the proportion of each category in the DataFrame.</li>
      <li>The result is multiplied by 100 to express the counts as percentages.</li>
    </ul>
  </li>

  <li><strong>Format for Readability:</strong>
    <ul>
      <li>The percentages are rounded to two decimal places using the <code>round(2)</code> method for better readability.</li>
    </ul>
  </li>

  <li><strong>Display the Results:</strong>
    <ul>
      <li>The final percentage distribution is printed to the console for easy inspection.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>
The purpose of this process is to <strong>calculate and display the percentage distribution of categories</strong> in the customer interaction dataset, providing a clear view of category prevalence.
</div>

```python
# Calculate percentage value counts
category_percentage = df_validation['category'].astype('str').value_counts(normalize=True) * 100

# Format for better readability
category_percentage = category_percentage.round(2)

print(category_percentage)
```

## **4.7 Comparison Category Prediction with Long Context vs Iterative**
### **4.7.1 Comparison of Category Prediction: Long Context vs. Iterative**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
This code compares the results of two methods for categorizing customer interactions:

<ol>
  <li><strong>Data Preparation:</strong>
    <ul>
      <li>The `category_percentage` Series (Iterative Prediction percentages) is converted into a DataFrame for easy merging and renamed to `Validation Percentage`.</li>
      <li>The `categorization_val_df` (Long Context Prediction percentages) is renamed to standardize column names, with its percentage column labeled as `Gemini Long Context Percentage`.</li>
    </ul>
  </li>

  <li><strong>Merge and Comparison:</strong>
    <ul>
      <li>The two DataFrames are merged on the `Category` column to align the results of both methods.</li>
      <li>Any missing data is handled by dropping NaN values, ensuring only categories present in both datasets are included in the comparison.</li>
      <li>The merged DataFrame is sorted by `Validation Percentage` for clearer analysis.</li>
    </ul>
  </li>
</ol>

<strong>Purpose:</strong>  
The objective of this comparison is to <strong>identify patterns, differences, and consistencies between the Long Context and Iterative Prediction methods</strong> for more informed decision-making in category prediction.
</div>

```python
# Convert the `category_percentage` Series to a DataFrame
category_percentage_df = category_percentage.reset_index()
category_percentage_df.columns = ['Category', 'Validation Percentage']

# Ensure `categorization_val_df` has the necessary columns
categorization_val_df = categorization_val_df.rename(columns={'Percentage': 'Gemini Long Context Percentage', 'Category Name': 'Category'})

# Merge the two DataFrames on the 'Category' column
comparison_df = pd.merge(categorization_val_df[['Category', 'Gemini Long Context Percentage']], category_percentage_df, 
                         on='Category', how='outer')

# Fill NaN values with 0 (if any categories are not in both DataFrames)
comparison_df.dropna(inplace=True)

comparison_df.sort_values('Validation Percentage', ascending=False)
```

### **4.7.2 Validation Results**
<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">

The comparison between categorization using the long-context approach and iterative validation demonstrates key insights regarding efficiency, accuracy, and practicality: 

1. **Accuracy and Observations**:  
   The results from both methods are largely consistent, with minimal differences in categorization percentages. However, slight variations are observed in categories such as **"Baggage"** and  **"Others"** These discrepancies likely stem from the ambiguous nature of some interactions, where context might not always be straightforwardly interpreted, especially in iterative validation.

2. **Efficiency and Performance**:  
   - **Long-Context Method**: This approach is significantly faster as it processes the entire interaction dataset within a single pass, leveraging cached context to ensure consistency and reduce computational overhead.
   - **Iterative Validation**: This method, by contrast, requires separate categorization for each interaction, resulting in processing times that are 15–20 times longer. The absence of context caching further amplifies the computational and temporal costs.

3. **Cost Implications**:  
   Iterative validation is not only time-intensive but also incurs substantially higher computational costs. Each interaction is processed independently, leading to repeated computations that could otherwise be streamlined with context-aware approaches.

4. **Practical Implications**:  
   While iterative validation might occasionally provide marginally more tailored results for edge cases, the long-context method offers a balance of speed, cost-efficiency, and accuracy that is more suitable for large-scale analysis.

<strong>Conclusion:</strong>  
The long-context method proves to be a more practical and efficient solution for categorizing customer interactions at scale. While iterative validation has its strengths in certain edge cases, the increased costs and time requirements make it less viable for routine analysis.
</div>

### **4.7.3 Identifying Discrepancies Between Categorization Methods**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">

This section of the code analyzes the discrepancies between the categorization results obtained using the **Gemini Long Context** approach and the **Iterative Validation** method. The goal is to identify the categories with the most significant differences in categorization percentages.

1. **Calculate Discrepancies**:
   - A new column, **Discrepancies**, is created to measure the absolute difference between the **Gemini Long Context Percentage** and the **Validation Percentage** for each category. This helps highlight where the two methods yield differing results.
   
2. **Identify Top Discrepancies**:
   - The code sorts the discrepancies in descending order and extracts the two categories with the largest discrepancies. These are the categories where the methods disagree the most in terms of percentage.

<strong>Purpose:</strong>
This analysis helps identify categories where the categorization methods diverge significantly, which can inform further analysis or adjustments to the categorization process. It also provides insight into the consistency and reliability of the categorization methods.

</div>

```python
comparison_df['Discrepancies'] = np.abs(comparison_df['Gemini Long Context Percentage'] - comparison_df['Validation Percentage'])

number_1_discrepancy = comparison_df.sort_values('Discrepancies', ascending=False)['Category'].iloc[0]
number_2_discrepancy = comparison_df.sort_values('Discrepancies', ascending=False)['Category'].iloc[1]
```

### **4.7.4 Sample of Categories with the Most Discrepancies**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">

This code retrieves and displays a sample of customer interactions that belong to the category with the most significant discrepancies in categorization between the **Gemini Long Context** and **Iterative Validation** methods.

1. **Filter by Category with the Highest Discrepancy**:
   - The code filters the **df_validation** DataFrame to find all interactions that belong to the category with the largest discrepancy (identified earlier as **number_1_discrepancy**).
   
2. **Display Sample Interactions**:
   - It then selects the first 20 interactions from this filtered category for review. This allows for a closer look at the interactions that contributed to the categorization differences.

<strong>Purpose:</strong>
This step helps examine the specific interactions in the category with the greatest discrepancy, allowing for a deeper understanding of why the two methods differ. It provides actionable insights into the nature of these interactions and potential improvements to the categorization process.

</div>

```python
# Samples on categories with most discrepancies
df_validation.loc[df_validation.category == number_1_discrepancy].head(20)
```

### **4.7.5 Analysis of Sample's Categories**

<div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">

This analysis focuses on examining ambiguous interactions that belong to categories with the highest discrepancies, offering insights into why these interactions may overlap across multiple categories.

<ol>
    <li>
        <strong>Interaction: "Do TSA have any responsibility for our loss?"</strong><br>
        <strong>Analysis:</strong> This query concerns the <strong>responsibility of the TSA</strong> (Transportation Security Administration) for lost items, emphasizing <strong>policies and procedures</strong>. While it relates to <strong>Baggage</strong>, it also overlaps with <strong>Airline Policies & Procedures</strong> because it involves security protocols.
    </li>
    <li>
        <strong>Interaction: "I have booked 4 flights and have paid for 15kilo luggage. I usually take 3 cases (2 large with 25 kilos each and the other with 10), but not sure if this is allowed with this airline."</strong><br>
        <strong>Analysis:</strong> This query pertains to <strong>baggage allowances and fees</strong>. However, since it references <strong>airline policies</strong> regarding luggage handling, it overlaps with procedural questions. Primarily, it fits under <strong>Baggage</strong>.
    </li>
    <li>
        <strong>Interaction: "I have been hearing stories of TSA officers confiscating such solution in carry-on baggage. The reasoning is that hydrogen peroxide can potentially be used as an explosive. I am wondering what people's experiences are with this?"</strong><br>
        <strong>Analysis:</strong> While this query involves <strong>baggage policies</strong>, it strongly relates to <strong>airline policies</strong> for carrying restricted items, making it overlap with procedural concerns.
    </li>
    <li>
        <strong>Interaction: "We're flying Manchester to Luxor in October and learnt that the baggage allowance is a total 20kg including 5kg max carry-on. Any experience on how strictly this is applied at MAN or Luxor?"</strong><br>
        <strong>Analysis:</strong> This query discusses <strong>baggage allowance limits</strong> and enforcement, directly relating to <strong>airline policies</strong> regarding luggage handling.
    </li>
    <li>
        <strong>Interaction: "My father has a lightweight wheelchair which folds down and goes into a zipped bag. He would like to take it away with him this year. Can anyone tell the procedure for doing this? When do I take him out of it, who takes it when we get on the plane, and lastly does it come through on the belt with cases?"</strong><br>
        <strong>Analysis:</strong> This question highlights <strong>procedures</strong> for traveling with assistive devices, making it fall under <strong>Airline Policies</strong>.
    </li>
</ol>

</div>

```python
# Samples on categories with most discrepancies
df_validation.loc[df_validation.category == number_2_discrepancy].head(20)
```

# **5. Why Gemini’s Long Context is a Game Changer?**

With the growing complexity of customer interactions and large data volumes, the ability to process long-context information efficiently has become a game changer. **Gemini 1.5's** long-context capability allows for deeper insights and better decision-making by overcoming the limitations of traditional models.
#
### 1. **Cost Reduction**
Gemini’s ability to process long data in a single pass significantly reduces processing costs. Unlike traditional models that require multiple passes or chunking, **Gemini 1.5**:
- **Lowers operational costs**: Eliminates redundant processing.
- **Improves speed**: Analyzes large datasets faster, using fewer resorce#s

### 2. **Enhanced Token Analysis**
By handling more tokens at once, **Gemini 1.5** can fully understand and maintain context over longer texts, ensuring:
- **Complete context**: No critical information is lost in long documents.
- **Higher accuracy**: The model processes more details and relationships, leading to better redictions.

```python
categorization_response.usage_metadata
```

```python
detailed_breakdown_response.usage_metadata
```

<div style="border: 2px solid #000; padding: 10px; margin: 10px;">

### **Cost Comparison: With vs. Without Context Caching**

Assuming an analyst is accessing this data 8 times a month, each asking 20 question in an hour sitting. Assume that there are around 500k characters worth of interactions.

#### **Case 1: Without Context Caching**

- **Input Cost:**  
  `500,200 characters x 20 questions x ($0.0003125 / 1000) = $3.13`
  
- **Output Cost:**  
  `400 output characters x 20 prompts x ($0.00375 / 1000) = $0.03`
  
- **Total Cost per Session:**  
  `$3.13 (input) + $0.03 (output) = $3.16`
  
- **Total Monthly Cost (8 sessions):**  
  `$3.16 x 8 = $25.28`

---

#### **Case 2: With Context Caching**

- **Cache Creation Cost:**  
  `500,000 input characters x ($0.0003125 / 1000) = $0.16`
  
- **Cache Storage Cost:**  
  `500,000 total characters x 1 hour x ($0.001125 / 1000) = $0.56`
  
- **Input Using Cache Cost:**  
  `200 characters x 20 questions x ($0.0003125 / 1000) = $0.00125`

- **Cached Input Cost:**  
  `500,000 cached characters x 20 requests x ($0.00007825 / 1000) = $0.78`
  
- **Output Cost:**  
  `400 output characters x 20 prompts x ($0.00375 / 1000) = $0.03`
  
- **Total Cost per Session:**  
  `$0.16 (cache creation) + $0.56 (storage) + $0.00125 (input) + $0.78 (cached input) + $0.03 (output) = $1.59`
  
- **Total Monthly Cost (8 sessions):**  
  `$1.59 x 8 = $12.72`

---

### **Cost Savings with Context Caching**

By implementing context caching, the total cost is reduced by **50%**

---

# **6. Future Development of the Use Case: Expanding Gemini’s Capabilities**
To build on Gemini 1.5’s transformative potential, future developments can integrate multimodal capabilities—combining text, audio, and video—to provide deeper, more robust insights for customer support.
Here’s a vision for the evolution of this use case:
## **6.1 Multimodal Analysis for Comprehensive Insights**
***Development Focus***

Combine text, audio, and video data into a single analysis pipeline for a complete understanding of customer interactions.
Leverage Gemini’s large context window to process multimodal datasets simultaneously.

1. *Integrating Audio Analysis for Call Center Insights*: Enable Gemini to analyze voice recordings from call center interactions alongside text-based data like chat logs and emails and Incorporate automatic speech recognition (ASR) to transcribe calls with high accuracy and analyze tone, sentiment, and keywords.
2. *Enhancing Video Analysis for Visual and Contextual Understanding*: Allow Gemini to process video recordings of customer interactions, product demonstrations, or tutorials and analyze visual cues (e.g., customer body language) and contextualize data within broader complaint trends.

Gemini 1.5 revolutionizes customer support by addressing the challenges of scale, complexity, and fragmented data analysis. Its advanced capabilities enable faster resolutions, accurate insights, and proactive customer care, transforming operations and delivering measurable business value.

1. **Efficiency Boost:** Resolution times cut by **30–40%**, doubling agent capacity, **saving 1 Mn USD – 1.5 Mn USD annually** per team of 10 agents.
2. **Improved Accuracy:** Resolution accuracy **increases by 30–50%**, reducing callbacks
3. **Customer Retention:** Churn reduced by **15–20%, retaining 3 Mn USD – 5 Mn USD annually** in revenue for 1 Mn customers.
4. **Customer Satisfaction:** Scores **improve by 20–30%** with faster, more accurate, and personalized resolutions.
5. **Proactive Solutions:** Recurring complaints **drop by 25–30%**, reducing strain and enhancing scalability.

*Note: Those numbers are estimated based on specific industries i.e. Telecom in SEA Countries*

![jpeg-optimizer_Future Development_new 2.jpg](attachment:544dba76-e515-41bf-b52b-8d04295fb970.jpg)

**Credit: AI4Indonesia Team**
1. Arda Putra Ryandika
2. Gewinner Sinaga
3. Faisal Rasbihan
4. Aries Fitriawan

![Closing new.jpg](attachment:2f284cc1-7044-42db-9d71-ba9840886be9.jpg)

# **7. References**
1. Ferraro, C., Demsar, V., Sands, S., Restrepo, M., & Campbell, C. (2024). The paradoxes of generative AI-enabled customer service: A guide for managers. Business Horizons, 67(5), 549-559. https://doi.org/10.1016/j.bushor.2024.04.013.
2. Gemini Team, Georgiev, P., Lei, V. I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., Mariooryad, S., Ding, Y., Geng, X., Alcober, F., Frostig, R., Omernick, M., Walker, L., Paduraru, C., Sorokin, C., Tacchetti, A., Gaffney, C., Daruki, S., Sercinoglu, O., Gleicher, Z., Love, J., Voigtlaender, P., Jain, R., Surita, G., Mohamed, K., Blevins, R., Ahn, J., Zhu, T., Kawintiranon, K., Firat, O., Gu, Y., Zhang, Y., Rahtz, M., Faruqui, M., Clay, N., Gilmer, J. D. Co-Reyes, I. P. et al. (2024). Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. https://arxiv.org/abs/2403.05530
3. Buesing, E., Cheta, O., Gupta, V., Kline, P., Sood, R., & Weihmann, S. (2024). Generative AI in customer care: Early successes and challenges. McKinsey & Company. https://www.mckinsey.com/capabilities/operations/our-insights/gen-ai-in-customer-care-early-successes-and-challenges.
4. Ghimire, P., Kim, K., & Acharya, M. (2024). Opportunities and Challenges of Generative AI in Construction Industry: Focusing on Adoption of Text-Based Models. Buildings, 14(1), 220. https://doi.org/10.3390/buildings14010220
5. Marr, B. (2024, January 26). How generative AI is revolutionizing customer service. Forbes. https://www.forbes.com/sites/bernardmarr/2024/01/26/how-generative-ai-is-revolutionizing-customer-service/
6. Marr, B. (2024, April 4). GenAI can help companies do more with customer feedback. Harvard Business Review. https://hbr.org/2024/04/genai-can-help-companies-do-more-with-customer-feedback
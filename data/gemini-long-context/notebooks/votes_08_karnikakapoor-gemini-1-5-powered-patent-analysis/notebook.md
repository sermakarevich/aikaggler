# ✨Gemini 1.5 Powered Patent Analysis 📚

- **Author:** Karnika Kapoor
- **Votes:** 62
- **Ref:** karnikakapoor/gemini-1-5-powered-patent-analysis
- **URL:** https://www.kaggle.com/code/karnikakapoor/gemini-1-5-powered-patent-analysis
- **Last run:** 2024-12-01 09:26:06.243000

---

# <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:center; padding:20px; background: #81667a; border-radius: 25px; box-shadow: 10px 10px 5px #8c8a93;" >✨ Gemini 1.5 Powered Patent Analysis 📚<br><div>  


**Gemini 1.5 Powered Patent Analysis: Exploring Novelty and Overlap 📚**

Patent analysis serves as the backbone of technological innovation, safeguarding intellectual property while driving advancements across industries. Yet, the sheer volume and complexity of patent data make this task a formidable challenge. In this notebook, we demonstrate how **Gemini 1.5** redefines patent analysis, leveraging its advanced AI capability to simplify workflows and uncover critical insights.


### **Project Goals** 🎯

This project explores Gemini 1.5’s potential in tackling key objectives:

1. **Uncover Overlaps**: Identify similarities in patent claims, descriptions, and ideas to refine innovation and reduce redundancy.
2. **Evaluate Novelty**: Highlight unique contributions and differentiate innovative features from existing prior art.
3. **Analyze Priority Dates**: Examine the significance of filing dates in establishing the originality and relevance of patents.

These goals align with the broader challenge of optimizing patent workflows while showcasing the power of AI-driven insights.


### **Challenges of Patent Analysis** 🧐

Patent analysis is a complex endeavor due to several inherent challenges:

- **🚧 Volume**: Patent documents are extensive and data-rich, making manual review inefficient.  
- **🧩 Complexity**: Technical and legal language introduces additional layers of difficulty in identifying overlaps and evaluating novelty.  
- **⏳ Time**: Traditional methods often require significant time investments and are prone to errors, limiting scalability.  

The process typically involves comparing the claims, abstracts, and descriptions of patents to identify similarities or overlapping ideas. Analysts must determine whether a patent introduces a novel feature or is merely an extension of prior inventions. Evaluating priority dates is critical in establishing originality, as earlier filings can often nullify claims of novelty in subsequent patents. This labor-intensive process requires cross-referencing multiple patents, often using databases and manual methods prone to inconsistencies.


### **How Gemini 1.5 Addresses These Challenges** 🤖

Gemini 1.5 provides an innovative solution to these issues by combining its **long-context processing capabilities** with **context caching**, enabling efficient analysis of large datasets. Notable features include:

- **Large Context Window**: Processes inputs **exceeding 100,000 tokens** (and up to 2 million tokens), eliminating the need for data chunking or reliance on external tools like vector databases.  
- **Context Caching**: Facilitates incremental and efficient workflows by retaining key details across multiple analyses.  
- **Precision Insights**: Enables accurate identification of overlaps, evaluation of novelty, and actionable recommendations for claim refinement.  
- **Streamlined Workflows**: Handles diverse sections of patents, such as Title, Abstract, Claims, and Description, within a unified framework.  

By automating and enhancing these traditionally manual tasks, Gemini 1.5 redefines patent analysis, transforming it into a scalable, efficient, and reliable process.


## 🎥 Accompanying Video
For a quick walkthrough of this analysis, check out th **[Video](https://youtu.be/YpmvsByIICc)**.


# <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:center; padding:10px; background: #81667a; border-radius: 25px; box-shadow: 10px 10px 5px #8c8a93;">📑 TABLE OF CONTENTS<br><div>  

- [1. 📋 Introduction](#1)
- [2. 🔍 Long Context and Caching for Patent Insights](#2)
- [3. 🗂️ Dataset Overview](#3)
- [4. 🛠️ Implementation Workflow](#4)
    - [4.1 🔮 Importing Libraries and Setting Up Gemini API](#4.1)
    - [4.2 📂 Data Preparation](#4.2)
    - [4.3 🧮 Calculating Similarity](#4.3)
    - [4.4 🔗 Preparing and Sending Prompts for Gemini 1.5](#4.4)
    - [4.5 ✨ Patent Analysis and 📊 Token Usage](#4.5)
- [5. 📊 Results and Insights](#5)
    - [5.1 ⚡ Case 1](#5.1)
    - [5.2 ⚡ Case 2](#5.2)
    - [5.3 ⚡ Case 3](#5.3)
- [6. 🌟 Leveraging Gemini 1.5 for Enhanced Patent Analysis](#6)
- [7. ✅ Conclusion](#7)
- [8. 📎 Resources](#8)

<div style="font-family: Arial, sans-serif; font-size: 120%; color: Black; text-align: center; margin-top: 20px;">
  📚 Citation: This post is written by Karnika Kapoor and is edited with the help of ChatGPT.
</div>

<a id="1"></a>

# <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:center; padding:10px; background: #81667a; border-radius: 25px; box-shadow: 10px 10px 5px #8c8a93;" >📋 Introduction <br><div>  

Patents are the building blocks of innovation, offering inventors legal protection for their groundbreaking ideas while fostering progress in science and technology. Analyzing patents, however, is no small feat—it involves meticulously navigating through:

- **Abstracts**: High-level summaries of the invention.
- **Claims**: Legal definitions of the invention's scope and boundaries.
- **Descriptions**: Comprehensive details on the technical implementation.

This task is daunting due to the **high volume** of data and the **technical complexity** inherent in patent documents. Traditional methods of manual analysis are time-consuming, error-prone, and unsuited to the pace of modern innovation.

Enter **Gemini 1.5**, a state-of-the-art AI model designed to overcome these limitations with its unmatched **long-context processing capabilities** and **context caching**. This notebook showcases how Gemini 1.5 revolutionizes patent analysis by:

- **Streamlining Comparisons**: Effortlessly identifying overlaps and evaluating novelty across multiple patents.
- **Accelerating Analysis**: Utilizing AI-driven automation to handle large datasets quickly and accurately.
- **Eliminating Constraints**: Processing entire patents in one pass, avoiding truncation, chunking, or reliance on external tools.

By integrating Gemini 1.5 into the workflow, we aim to demonstrate how AI can enable faster, more precise patent insights, ultimately empowering inventors, researchers, and businesses to innovate with confidence.

<a id="2"></a>
# <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:center; padding:10px; background: #81667a; border-radius: 25px; box-shadow: 10px 10px 5px #8c8a93;" >🔍 Long Context and Caching for Patent Insights <br><div>  

**The Power of Long Context Windows**

In artificial intelligence, the **context window** refers to the amount of information a model can process in a single operation. It is measured in **tokens**, which are the basic units of text that AI models interpret. A larger context window allows the model to understand broader narratives, retain detailed information, and produce more accurate results, especially for complex tasks.

**Gemini 1.5 Flash** stands out with its exceptional long-context capacity, making it a ideal in analyzing large-scale datasets like patents. This capability is particularly beneficial for workflows that involve extensive textual content.

### **The Significance of Tokens and Tokenization**  
Tokens are the building blocks of text for AI models. Depending on the tokenization method, a word can be represented as a single token or divided into smaller components. For example:
- The word **innovation** could be tokenized as a single token or split into parts like **inno** and **vation** based on its etymological root.

This process is fundamental to how AI models interpret and generate language. 
<img src="https://github.com/KarnikaKapoor/Files/blob/main/Blue%20Red%20and%20Orange%20Gridded%20English%20Quiz%20Presentation%20(1).png?raw=true">

**Tokenization** is essential as it defines how text is divided for AI processing, directly impacting the model's ability to interpret and generate language effectively.

### **Gemini 1.5’s Long Context Window in Patent Analysis**  
The extended context window of **2 million tokens** equips Gemini 1.5 with unmatched capabilities. It eliminates the need for chunking and streamlines workflows by processing entire documents in one pass. Its potential in patent analysis includes:

1. **Comprehensive Understanding**: Gemini 1.5 processes entire patent documents (claims, descriptions, prior art) at once, leaving no crucial detail behind.  
2. **Enhanced Accuracy**: With a holistic view, the model identifies overlaps and evaluates novelty more precisely.  
3. **Novelty Detection**: It assesses inventions against vast patent datasets, ensuring timely and accurate innovation insights.  
4. **Efficient Classification**: Patents can be sorted into relevant categories for seamless retrieval and analysis.  

### **Comparative Study: Context Window Sizes in AI Models**

The ability to process extensive context is a defining feature of advanced AI models, directly influencing their performance in tasks requiring comprehensive understanding. Below is a comparative overview of context window sizes across various prominent models:

| **Model**             | **Context Window Size** | **Description**                                                                                     |
|-----------------------|-------------------------|-----------------------------------------------------------------------------------------------------|
| **GPT-3.5**           | 4,096 tokens            | Early model with limited context processing capabilities.                                           |
| **GPT-4**             | 8,192 tokens            | Improved context handling, enabling more coherent responses.                                        |
| **GPT-4 Turbo**       | 128,000 tokens          | Significantly expanded context window, allowing for processing of larger documents.                 |
| **Claude 3.5 Sonnet** | 200,000 tokens          | Designed for extensive context understanding, suitable for complex tasks.                           |
| **Llama 3**           | 128,000 tokens          | Advanced model with enhanced context processing, supporting large-scale analyses.                   |
| **Gemini 1.5**  | 2,000,000 tokens        | Exceptional long-context capability, enabling seamless analysis of extensive datasets.              |

*Note: Token counts are approximate and represent the maximum context window size each model can handle.*


*Illustration: Context Window Comparison*
<img src="https://github.com/KarnikaKapoor/Files/blob/main/Blue%20Red%20and%20Orange%20Gridded%20English%20Quiz%20Presentation.png?raw=true">


### **Why Context Size Matters in Patent Analysis**  
A model’s ability to retain extended contexts is critical for nuanced tasks like patent workflows, where:

- **Longer Contexts** mean more comprehensive data processing in a single iteration.  
- **Reduced Redundancy** minimizes errors caused by segmenting or truncating information.  

### **Introducing Context Caching**

Gemini 1.5’s **context caching** feature optimizes patent analysis by storing and reusing previously processed information. This reduces computational overhead and enhances efficiency.

- **How It Works**: When analyzing documents with repetitive sections or shared contexts, cached data is reused in subsequent analyses, avoiding redundant processing.  
- **Benefits**: Enhances efficiency by speeding up repeated analyses and reducing computational resource costs. 
    

This diagram showcases how context caching works in conjunction with the LLM, reducing redundant processing and enhancing efficiency. Read more [here](https://medium.com/google-cloud/vertex-ai-context-caching-with-gemini-189117418b67)

<img src="https://github.com/KarnikaKapoor/Files/blob/main/Blue%20Red%20and%20Orange%20Gridded%20English%20Quiz%20Presentation%20(2).png?raw=true">


This project integrates **context caching**, enabling Gemini 1.5 to deliver faster and more effective analyses while allowing users to ask follow-up questions seamlessly.

By Using the capabilities of Gemini 1.5’s long-context processing and context caching, we demonstrate a scalable and efficient patent analysis workflows. These advancements pave the way for transformative innovation across document-intensive domains, including legal research, scientific exploration, and education.

<a id="3"></a>

# <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:center; padding:10px; background: #81667a; border-radius: 25px; box-shadow: 10px 10px 5px #8c8a93;" > 📁 Dataset Overview <br><div>  


Patents provide a wealth of innovation, requiring detailed, structured datasets for effective analysis. This project utilizes a dataset from **Google Patents**, focusing on machine learning and healthcare technologies to showcase **Gemini 1.5**'s ability to handle complex, high-context data.

### 📊 Key Dataset Attributes

- **Patent ID**: Unique identifier for each patent.  
- **Title**: Concise summary of the innovation.  
- **Assignee**: Holder of the patent rights.  
- **Priority Date**: Earliest filing date, critical for novelty evaluation.  
- **Publication Date**: Official publication date, essential for timeline analysis.  
- **Abstract**: High-level summary of the invention.  
- **Claims**: Legally defined scope of protection.  
- **Description**: In-depth technical details of the invention.

### Dataset Relevance

The dataset forms the backbone of this project by enabling:

- **Similarity Detection**: Identifying overlaps in ideas and claims.  
- **Novelty Evaluation**: Assessing unique contributions.  
- **Timeline Analysis**: Understanding the significance of priority and publication dates.  

### Why This Dataset?

- **Comprehensive Fields**: Includes claims, descriptions, and abstracts, ensuring Gemini 1.5 can process full patents seamlessly.  
- **Domain-Specific Focus**: Highlights trends and overlaps in cutting-edge machine learning and healthcare innovations.  
- **Scalability**: Structured to align with Gemini’s ability to process up to 2 million tokens, enabling efficient handling of large patent families.

This dataset is integral to demonstrating **Gemini 1.5**'s ability to construct detailed prompts, process long-context inputs efficiently, and generate actionable insights into innovation and novelty.

[📄 View the Dataset](https://www.kaggle.com/datasets/karnikakapoor/ml-in-healthcare-patent-data)

<a id="4"></a>
# <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:center; padding:10px; background: #81667a; border-radius: 25px; box-shadow: 10px 10px 5px #8c8a93;" >🛠️ Implementation Workflow <br><div>  

 

This section provides an overview of the structured steps involved in our patent analysis pipeline. The pipeline leverages the advanced capabilities of **Gemini 1.5 Flash**, including **long-context processing** and **context caching**, to extract actionable insights from large-scale patent datasets. Notably, **context caching** is utilized to enable efficient follow-up questions, allowing for deeper, iterative analyses based on previously processed data.

### **Workflow Overview**  
The pipeline consists of the following key phases:  
- **Data Preparation**: Cleaning and preprocessing patent data to ensure high-quality, consistent inputs for analysis.  
- **Similarity Calculations**: Using TF-IDF vectorization and cosine similarity to identify overlapping ideas and cluster patents based on semantic similarities.  
- **Prompt Construction and Optimization**: Creating structured prompts that integrate key patent details and metadata while adhering to token limits of free tier Gemini 1.5 Flash. 
- **Gemini 1.5 Analysis**: Harnessing long-context processing to analyze overlaps, evaluate novelty, and deliver actionable insights.  
- **Context Caching for Follow-Up Questions**: Reusing cached contexts to answer follow-up questions with reduced computational overhead and faster response times.

### **Why Context Caching Matters**  
The integration of **context caching** enhances efficiency by:  
- Allowing users to seamlessly build on prior analyses without reprocessing identical inputs.  
- Supporting iterative workflows where follow-up questions can be quickly addressed using stored data.  
- Reducing redundant computations, making the pipeline scalable and cost-effective.  

The detailed implementation of these phases is covered in the following subsections:  
- **4.1 ⚙ Importing Libraries and Setting Up Gemini API**: Initializing the environment and configuring the API.  
- **4.2 📂 Data Preparation**: Cleaning and preparing the dataset for downstream tasks.  
- **4.3 🧮 Calculating Similarity**: Identifying related patents using semantic similarity metrics.  
- **4.4 🔗 Preparing and Sending Prompts for Gemini 1.5**: Crafting and submitting structured prompts for in-depth analysis.  
- **4.5 ✨ Patent Analysis and 📊 Token Usage**: Demonstrating the pipeline's outputs, including follow-up questions using cached contexts.  

With **context caching** as a cornerstone, this pipeline sets a new standard for efficiency and scalability in patent analysis. Let’s explore the implementation details in the subsections below! 🎯

<a id="4.1"></a>

## **4.1 ⚙ Importing Libraries and Setting Up Gemini API**


Let start by getting our set up ready by ensuring all necessary libraries are imported and the **Gemini API** is properly configured. This step is crucial for utilizing Gemini 1.5 Flash’s long context capabilities for our patent analysis.

### **Importing Libraries**
We begin by importing the libraries required for data handling, similarity calculations, and Gemini API integration. Each library plays a specific role:

```python
# Import Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import google.generativeai as genai
from google.generativeai import caching
from tqdm.notebook import tqdm
import re
import time
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import networkx as nx
import matplotlib.patches as mpatches
import datetime

#Setting up colors prefrences
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
palette = ['#d1f0b1', '#b6cb9e', '#92b4a7', '#8c8a93', '#81667a']
cmap = colors.ListedColormap(['#d1f0b1', '#b6cb9e', '#92b4a7', '#8c8a93', '#81667a'])
```

### **Setting Up the Gemini API**

To use **Gemini 1.5**, you’ll need to configure the API by obtaining an API key from [Google AI Studio](https://ai.google.dev/). For detailed instructions on generating and securely attaching your API key, refer to the following resources:

* [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
* [How to Upload Large Files to Gemini 1.5](https://www.kaggle.com/code/paultimothymooney/how-to-upload-large-files-to-gemini-1-5)

  
These guides provide clear and practical steps to help you configure Gemini 1.5 effectively.

```python
# Gemini API
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("my_key")
genai.configure(api_key=GEMINI_API_KEY)
```

**💡 Why Gemini 1.5 Flash?**

Among the various Gemini models available, we chose Gemini 1.5 Flash for its unmatched combination of speed, flexibility, and efficiency. Its long-context window, supporting up to 1 million tokens per request, makes it particularly suited for analyzing lengthy patents without the need for splitting or truncating, unlike some other models.  

The free tier provides generous limits—15 requests per minute, 1 million tokens per minute, and 1,500 requests per day—making it an excellent option for testing and scaling. Additionally, its context caching feature, which stores up to 1 million tokens per hour, ensures faster repeated analyses and smoother workflows.  

While other Gemini models, like Pro, offer unique capabilities, Gemini 1.5 Flash is the most cost-effective and practical choice for this notebook.


<a id="4.2"></a>

## 4.2 📂 Data Preparation

In this section, we load, clean, and preprocess the patent dataset to ensure high-quality input for analysis. The process involves handling missing values, filtering irrelevant or erroneous entries, and standardizing text fields for consistency.

### **Steps in Data Loading and Cleaning**

1. **📥 Loading the Dataset**  
   The dataset is loaded from the provided file path using `pandas`. The initial dataset size is logged to track the impact of cleaning and filtering.

2. **🧹 Text Cleaning**  
   Text fields (`Title`, `Abstract`, `Claims`, `Description`) are cleaned to ensure uniformity:
   - Special characters and multiple spaces are removed.
   - Text is converted to lowercase and stripped of extra spaces.
   - A custom `clean_text` function is applied to process each relevant column.

3. **🚦 Filtering Entries**  
   Invalid or incomplete records are filtered out:
   - Removed entries with "abstract not found" or error messages in `Abstract` or `Claims`.
   - Excluded rows with null values in critical fields like `Abstract` or `Claims`.

4. **✨ Preprocessing Output**  
   - Dropped unnecessary columns, such as `Result Link`, for a cleaner dataset.
   - Logged the dataset size after all preprocessing steps to ensure proper data handling.

```python
# Function to clean text fields
def clean_text(text):
    """
    Cleans individual text fields by removing special characters, multiple spaces,
    and converting text to lowercase.
    """
    if pd.isnull(text):  
        return None
    text = re.sub(r'\s+', ' ', text)   
    text = re.sub(r'[^\w\s]', '', text)   
    return text.lower().strip()

# Function to load and clean the dataset
def load_data(file_path):

    print("Loading dataset...")
    data = pd.read_csv(file_path)  
    print(f"Initial dataset size: {data.shape}")

    # Clean text 
    for column in ['Title', 'Abstract', 'Claims', 'Description']:
        if column in data.columns:
            print(f"Cleaning '{column}' column...")
            data[column] = data[column].apply(clean_text)

    # Filter entries
    data = data[
        (data['Abstract'].str.lower() != "abstract not found") & 
        (~data['Abstract'].isnull()) &
        (~data['Claims'].isnull()) &
        (~data['Abstract'].str.contains("error", case=False, na=False)) &
        (~data['Claims'].str.contains("error", case=False, na=False))
    ]
    data = data.drop(["Result Link"], axis = 1)
    print(f"Dataset size after preprocessing: {data.shape}")
    return data

# File path for our dataset
file_path = '/kaggle/input/ml-in-healthcare-patent-data/patent_analysis_data.csv'

# Load and preprocess the dataset
data = load_data(file_path)
```

#### **Dataset Overview**
After preprocessing, the dataset retains only high-quality, relevant entries for analysis. Here's an overview:
 
- **Initial Size**: _(23,339, 9)_
- **Final Size**: _(18,109, 8)_

The cleaned dataset now provides a solid foundation for downstream tasks like similarity calculations and prompt construction.

💡 Display a preview of the cleaned data for additional context:

```python
data.head()
```

🧽 This concludes the data loading and cleaning phase. Next, we’ll dive into **similarity calculations** to uncover overlaps and patterns in the dataset.

<a id="4.3
"></a>
## 4.3 🧮 Calculating Similarity

In this step, we identify patents with significant overlaps by calculating pairwise similarities using the following techniques:

### **Techniques Used**

1. **TF-IDF Vectorization**: Converts text fields into numerical vectors based on term frequency and inverse document frequency, capturing the importance of terms within each patent.

2. **Cosine Similarity**: Measures the cosine of the angle between two vectors to quantify their similarity, effectively identifying semantic overlap between patents.

3. **Threshold**: A similarity score of **0.9 or higher** is used to define overlaps. This high threshold ensures that only patents with substantial semantic similarity are identified, focusing the analysis on meaningful relationships. The threshold can be adjusted based on sp
   ecific use cases:
   - **Higher Thresholds (e.g., 0.95)**: Detect near-duplicate patents with almost identical content.
   - **Lower Thresholds (e.g., 0.7)**: Broaden the scope to include patents with conceptual or thematic similarities.

### **Function to Find Similar Patents**
The following function identifies patents with multiple overlapping matches based on the you’d like further tweaks!

```python
def find_similar_patents(data, threshold=0.9):
    
    print("Analyzing patents for multi-source similarities...")

    # Combine relevant text fields
    combined_text = data['Title'] + " " + data['Abstract'] + " " + data['Claims'] + " " + data['Description']

    # Convert text data into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Extract patents with multiple high-similarity matches
    multi_similar = []
    patent_ids = data['Patent ID'].values
    titles = data['Title'].values

    for idx in range(similarity_matrix.shape[0]):
        # Find patents with similarity above the threshold
        similar_indices = np.where(similarity_matrix[idx] >= threshold)[0]
        similar_indices = similar_indices[similar_indices != idx]

        if len(similar_indices) >= 2:  # At least two similar patents 
            multi_similar.append({"Patent ID": patent_ids[idx],"Title": titles[idx],"Similar Patents": [patent_ids[j] for j in similar_indices],"Similarity Scores": [similarity_matrix[idx, j] for j in similar_indices]})

    print(f"Identified {len(multi_similar)} patents with multi-source similarities.")
    return multi_similar
```

### **Running Similarity Analysis**

To identify patents with multiple overlapping matches, we run the function on the cleaned dataset:

```python
# Runing the similarity analysis
similar_patents = find_similar_patents(data)
```

### **Results: Multi-Source Similarities**

The following snippet displays the first few clusters of patents with significant overlaps:

```python
# Display the first five clusters
for cluster in similar_patents[:5]:
    print(f"Patent ID: {cluster['Patent ID']}")
    print(f"Title: {cluster['Title']}")
    print(f"Similar Patents: {cluster['Similar Patents']}")
    print(f"Similarity Scores: {cluster['Similarity Scores']}\n")
```

#### 1️⃣ **Patent Similarity Network Visualization**
The network visualization provides an intuitive overview of relationships between patents based on their similarity:

- **Main Patents**: Represented as **light pink nodes**, acting as central connections.
- **Similar Patents**: Represented as **darker green nodes**, linked to the main patents by edges.
- **Edges**: Represent similarity scores above the threshold (**0.9**), forming clusters of closely related patents.

This graph highlights clusters of patents with strong semantic similarities, helping identify:
- **Highly Connected Patents**: Suggesting potential overlaps or redundancies.
- **Isolated Nodes**: Indicating patents with unique or less significant overlaps.ctionable insights into innovation. 🚀

```python
def visualize_patent_network_without_scale(similar_patents, threshold=0.9, figsize=(12,8)):
    # Custom colors palette
    main_color = '#FFC1CC'
    main_edge_color = '#b6cb9e'
    similar_color = '#92b4a7'
    similar_edge_color = '#81667a'
    edge_color = '#8c8a93'
    
    # Create a graph
    G = nx.Graph()
    
    # Process edges and create edge list with weights
    edge_list = []
    for pair in similar_patents:
        main_patent = pair["Patent ID"]
        for similar_patent, score in zip(pair["Similar Patents"], pair["Similarity Scores"]):
            if score > threshold:
                edge_list.append((main_patent, similar_patent, score))
    
    # Edges and nodes
    main_patents = set()
    similar_patents = set()
    
    for main, similar, weight in edge_list:
        G.add_edge(main, similar, weight=weight)
        main_patents.add(main)
        similar_patents.add(similar)
    
    # Create figure with a specific layout
    fig, ax_main = plt.subplots(figsize=figsize, facecolor='#FFF9ED', dpi=100)
    ax_main.set_facecolor('#FFF9ED')
    
    # Calculate node positions
    pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=50, seed=42)
    
    # Draw edges with transparency based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Normalize weights
    min_weight = min(weights)
    max_weight = max(weights)
    norm_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
    
    # Draw edges
    for (u, v), weight in zip(edges, norm_weights):
        ax_main.plot([pos[u][0], pos[v][0]], 
                     [pos[u][1], pos[v][1]], 
                     '-', 
                     color=edge_color,
                     alpha=0.1 + weight * 0.2,
                     linewidth=0.5 + weight * 1.5)
    
    # Draw main patent nodes
    nx.draw_networkx_nodes(G, pos,nodelist=list(main_patents),node_color=main_color,node_size=300,alpha=0.8,edgecolors=main_edge_color,linewidths=1.5,ax=ax_main)
    
    # Draw similar patent nodes
    nx.draw_networkx_nodes(G, pos,nodelist=list(similar_patents), node_color=similar_color, node_size=200, alpha=0.5, edgecolors=similar_edge_color, linewidths=1.5,ax=ax_main)
    
    # Legend patches 
    main_patch = mpatches.Patch(facecolor=main_color, edgecolor=main_edge_color, alpha=0.8, label=f'Main Patents ({len(main_patents)})')
    
    similar_patch = mpatches.Patch(facecolor=similar_color, edgecolor=similar_edge_color, alpha=0.8, label=f'Similar Patents ({len(similar_patents)})')
    
    # Add legend
    ax_main.legend(handles=[main_patch, similar_patch], loc='upper right', fontsize=12, framealpha=0.9, facecolor='white', edgecolor='#dcdde1')
    
    # Add title
    ax_main.set_title(f'Patent Similarity Network\n'
                      f'Similarity threshold: {threshold}\n'
                      f'Total Connections: {len(edge_list)}\n'
                      f'Average Similarity: {np.mean([x[2] for x in edge_list]):.3f}',
                      fontsize=14,
                      pad=20)
    
    # Remove axes
    #ax_main.axis('off')

    return G

# Call the function
G = visualize_patent_network_without_scale(similar_patents, threshold=0.9)
```

#### 2️⃣ **Distribution of Similarity Scores Across Clusters**
The histogram complements the network visualization by showing how similarity scores are distributed across clusters:

- **Threshold (0.9)**: A dashed line marks the threshold, highlighting significant overlaps.
- **Distribution Patterns**:
  - High frequencies near **1.0** indicate near-duplicates or highly redundant patents.
  - Lower frequencies in the **0.9–0.95 range** suggest meaningful thematic overlaps.
nsights into innovation. 🚀

```python
# The distribution of Similarity Scores

all_scores_filtered = [score for cluster in similar_patents for score in cluster["Similarity Scores"]]

plt.figure(figsize=(12, 6))
plt.hist(all_scores_filtered, bins=20, color=palette[2], edgecolor=palette[3], alpha=0.8) 
plt.title("Distribution of Similarity Scores Across Clusters", fontsize=14, pad=10)
plt.xlabel("Similarity Score", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.axvline(0.9, color=palette[-1], linestyle="dashed", linewidth=2, label="Threshold (0.9)")  

# Legend and grid
plt.legend(fontsize=12, frameon=True, facecolor="white", edgecolor=palette[3], loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.6, color=palette[4])
plt.show()
```

#### Key Insights:
- **Redundancy and Uniqueness**: Combined visualizations identify redundant patents and unique innovations.
- **Cluster Highlights**: Network clusters and histogram peaks reveal areas for deeper analysis.
- **Threshold Validation**: Both visualizations reinforce the effectiveness of the **0.9 threshold** in focusing on significant overlaps.




With the relationships and overlaps visualized, the next step is leveraging **Gemini 1.5** for detailed analysis. We’ll focus on interpreting patterns, identifying novel contributions, and assessing actionable insights across the dataset.
 further refinements!

<a id="4.4"></a>

## **4.4 🔗 Preparing and Sending Prompts for Gemini 1.5**

In this section, we prepare, send, and process structured prompts to analyze patents using **Gemini 1.5 Flash**. The workflow includes combining relevant patent fields, constructing prompts, submitting them to the model, and recording results and token usage.


#### **What Does This Section Cover?**

1. **Preparing Prompts**: Combining patent fields and metadata to create structured inputs for analysis.  
2. **Sending Prompts**: Submitting the prompts to **Gemini 1.5 Flash** and handling API constraints dynamically.  
3. **Processing Results**: Recording analysis outputs and token usage statistics while tracking skipped patents.


#### **Why Construct Prompts?**

The quality of the analysis heavily depends on the clarity and detail of the input. **Gemini 1.5 Flash** relies on structured prompts to understand the context and perform specific tasks. This workflow ensures:
1. **Comprehensive Inputs**: Combining fields like `Title`, `Abstract`, `Claims`, and `Description` gives the model a holistic view of the patents.  
2. **Contextual Metadata**: Including metadata such as `Priority Dates` and `Similarity Scores` helps the model contextualize the analysis.  
3. **Actionable Outputs**: Structured tasks, such as evaluating overlaps and novelty, guide the model toward producing interpretable results.  
4. **Optimized Token Usage**: By dynamically limiting the number of similar patents and calculating word counts, this approach leverages Gemini’s long-context capabilities effectively.  


#### **Workflow**

1. **Select Main and Similar Patents**:  
   - Identify a main patent and up to a specified number of similar patents based on similarity scores.  
   - Combine their full texts and metadata for a comprehensive prompt.  

2. **Construct and Format Prompts**:  
   - Use a structured template that specifies the analysis tasks for the model.  
   - Calculate the total word count to ensure the prompt fits within the model's token limit.  

3. **Analyze Using Gemini 1.5 Flash**:  
   - Submit the prompt to the model.  
   - Record token usage statistics and analysis results for further evaluation.  


#### **`compare_patent_groups(data, similar_patents, prompt_template, ...)`**
This function analyzes patents with efficient token management. It:
- Combines main patent data and top similar patents into a structured prompt.
- Skips patents exceeding the specified word count limit to manage resources.
- Tracks total token usage and pauses for 2 minutes when approaching the free-tier limit.
- Sends prompts to Gemini 1.5 Flash and aggregates results and token usage.
- Returns two DataFrames: analysis results and token usage statistics.

```python
def compare_patent_groups(data, similar_patents, prompt_template, max_compare=3, word_limit=100000, max_success=10):
    """
    Compare a main patent with similar patents and perform analysis, with token usage pause.

    Parameters:
    - data (pd.DataFrame): DataFrame containing patent data.
    - similar_patents (list): List of patents with multi-source similarities.
    - prompt_template (str): Template for constructing the analysis prompt.
    - max_compare (int): Maximum number of similar patents to include.
    - word_limit (int): Word limit for skipping analysis.
    - max_success (int): Maximum number of successful analyses.

    Returns:
    - pd.DataFrame: Results DataFrame summarizing analysis.
    - pd.DataFrame: Token usage DataFrame for token statistics.
    """
    results = []
    token_usage_records = []
    success_count = 0  # Count of successful analyses
    total_tokens_processed = 0  # Track total tokens processed

    # Ensure 'Patent ID' is used as the index
    if 'Patent ID' in data.columns:
        data = data.set_index('Patent ID')

    for patent in similar_patents:
        if success_count >= max_success:
            break

        # Check if we need to pause due to token usage
        if total_tokens_processed > 95000:
            print(f"Token usage ({total_tokens_processed}) exceeded 95,000. Pausing for 2 minutes...")
            time.sleep(120)  # 2-minute pause
            total_tokens_processed = 0  # Reset token count after pause

        main_id = patent.get('Patent ID')
        if not main_id or main_id not in data.index:
            print(f"Patent {main_id} is missing or not found. Skipping.")
            results.append({
                "Main Patent": main_id,
                "Similar Patents": [],
                "Status": "Skipped",
                "Reason": "Patent missing or not found."
            })
            continue

        # [Rest of the existing function remains the same]
        # ... [previous implementation]

        main_id = patent.get('Patent ID')
        if not main_id or main_id not in data.index:
            print(f"Patent {main_id} is missing or not found. Skipping.")
            results.append({
                "Main Patent": main_id,
                "Similar Patents": [],
                "Status": "Skipped",
                "Reason": "Patent missing or not found."
            })
            continue

        # Combine full text for the main patent
        main_text = " ".join([
            str(data.loc[main_id, col]) if col in data.columns and pd.notnull(data.loc[main_id, col]) else ""
            for col in ['Title', 'Abstract', 'Claims', 'Description']
        ])

        # Sort similar patents by similarity score and limit by max_compare
        sorted_similar = sorted(
            zip(patent['Similar Patents'], patent['Similarity Scores']),
            key=lambda x: x[1],
            reverse=True
        )[:max_compare]

        similar_ids = [p[0] for p in sorted_similar]
        similar_scores = [p[1] for p in sorted_similar]

        # Combine full text for similar patents
        similar_details = "\n".join([
            f"**Similar Patent {i + 1}**:\n"
            f"- ID: {similar_ids[i]}\n"
            f"- Title: {data.loc[similar_ids[i], 'Title']}\n"
            f"- Priority Date: {data.loc[similar_ids[i], 'Priority Date']}\n"
            f"- Publication Date: {data.loc[similar_ids[i], 'Publication Date']}\n"
            f"- Full Text: {' '.join([str(data.loc[similar_ids[i], col]) if col in data.columns and pd.notnull(data.loc[similar_ids[i], col]) else '' for col in ['Title', 'Abstract', 'Claims', 'Description']])}"
            for i in range(len(similar_ids))
        ])

        # Estimate word count for combined text
        total_word_count = len(main_text.split()) + sum(
            len(" ".join([
                str(data.loc[sim_id, col]) if col in data.columns and pd.notnull(data.loc[sim_id, col]) else ""
                for col in ['Title', 'Abstract', 'Claims', 'Description']
            ]).split()) for sim_id in similar_ids
        )

        # Skip if word count exceeds the limit
        if total_word_count > word_limit:
            print(f"Skipping {main_id} due to word limit ({total_word_count} > {word_limit}).")
            results.append({
                "Main Patent": main_id,
                "Similar Patents": similar_ids,
                "Status": "Skipped",
                "Reason": f"Word limit exceeded ({total_word_count} > {word_limit})."
            })
            continue

        # Construct the prompt
        prompt = prompt_template.format(
            main_patent_id=main_id,
            main_title=data.loc[main_id, 'Title'],
            main_priority_date=data.loc[main_id, 'Priority Date'],
            main_publication_date=data.loc[main_id, 'Publication Date'],
            main_full_text=main_text,
            similar_patents=similar_details,
            similarity_scores=", ".join(f"{score:.2f}" for score in similar_scores)
        )

        
        # Analyze the group using the model
        result = process_single_prompt({"Patent ID": main_id}, prompt)

        # Update total tokens processed
        if result.get("token_usage"):
            total_tokens_processed += result["token_usage"].get("Total Tokens", 0)

        # Save token usage and result
        token_data = result.get("token_usage", {})
        token_usage_records.append({
            "Main Patent": main_id,
            "Prompt Tokens": token_data.get("Prompt Tokens", "N/A"),
            "Output Tokens": token_data.get("Output Tokens", "N/A"),
            "Total Tokens": token_data.get("Total Tokens", "N/A")
        })
  
        results.append({
            "Main Patent": main_id,
            "Similar Patents": similar_ids,
            "Similarity Scores": similar_scores,
            "Status": "Analyzed" if result.get("success") else "Failed",
            "Analysis": result.get("analysis", "N/A"),
            "Token Usage": token_data,
        })

        if result.get("success"):
            success_count += 1

    # Create DataFrames for results and token usage
    results_df = pd.DataFrame(results)
    token_usage_df = pd.DataFrame(token_usage_records)

    return results_df, token_usage_df
```

```python
# Function to calculate word count in the input text 
# This is to put a check in place to quantify the input being sent for optimization purpose. 

def calculate_word_count(text):
    if not text:
        return 0
    return len(text.split())
```

#### **`process_single_prompt(patent_data, prompt, model_name="gemini-1.5-flash")`**

This function sends a single prompt to the **Gemini 1.5 Flash** model for analysis. It:

- Submits the prompt to the model and retrieves the generated content.
- Leverages **context caching** to optimize performance by storing and reusing previously analyzed prompts, reducing redundant processing.
- Captures token usage details, including the number of tokens used for the input, output, total, and cached content.
- Handles errors gracefully and flags any unsuccessful attempts.

```python
def process_single_prompt(patent_data, prompt, model_name="gemini-1.5-flash"):
    try:
        # Ensure prompt meets minimum token 
        if len(prompt) < 1000:  # Add more content if prompt is too short
            prompt = prompt + " " + " ".join([
                "Additional context to meet token requirements. ",
                "Expanding on the original analysis to ensure sufficient content. ",
                "Providing supplementary information to enhance the cached content."
            ] * 5)

        # Create cache with extended TTL
        cache = genai.caching.CachedContent.create(
            model=f'models/{model_name}-001',
            display_name=f'Patent Analysis: {patent_data.get("Patent ID", "Unknown")}',
            system_instruction=(
                'You are an expert patent analysis assistant, '
                'providing detailed and accurate patent overlap assessments.'
            ),
            contents=[prompt],
            ttl=datetime.timedelta(hours=2)  # Cache time
        )

        # Model configuration
        generation_config = { "temperature": 0.2, "top_p": 1.0, "top_k": 1,  "max_output_tokens": 8192,  "response_mime_type": "text/plain",}

        # Generate using cached content
        model = genai.GenerativeModel.from_cached_content(cached_content=cache, generation_config=generation_config)
        
        response = model.generate_content(prompt)
        
        # Token usage tracking
        token_usage = {
            "Prompt Tokens": response.usage_metadata.prompt_token_count,
            "Output Tokens": response.usage_metadata.candidates_token_count,
            "Total Tokens": response.usage_metadata.total_token_count,
            "Cached Tokens": response.usage_metadata.cached_content_token_count
        } if response.usage_metadata else None

        print(f"Analysing: Token usage: {token_usage}")

        return {"analysis": response.text, "token_usage": token_usage, "success": True }
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"analysis": None,"token_usage": None, "success": False }
```

#### **`prompt_template`**
This is a formatted string template to guide the model’s analysis. It organizes the data from the main patent and its similar patents into clear sections, helping the model perform tasks like identifying overlaps, evaluating novelty, and providing recommendations.

```python
prompt_template = """
You are tasked with analyzing the overlap and novelty of patents to assist a patent-granting officer in making informed decisions about granting or rejecting patents. The goal is to evaluate both overlaps and unique contributions, ensuring that novelty is highlighted wherever possible. Use the following format for your analysis.

### Main Patent:
- **ID:** {main_patent_id}
- **Title:** {main_title}
- **Priority Date:** {main_priority_date}
- **Publication Date:** {main_publication_date}
- **Full Text:** {main_full_text}

### Similar Patents:
{similar_patents}

### Similarity Scores:
{similarity_scores}

### Analysis Tasks:
1. **Scope and Timeline Analysis**:
   - Compare the priority dates of the patents. Discuss whether the main patent predates or follows the similar patents and the implications for novelty.
   - Provide an overview of the titles to explain the focus areas of the similar patents.

2. **Overlap Determination**:
   - Identify overlapping claims, features, or technologies between the main patent and the similar patents. Highlight specific overlapping elements.
   - Consider whether the overlap is substantial or incidental, providing reasoning for your conclusion.
   - Avoid flagging overlap unless it clearly impacts the novelty or utility of the main patent.

3. **Novelty Evaluation**:
   - Focus on identifying unique features or technical contributions of the main patent compared to the similar patents.
   - Highlight aspects of the main patent that improve upon or differentiate it from prior work.
   - Specify whether the main patent applies existing techniques in a novel way or introduces entirely new concepts.

4. **Flagging for Overlap**:
   - Provide a clear conclusion: Should the main patent be flagged as overlapping with one or more similar patents? Only flag if the overlap undermines the novelty or utility of the main patent.
   - Justify your decision with evidence.

5. **Recommendations for Patent Officer**:
   - If flagged as overlapping:
     - Suggest specific areas where the claims or descriptions need refinement to establish novelty.
     - Recommend further investigation into the patent family or related filings to confirm overlap.
   - If not flagged as overlapping:
     - Explain why the main patent is sufficiently distinct and can be considered for granting.

6. **Risk Assessment (Optional)**:
   - Identify potential risks of granting this patent (e.g., legal disputes, infringement, or redundancies in existing patents).

### Example Output:
#### Scope and Timeline Analysis:
- **Priority Dates:** 
  - Main Patent: {main_priority_date}
  - Similar Patents: [List dates with IDs]
- **Focus Areas:**
  - Main Patent: {main_title}
  - Similar Patents: [List titles with IDs]

#### Overlap Determination:
- **Overlapping Features:** [List overlapping features]
- **Assessment of Overlap:** [Explain whether the overlap is substantial or incidental.]

#### Novelty Evaluation:
- **Unique Features:** [List unique features.]
- **Novelty Conclusion:** [Summarize the novelty or lack thereof, focusing on any identified unique contributions.]

#### Flagging for Overlap:
- **Flagged:** [Yes/No]
- **Justification:** [Provide evidence for the decision.]

#### Recommendations for Patent Officer:
- **Actions to Take:** [Suggest actions or refinements.]

#### Risk Assessment:
- **Potential Risks:** [List risks if applicable.]
"""
```

#### **Running the Analysis**

```python
results_df, token_usage_df = compare_patent_groups(
    data=data,
    similar_patents=similar_patents,
    prompt_template=prompt_template,
    max_compare=3,word_limit=150000, max_success=3)
```

<a id="4.5"></a>


## **4.5 ✨ Patent Analysis and 📊 Token Usage**

This section demonstrates the use of **Gemini-1.5-Flash** to analyze three patents and their overlaps. By focusing on a smaller subset of high-similarity patents, the analysis highlights novelty, overlapping claims, and actionable recommendations.


#### **Token Usage Summary**
- **Prompt Tokens:** ~322,662 tokens per analysis.
- **Output Tokens:** Ranges between 940 and 1,393 tokens.
- **Total Tokens:** ~323,000 tokens per analysis.
- **Cached Tokens:** Consistently around 161,339 tokens per analysis, showcasing the efficiency of context caching.

These analyses illustrate the practicality of long-context models like **Gemini 1.5 Flash** in handling detailed patent datasets efficiently.


### **Analysis Results and Token Usage**
Let’s review some key outputs from the analysis:

```python
print("Analysis Results:")
results_df.head()
```

```python
print("Token Usage Statistics:")
token_usage_df.head()
```

```python
# Plot token usage 
plt.figure(figsize=(10, 6))
plt.bar(token_usage_df['Main Patent'], token_usage_df['Prompt Tokens'], alpha=0.7, color=palette[2], edgecolor=palette[3], label="Prompt Tokens")
plt.xlabel("Main Patent ID", fontsize=12)
plt.ylabel("Number of Tokens", fontsize=12)
plt.title("Token Usage Across Patent Analyses", fontsize=14, pad=10)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()
```

### **Key Takeaways**

#### **Patent Analysis ✅**
- **Overlaps:** Identified overlapping claims and highlighted unique contributions for 3 key patents.
- **Novelty:** Evaluated novelty based on shared priority dates and specific claims.
- **Actionable Recommendations:** Provided insights to refine patent claims and enhance intellectual property protection.

#### **Token Usage ✅**
- **Efficiency:** Efficiently processed 3 high-similarity patents, with total tokens consistently around 323,000 per analysis.
- **Visualization:** The bar chart illustrates token consumption across multiple analyses, highlighting the model’s ability to handle long-context inputs seamlessly.

#### **Workflow Optimization:**
- **Focused Analysis:** Limiting the analysis to 3 patents ensures token limits remain manageable while yielding actionable insights.
- **Context Caching:** Cached tokens consistently reduce redundant computations and improve efficiency.

---

With the prompts prepared and analyzed, we extracted valuable insights from the overlapping and unique claims of the patents. By leveraging **Gemini 1.5 Flash’s long-context capabilities**, we efficiently processed and evaluated high-similarity patents while optimizing token usage through **context caching**. 

The next step explores **Results and Insights**, focusing on how this workflow identifies novel contributions, assesses redundancy, and provides actionable recommendations to improve intellectual property protection.

<a id="5"></a>

# <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:center; padding:10px; background: #81667a; border-radius: 25px; box-shadow: 10px 10px 5px #8c8a93;" > 📊 Results and Insights <br><div> 


### **Detailed Patent Analysis Reports**

This section presents the results of analyzing **three patents** using **Gemini 1.5 Flash**. The analysis focuses on identifying overlaps, evaluating novelty, and providing actionable recommendations for refining patent claims. 

Follow-up questions leveraging **context caching** demonstrate how previously analyzed data can be reused to generate deeper insights efficiently.

```python
#Function to print output as Markdown
def display_analysis_in_markdown(results_df, indices):

    for idx in indices:
        if idx < 0 or idx >= len(results_df):
            print(f"Index {idx} is out of bounds.\n")
            continue

        row = results_df.iloc[idx]
        md_content = f"### Patent Analysis Report: {row['Main Patent']}\n"
        md_content += f"#### Status: **{row['Status']}**\n"

        if row['Status'] == "Skipped":
            md_content += f"**Reason for Skipping:** {row['Reason']}\n"
        else:
            similar_patents = row.get('Similar Patents', [])
            md_content += "#### Similar Patents\n" + (
                "\n".join(f"- **Patent {i+1}:** {patent}" for i, patent in enumerate(similar_patents))
                if similar_patents else "No similar patents found."
            )
            md_content += f"\n\n#### Detailed Analysis\n{row['Analysis']}\n"

        md_content += "---\n"
        display(Markdown(md_content))


# Function to ask follow up question
def ask_patent_follow_up(results_df, index, custom_question):

    if not 0 <= index < len(results_df):
        raise ValueError(f"Invalid index. Must be between 0 and {len(results_df) - 1}")
    
    # Extract patent details
    patent_id = results_df.iloc[index]['Main Patent']
    caches = genai.caching.CachedContent.list()
    current_cache = next((cache for cache in caches if f"Patent Analysis: {patent_id}" in cache.display_name), None)
    
    if not current_cache:
        raise ValueError(f"No cached content found for patent {patent_id}")
    
    # Generate responce
    model = genai.GenerativeModel.from_cached_content(
        cached_content=current_cache,
        generation_config={"temperature": 0.2}  
    )
    response = model.generate_content(f"In no more than three paragraphs, answer: {custom_question}") 
    #added this limitation for the purpose of demontration 
    
    # Format and display output
    markdown_output = f"""
##### Patent: {patent_id}
#### 🧐 Question:
**{custom_question}**

#### 🤖 Answer:
{response.text.strip()}
"""
    display(Markdown(markdown_output))
```

<a id="5.1"></a>

## Case 1: 


Analysis and follow up question.

```python
# Display analysis for the first patent
selected_indices = [0] 
display_analysis_in_markdown(results_df, selected_indices)
```

<div class="alert alert-block alert-info" style="font-family: Arial; font-size: 115%; color: black; background-color: #e7f3fe; padding: 10px; border-radius: 5px;">
  <h3 style="font-weight: bold;">📄 Insights From Our Patent Analysis: US-10438354-B2</h3>
  <p>
    The analysis of <strong>US-10438354-B2</strong> highlights its novelty and overlap with similar patents, particularly <strong>US-11003988-B2</strong>. Both patents utilize deep learning but differ in focus: the main patent emphasizes medical imaging techniques and procedures, while the similar patent focuses on hardware system design improvements.
  </p>
  <p>
    <strong>Key Findings:</strong>
    <ul>
      <li><strong>Overlaps:</strong> Both patents use deep learning and explore hardware-software integration, showing some conceptual similarities.</li>
      <li><strong>Novelty:</strong> The main patent stands out with its broader focus on medical imaging techniques, relying on unique methods and applications.</li>
    </ul>
  </p>
  <p>
    <strong>Next Steps:</strong> To further strengthen the case for <strong>US-10438354-B2</strong>, additional clarification is necessary:
    <ul>
      <li>How do the main patent’s specific medical techniques and systems differ from prior art and similar patents?</li>
      <li>Highlighting these unique features can mitigate overlap risks and reinforce the patent’s novelty.</li>
    </ul>
  </p>
</div>

```python
Ques_1 = "Highlighting these unique features can mitigate overlap risks and reinforce the patent’s novelty."

ask_patent_follow_up(results_df,index=0,custom_question= Ques_1)
```

---
<a id="5.2"></a>

## Case 2: 

Analysis and follow up question.

```python
# Display analysis for the second patent
selected_indices = [1]  
display_analysis_in_markdown(results_df, selected_indices)
```

<div class="alert alert-block alert-info" style="font-family: Arial; font-size: 115%; color: black; background-color: #e7f3fe; padding: 10px; border-radius: 5px;">
  <h3 style="font-weight: bold;">📄 Insights From Our Patent Analysis: US-10896352-B2</h3>
  <p>
    The analysis of <strong>US-10896352-B2</strong> highlights its focus on leveraging deep learning for image reconstruction and quality evaluation in medical systems. Significant overlap exists with related patents, particularly <strong>CN-110121749-B</strong>, which also addresses deep learning applications for medical imaging, specifically image acquisition settings.
  </p>
  <p>
    <strong>Key Unique Features:</strong>
    <ul>
      <li>Automatic generation of an image quality metric using deep learning techniques.</li>
      <li>Introduction of a comprehensive Image Quality Index (IQI) that evaluates image attributes such as spatial resolution, noise levels, and detectability.</li>
      <li>Adaptive feedback loops that trigger real-time changes in image acquisition or reconstruction settings based on the IQI.</li>
    </ul>
  </p>
  <p>
    <strong>Challenges and Recommendations:</strong>
    <ul>
      <li><strong>Overlap Risks:</strong> The substantial thematic overlap with <strong>CN-110121749-B</strong> and other similar patents increases the potential for redundancy and legal disputes.</li>
      <li><strong>Mitigation Strategies:</strong> Refine the claims of US-10896352-B2 to emphasize its unique contributions, such as the detailed use of the IQI for adaptive imaging workflows. Additionally, a thorough review of the patent family and related filings is recommended to ensure clear distinctions.</li>
    </ul>
  </p>
  <h4 style="font-weight: bold;">🧐 Potential Follow-Up Question:</h4>
  <p>
    How does the adaptive feedback loop mechanism in US-10896352-B2, driven by the Image Quality Index (IQI), enhance diagnostic precision compared to conventional image quality assessment methods in similar patents?
  </p>
</div>

```python
Ques_2 ="How does the adaptive feedback loop mechanism in US-10896352-B2, driven by the Image Quality Index (IQI), enhance diagnostic precision compared to conventional image quality assessment methods in similar patents?"

ask_patent_follow_up(results_df, index=1,custom_question= Ques_2)
```

---
<a id="5.3"></a>

#### Case 3:

Analysis and follow up question.

```python
# Display analysis for the third patent
selected_indices = [2]  
display_analysis_in_markdown(results_df, selected_indices)
```

<div class="alert alert-block alert-info" style="font-family: Arial; font-size: 115%; color: black; background-color: #e7f3fe; padding: 10px; border-radius: 5px;">
  <h3 style="font-weight: bold;">📄 Insights From Our Patent Analysis: CN-110121749-B</h3>
  <p>
    The analysis of <strong>CN-110121749-B</strong> highlights its focus on using deep learning for medical image acquisition. While the patent emphasizes a specific application area, it faces notable overlap with similar patents such as <strong>US-10896352-B2</strong> and <strong>US-10438354-B2</strong>, which also address deep learning for medical imaging tasks.
  </p>
  <p>
    <strong>Key Unique Features:</strong>
    <ul>
      <li>Specific focus on optimizing image acquisition in medical systems using deep learning.</li>
      <li>Potentially novel configurations for integrating deep learning into medical image acquisition workflows.</li>
      <li>Emphasis on improving accuracy and efficiency in imaging settings, tailored to specific diagnostic needs.</li>
    </ul>
  </p>
  <p>
    <strong>Challenges and Recommendations:</strong>
    <ul>
      <li><strong>Overlap Risks:</strong> Substantial thematic overlap exists with patents focusing on image reconstruction and system design, raising concerns about novelty.</li>
      <li><strong>Mitigation Strategies:</strong> Conduct a detailed claim-by-claim comparison to emphasize the unique contributions of <strong>CN-110121749-B</strong>. Investigate prior art and related filings to refine its claims and reduce redundancy.</li>
    </ul>
  </p>
  <h4 style="font-weight: bold;">🧐 Potential Follow-Up Question:</h4>
  <p>
    What specific innovations in CN-110121749-B differentiate its approach to medical image acquisition from similar patents, and how do these innovations improve diagnostic precision or efficiency?
  </p>
</div>

```python
Ques_3 = "What specific innovations in CN-110121749-B differentiate its approach to medical image acquisition from similar patents, and how do these innovations improve diagnostic precision or efficiency?"

ask_patent_follow_up(results_df,index=2,custom_question= Ques_3)
```

This section presented three selected outputs from Gemini 1.5 Flash's analysis, showcasing varying levels of overlap and novelty. The examples demonstrated:

- **Actionable Recommendations**: Practical insights for refining patent claims and strengthening intellectual property protection.
- **Impact of Priority Dates**: Addressing shared filing timelines to balance novelty and mitigate redundancy risks.
- **Broader Implications**: Providing clarity on overlap patterns and highlighting unique contributions to the field.

Together, these outputs illustrate how Gemini 1.5 Flash transforms patent workflows, enabling enhanced scalability and innovation. 🚀

<a id="6"></a>

# <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:center; padding:10px; background: #81667a; border-radius: 25px; box-shadow: 10px 10px 5px #8c8a93;" > 🎯 Leveraging Gemini 1.5 for Enhanced Patent Analysis <br><div> 

This project demonstrated the potential of **Gemini 1.5** to redefine workflows for analyzing complex patent data. By utilizing Gemini’s long-context processing capabilities and **context caching**, we seamlessly integrated inputs exceeding 160,000 tokens. Context caching enabled efficient reuse of previously analyzed data, reducing redundant processing and improving overall performance. 

The workflow incorporated diverse sections, including Titles, Abstracts, Claims, and Descriptions, alongside metadata such as Priority and Publication Dates. This holistic approach facilitated detailed evaluations of overlaps, novelty, and claim refinement opportunities, providing actionable insights to support informed decision-making.

To operate effectively within free-tier constraints, we implemented **context caching**, pre-analysis word count checks, and pre-calculated similarity scores. These measures ensured that inputs adhered to token limits and computations were focused on the most relevant comparisons. The streamlined, single-pass process eliminated retries and batching, enhancing efficiency while maintaining consistency.

This methodology has broad implications across industries:

- **Legal Documentation**: Analyzing and comparing contracts or case law for improved decision-making.  
- **Scientific Research**: Detecting overlaps in research papers and generating summaries for innovation tracking.  
- **Education and Media**: Tailoring content recommendations by analyzing long-form materials like lectures or books.  

Looking ahead, expanding the analysis to larger datasets and integrating multimodal capabilities—such as technical schematics—will further enhance the applicability of Gemini 1.5 in real-world scenarios.

<a id="7"></a>

# <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:center; padding:10px; background: #81667a; border-radius: 25px; box-shadow: 10px 10px 5px #8c8a93;" > ✅ Conclusion <br><div>  

This project successfully showcased how **Gemini 1.5** can transform patent workflows, addressing challenges inherent in analyzing complex, high-volume datasets. By processing over 160,000 tokens seamlessly, Gemini delivered detailed, actionable insights, setting a new benchmark for long-context AI applications.

### 🔍 Key Takeaways:
- **Efficient Patent Analysis**: Comprehensive evaluations of overlaps, novelty, and claims were achieved without traditional bottlenecks.  
- **Resource Optimization**: Free-tier constraints were managed effectively through **context caching** and targeted token management.  
- **Scalable Potential**: Demonstrated readiness for broader applications, such as analyzing larger datasets or extending document contexts.

### 🌟 Reflections:
The insights gained from this project extend beyond patent analysis, highlighting how **Gemini 1.5** simplifies workflows in industries like intellectual property, research, and education. This approach enables deeper analysis and actionable outcomes, paving the way for broader adoption in document-heavy tasks.

Future directions include scaling workflows to larger patent families, integrating citation metadata for richer insights, and exploring multimodal capabilities like handling technical diagrams alongside text. These advancements will enhance the utility of **Gemini 1.5**, making it indispensable for real-time monitoring, R&D insights, and more.

By redefining what’s possible in long-context processing, this project underscores Gemini’s value in creating innovative, data-driven solutions across industries.


---

## 📎 Resources

Here are some key resources referenced or utilized in this project:

- **[Gemini 1.5 Documentation](https://ai.google.dev/gemini-api/docs)**: Learn more about Gemini’s long-context capabilities and API usage.  
- **[Notebook on Large-Scale Inputs](https://www.kaggle.com/code/paultimothymooney/how-to-upload-large-files-to-gemini-1-5)**: A guide on uploading large files for Gemini analysis.  
- **[Google AI Studio](https://ai.google.dev/)**: Access Gemini and manage your API keys.  
- **[Kaggle Competition](https://www.kaggle.com/competitions/gemini-long-context)**: Explore related data science challenges.  



<a id="8"></a>
# <div style="font-family: 'Playfair Display', serif; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:center; padding:10px; background: #81667a; border-radius: 25px; box-shadow: 10px 10px 5px #8c8a93;" >⚡ Fin ⚡<br><div>
# fake news analysis Part1

- **Author:** tomasz szyrowski
- **Votes:** 55
- **Ref:** tomaszszyrowski/fake-news-analysis-part1
- **URL:** https://www.kaggle.com/code/tomaszszyrowski/fake-news-analysis-part1
- **Last run:** 2024-11-07 14:50:05.300000

---

# **Using Gemini to Detect Fake News**

```python
import base64
from IPython.display import HTML, display

support_video = "/kaggle/input/fakenewsanalysispart1-support-video/FakeNewsAnalysisPart1.mp4"

with open(support_video, "rb") as f:
    video_data = f.read()
video_base64 = base64.b64encode(video_data).decode()

html_code = f"""
<style>
.responsive-video-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
}}

.responsive-video {{
    width: 100%;
    max-width: 800px;
    height: auto;
    margin-top: 0px;
}}
</style>

<div class="responsive-video-container">
    <h3>Analysis Overview Video</h3>
    <video class="responsive-video" controls>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>
"""
display(HTML(html_code))
```

>**If you find this notebook helpful, please consider giving it an <b><span style="color:red">upvote</span></b>!**<br> 
Your support encourages us to create more insightful content. <span style="color:red">Thank you</span>

## **Aims and Purposes**

In the 21st century, misleading information has become a powerful force across news platforms, from major global outlets to social media streams. We’re excited to explore how Gemini AI can tackle the spread of fake news, playing a vital role in protecting individuals and society from opinions shaped by false or deliberately manipulative content.

**Our objectives include:**

- **Swift Detection:** Assessing how quickly Gemini can identify fake news by recognizing subtle patterns indicative of unreliability or manipulation.
- **Accuracy:** Evaluating the precision of Gemini in distinguishing between genuine and fake content.
- **Scalability:** Exploring the model's capability to analyze large volumes of data across diverse media sources.
  
Through this analysis, we seek to demonstrate the potential of AI-driven solutions in enhancing information integrity and supporting informed decision-making.


> This notebook serves as our first attempt to harness Gemini's potential to analyze the integrity of article content and quantify the presence of fake news.

> **⚠️ Important Note:**
>
> To manage API costs, if a **Gemini API Key** is not provided, this notebook will utilize a free alternative Language Model for demonstration purposes.
>
> **Please be aware:** The free model **does not comply with Gemini's capabilities**, and results may vary in accuracy and performance.

After analysing prompt construction in this notebook we have continued further in [Part2](https://www.kaggle.com/code/tomaszszyrowski/fake-news-analysis-part2)

## **Setting Up the Environment**

Before diving into the analysis, we need to set up our working environment. This includes importing necessary libraries, installing missing dependencies, and configuring the AI models required for our fake news detection tasks.

```python
from datetime import datetime
import os
import re
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

try:
    import requests
except ModuleNotFoundError:
    !pip install requests
    from bs4 import requests
try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    !pip install beautifulsoup4
    from bs4 import BeautifulSoup

from IPython.display import display
```

Setting up the necessary libraries and tools is crucial for efficient data extraction and analysis. Here's what this setup accomplishes:

- **Web Scraping:** Utilizes the `requests` and `BeautifulSoup` libraries to extract specific elements such as paragraphs and headlines from web articles.
- **Dependency Management:** Automatically installs any missing libraries, ensuring that the notebook runs smoothly across different environments without manual intervention.
- **Flexibility:** Prepares the environment to handle various data sources and formats, facilitating seamless data processing for our analysis.

```python
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    !pip install google-generativeai
    import google.generativeai as genai
try:
    from kaggle_secrets import UserSecretsClient
except ModuleNotFoundError:
    print("I guess you running it local so don't worry")
    print("You may consider creating kaggle_secrets.UserSecretsClient for local use")
from IPython.display import Markdown
# Set up Google Gemini API key from Kaggle secrets
try:
    user_secrets = UserSecretsClient()
    apiKey = user_secrets.get_secret("GEMINI_API_KEY")
    genai.configure(api_key=apiKey)
    model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
    GEMINI_MODEL = True
except:
    !pip install transformers
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    GEMINI_MODEL = False
```

```python
if GEMINI_MODEL:
    print("Loaded Gemini model")
    print(model)
else:
    print("Loaded Free model")
    print(classifier)
    print(dir(classifier))
```

This code sets up an AI model for text analysis, making it adaptable based on the resources available. It first tries to load Google’s Gemini model for advanced capabilities; if it isn’t installed, the code automatically installs it and retrieves the API key securely through Kaggle secrets. If the Gemini model can’t be used, it switches to a free alternative model, which ensures the analysis can still proceed. By printing details on the loaded model, this setup provides confirmation of which model is active, helping streamline the analysis process while keeping it flexible and resource-efficient.

## **Input**

```python
article_url = "https://en.wikipedia.org/wiki/Fake_news"
```

It was decided that giving a URL, in this case a Wikipedia page on "Fake News"

Providing a URL, like this Wikipedia page on "Fake News," is much more efficient than copying and pasting large blocks of text. This approach allows Gemini to access, define, and analyze the article directly through its link, making the process faster, cleaner, and less prone to formatting issues. By simply referencing the URL, the code can pull in updated content, ensuring that the analysis always uses the latest information available online.

```python
def fetch_article_text(url):
    """
    Fetches the main text from a news article URL.

    Parameters
    ----------
    url : str
        The URL of the article to be analyzed.

    Returns
    -------
    str
        The extracted main text from the article.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        article_text = "\n".join([para.get_text() for para in paragraphs])
        return article_text
    else:
        print("Failed to retrieve the article.")
        return ""
```

```python
article = fetch_article_text(article_url)
```

This function is designed to retrieve and extract the main text from the specified URL. 

It makes an HTTP request to the URL and parse the HTML, pulling out paragraphs from the source. This allows the article's main content to be easily accessed for analysis. If the request fails, it prints an error message, ensuring the user is informed when the content cannot be retrieved. The extracted text is then stored in the article variable, ready for further processing.

```python
prompts = [
    {
        "number": 1,
        "text": (
            f"Analyse the following article and provide a single number as a percentage of fake news."
            f"```"
            f"{article}"
            f"```"
        ),
    },
        {
        "number": 2,
        "text": (
            f"Imagine you are assessing the truthfulness of this article as an expert fact-checker. "
            f"Estimate and provide a single percentage to represent the amount of fake news in the article, "
            f"even if the percentage is an approximation. Justify any subjective factors briefly but ensure "
            f"your output is a single percentage value followed by a short explanation."
            f"```"
            f"{article}"
            f"```"
        ),
    },
    {
        "number": 3,
        "text": (
            f"Imagine you are an expert fact-checker. Assess only the percentage of "
            f"fake news in this article and provide a single percentage representing "
            f"the amount of fake news present. Start your response with just the "
            f"percentage (e.g., '90%') followed by a brief explanation if necessary."
            f"```"
            f"{article}"
            f"```"
        ),
    },
    {
        "number": 4,
        "text": (
            f"Evaluate this article as an expert fact-checker and provide a percentage "
            f"representing the amount of factually incorrect information it contains. "
            f"Your answer should only consider concrete inaccuracies or verifiable "
            f"errors. Provide only a single percentage number as your response."
            f"```"
            f"{article}"
            f"```"
        ),
    },
    {
        "number": 5,
        "text": (
            f"As a media analyst, assess this article for fake news, defined as any "
            f"content that could be misleading, biased, or lacking in evidence. "
            f"Estimate the percentage of such information in the article and briefly "
            f"explain any factors influencing your assessment, but start with a "
            f"single percentage value."
            f"```"
            f"{article}"
            f"```"
        ),
    },
    {
        "number": 6,
        "text": (
            f"As an expert in media literacy, determine the percentage of this article "
            f"that consists of unsubstantiated claims or exaggerated statements, which "
            f"could contribute to misinformation or misunderstanding. Start your response "
            f"with a single percentage value, then give a brief explanation of any examples "
            f"of exaggeration or lack of substantiation if necessary."
            f"```"
            f"{article}"
            f"```"
        ),
    }
]
```

By storing the article in a list of prompts, each prompt requests a slightly different analysis of the article’s authenticity. One prompt simply asks for a fake news percentage, while others request a deeper assessment, asking Gemini to estimate fake news levels with a brief justification. This structured approach helps the AI model evaluate the article's reliability from multiple angles, aiming to provide a well-rounded analysis of its truthfulness.

This approach adds a valuable layer to our project analysis by exploring how different prompts can lead to varying outcomes, highlighting AI's sensitivity to the way questions are framed. This examination reveals the nuances in AI responses and underscores the importance of prompt design in achieving accurate and reliable insights.

## **Analysis**

```python
def analyse_fake_news_amount(prompt_text):
    """
    Runs the analysis on the provided prompt text to determine the percentage
    of fake news.

    Parameters
    ----------
    prompt_text : str
        The prompt text to analyze.

    Returns
    -------
    str or object
        The full response object from the model or classifier.
    """
    if GEMINI_MODEL:
        return model.generate_content(prompt_text)
    else:
        labels = ["fake news", "real news"]
        output = classifier(prompt_text, candidate_labels=labels)
        return f"Result: \n{output['labels'][0]}: {output['scores'][0]}\n{output['labels'][1]}: {output['scores'][1]}"
```

This function is useful because it enables rapid assessment of text authenticity, helping identify potentially misleading or inaccurate content. By automatically switching to the best available model, it ensures that analyses can proceed regardless of resource limitations. This functionality supports critical tasks like content verification and fact-checking, providing a convenient and efficient tool to flag information that may require further scrutiny.

## **Outputs**

```python
import textwrap

def analyse_with_prompts(prompts):
    responses = {}
    for prompt in prompts:
        prompt_number = prompt["number"]
        prompt_text = prompt["text"]
        response = analyse_fake_news_amount(prompt_text)
        responses[prompt_number] = response
    return responses
               
responses = analyse_with_prompts(prompts)    


def print_responses(responses):
    for prompt_number, response in responses.items():
        print("\n" + "="*80)
        print(f"Prompt {prompt_number} output:\n")
        if GEMINI_MODEL:
            # Extract the response text and usage metadata
            try:
                response_text = response.candidates[0].content.parts[0].text.strip()
                # response_text = response.text.strip()
                usage_metadata = response.usage_metadata
            except Exception as e:
                print(f"Error processing response for prompt {prompt_number}: {e}")
                continue
            
            # Ensure the response text is clean and wrapped for better readability
            response_text = response_text.replace('\r\n', '\n').replace('\r', '\n')
            
            # Print the response text
            print("Response Text:\n")
            print(textwrap.fill(response_text, width=80))
            
            # Print the usage metadata
            print("\nUsage Metadata:")
            print(f"  Prompt Token Count: {usage_metadata.prompt_token_count}")
            print(f"  Candidates Token Count: {usage_metadata.candidates_token_count}")
            print(f"  Total Token Count: {usage_metadata.total_token_count}")
        else:
            # For the classifier, just print the response
            print(f"{response}\n")
        print("\n" + "="*80 + "\n")

print_responses(responses)
```

For each prompt, the organization makes it easy to keep track of each analysis result and compare responses. By printing each output immediately, it provides instant feedback, allowing users to quickly review how each prompt was assessed for fake news, making it easier to spot differences and insights across prompts.

## **Final output**

```python
def parse_fake_news_percentage(response):
    """
    Parses the model output to extract a percentage value indicating the amount
    of fake news, if present.

    Parameters
    ----------
    response : str or object
        The model output; a string for Hugging Face classifier or a response 
        object for Gemini.

    Returns
    -------
    str
        A formatted string with the parsed percentage or a message indicating
        that the prompt did not succeed in providing a percentage.
    """
    if GEMINI_MODEL:
        try:
            # Access the response text directly for Gemini
            response_text = response.candidates[0].content.parts[0].text.strip()
            # response_text = response.text.strip()
            
            # Use a regex to find a percentage in the response text, including those within special characters
            match = re.search(r'(\d{1,3})%\b', response_text)
            if not match:
                # If standard search fails, try searching for percentage surrounded by characters like ** or ##
                match = re.search(r'[*#]*(\d{1,3})%[*#]*', response_text)

            if match:
                percentage = int(match.group(1))
                return f"Percentage of fake news in text: {percentage}%"
            else:
                return "The prompt did not succeed in providing a clear percentage."

        except (AttributeError, IndexError) as e:
            print("Error accessing Gemini response:", e)
            return "The prompt did not succeed in providing a clear percentage."

    else:
        # Handle Hugging Face output format
        try:
            # Match and extract the fake news score in Hugging Face output
            match = re.search(r'fake news: ([0-9.]+)', response)
            if match:
                fake_news_score = float(match.group(1)) * 100
                return f"Percentage of fake news in text: {fake_news_score:.2f}%"
            else:
                return "The prompt did not succeed in providing a clear percentage."
        except Exception as e:
            print("Error processing Hugging Face response:", e)
            return "The prompt did not succeed in providing a clear percentage."
```

### Update log of percentage between runs

```python
def initialize_paths(log_file_name="fake_news_analysis_log.csv"):
    """Initialize paths for saving and loading the log file."""
    working_log_file_path = os.path.join('/kaggle/working', log_file_name)
    dataset_log_file_path = os.path.join('/kaggle/input/fake-news-analysis-log', log_file_name)
    return working_log_file_path, dataset_log_file_path

def initialize_run_data():
    """Initialize the run data with the current timestamp."""
    return {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": datetime.now().strftime("%Y%m%d%H%M%S")
    }

def extract_prompt_percentages(responses):
    """Extract and populate fake news percentages from responses into run_data."""
    run_data = initialize_run_data()
    for prompt_number, response in responses.items():
        parsed_percentage = parse_fake_news_percentage(response)
        match = re.search(r'(\d{1,3})%', parsed_percentage)
        run_data[f"prompt_{prompt_number}"] = int(match.group(1)) if match else None
    return run_data

def load_existing_log(working_log_file_path, dataset_log_file_path):
    """Load the existing log file, or create a new DataFrame if none exists."""
    if os.path.exists(working_log_file_path):
        print("Log file loaded from /kaggle/working.")
        return pd.read_csv(working_log_file_path)
    elif os.path.exists(dataset_log_file_path):
        print("Log file loaded from dataset.")
        return pd.read_csv(dataset_log_file_path)
    else:
        print("No existing log file found. Starting a new DataFrame.")
        return pd.DataFrame()

def save_log_file(fake_news_df, working_log_file_path):
    """Save the log DataFrame to the specified working log file path."""
    fake_news_df.to_csv(working_log_file_path, index=False)
    print("Log updated successfully.")
    print(f"To persist the changes, please download the updated log file from '{working_log_file_path}' and upload it to your Kaggle dataset '<user>/fake-news-analysis-log'.")

def update_log_with_new_data(responses):
    """Update the log file by loading, appending new data, and saving."""
    working_log_file_path, dataset_log_file_path = initialize_paths()
    run_data = extract_prompt_percentages(responses)
    fake_news_df = load_existing_log(working_log_file_path, dataset_log_file_path)
    fake_news_df = pd.concat([fake_news_df, pd.DataFrame([run_data])], ignore_index=True)
    save_log_file(fake_news_df, working_log_file_path)
    return fake_news_df
```

```python
fake_news_df = update_log_with_new_data(responses)

print("First 10 rows of the DataFrame:")
display(fake_news_df.tail(10))
```

```python
def prepare_fake_news_data(fake_news_df):
    """Prepare the DataFrame for plotting by melting and cleaning."""
    sns.set_style("whitegrid")
    fake_news_df = fake_news_df.replace([float("inf"), -float("inf")], float("NaN"))

    prompt_columns = [col for col in fake_news_df.columns if col.startswith("prompt_")]

    # Melt the DataFrame to prepare for plotting
    melted_df = fake_news_df.melt(
        id_vars=["datetime", "version"],
        value_vars=prompt_columns,
        var_name="Prompt",
        value_name="Fake News Percentage"
    )
    melted_df = melted_df.dropna(subset=["Fake News Percentage"])
    melted_df["Fake News Percentage"] = pd.to_numeric(melted_df["Fake News Percentage"])
    all_prompts = [f'prompt_{i}' for i in range(1, 7)]
    melted_df['Prompt'] = pd.Categorical(melted_df['Prompt'], categories=all_prompts)
    
    return melted_df, all_prompts

def plot_distribution(melted_df, all_prompts):
    """Plot the distribution of fake news percentages with histogram and KDE."""
    palette = sns.color_palette("husl", len(all_prompts))
    palette_dict = dict(zip(all_prompts, palette))

    plt.figure(figsize=(14, 7))
    ax = sns.histplot(
        data=melted_df,
        x="Fake News Percentage",
        hue="Prompt",
        bins=20,
        kde=False,
        palette=palette,
        alpha=0.6,
        multiple='layer', 
        edgecolor='none',
        legend=False 
    )
    for prompt in melted_df["Prompt"].cat.categories:
        subset = melted_df[melted_df["Prompt"] == prompt]
        sns.kdeplot(
            data=subset,
            x="Fake News Percentage",
            color=palette_dict[prompt],
            label=prompt,
            ax=ax,
            bw_adjust=0.5,
            linewidth=1.5  
        )

    plt.xlim(0, 100)
    plt.title("Distribution of Fake News Percentages per Prompt")
    plt.xlabel("Fake News Percentage (bins)")
    plt.ylabel("Frequency")
    ax.legend(title="Prompt", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_boxplot(melted_df, all_prompts):
    """Plot a boxplot showing spread and variability of fake news percentages."""
    palette = sns.color_palette("husl", len(all_prompts))
    
    plt.figure(figsize=(14, 7))
    sns.boxplot(
        data=melted_df,
        x="Prompt",
        y="Fake News Percentage",
        palette=palette
    )
    plt.title("Box Plot of Fake News Percentages per Prompt")
    plt.xlabel("Prompt")
    plt.ylabel("Fake News Percentage")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def display_plots(fake_news_df, plot_type="both"):
    """
    Display plots based on the plot_type argument.
    
    :param fake_news_df: DataFrame containing fake news data to be processed
    :param plot_type: Options include 'distribution', 'boxplot', or 'both'
    """
    melted_df, all_prompts = prepare_fake_news_data(fake_news_df)

    if plot_type in ["distribution", "both"]:
        plot_distribution(melted_df, all_prompts)

    if plot_type in ["boxplot", "both"]:
        plot_boxplot(melted_df, all_prompts)

display_plots(fake_news_df, plot_type="both")
```

```python
print("Final run parsed percentages:")
for prompt_number, response in responses.items():
    print(f"Prompt {prompt_number} percentage:\n{parse_fake_news_percentage(response)}")
```

This code extracts and formats the percentage of fake news from each model’s output, making it easy to interpret the results. For each response, it tries to pull a clear percentage of fake news; if it’s unable to do so, it provides a message indicating that the percentage wasn’t obtained. This final step ensures that each analysis result is presented in a clear, usable format, allowing for straightforward comparison of fake news estimates across all prompts.

## Analysis of Covid Article

As part of this work we focus on Fake News analysis in information about **`Covid19`**

```python
article_url = "https://en.wikipedia.org/wiki/COVID-19"
article = fetch_article_text(article_url)
responses = analyse_with_prompts(prompts)

print("Final run parsed percentages:")
for prompt_number, response in responses.items():
    print(f"Prompt {prompt_number} percentage:\n{parse_fake_news_percentage(response)}")
```

```python
time.sleep(60)  # requests in this notebook are likely exceed call per minute
num_runs = 10
covid_prompt_responses = []
for _ in range(num_runs):
    responses = analyse_with_prompts(prompts)
    run_data = extract_prompt_percentages(responses)
    covid_prompt_responses.append(run_data)
    time.sleep(60)
covid_df = pd.DataFrame(covid_prompt_responses)
display(covid_df.tail(10))
```

```python
display_plots(covid_df, plot_type="boxplot")
```

# **Conclusion**

Overall, this code will give us a framework for assessing the authenticity of online content by evaluation the percentage of fake news present in a text. This is the beginning that we hope to expand and apply to various media sources and explore a variety of ways we can use AI Gemini to analyse misleading information. 

The varied results from the six prompts highlight the complex and subjective nature of assessing "fake news" in a text. Each prompt frames the concept of fake news differently, from strictly factually incorrect information to broader, more subjective elements like bias, exaggeration, or lack of evidence. This variability underscores how different interpretations of "fake news" can lead to differing conclusions, especially when using prompts that ask for subjective or interpretive analysis. For instance, some prompts focused on unsubstantiated claims returned higher fake news percentages due to anecdotal or biased elements, whereas more fact-check-focused prompts returned lower percentages when the content was largely accurate but included minor unsupported claims.

Moving forward, a valuable direction would be to refine prompt design to create more consistent and objective assessments. Future work could involve experimenting with clearer definitions for categories like "bias," "misleading," or "unsupported claims," and incorporating a system of weighted metrics to give structure to these categories. This approach would help reduce the ambiguity in model responses and lead to more consistent percentages for fake news. Additionally, using AI to analyze content across a larger dataset could help identify common patterns in misinformation, improving both accuracy and reliability in detecting fake news across diverse media sources.
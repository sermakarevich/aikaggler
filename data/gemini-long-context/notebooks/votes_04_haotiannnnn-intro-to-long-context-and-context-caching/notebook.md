# Intro to Long Context and Context Caching

- **Author:** Haotian Huang
- **Votes:** 96
- **Ref:** haotiannnnn/intro-to-long-context-and-context-caching
- **URL:** https://www.kaggle.com/code/haotiannnnn/intro-to-long-context-and-context-caching
- **Last run:** 2024-12-07 22:03:11.953000

---

# Using Gemini to Understand Everything in Biology
## Intro to Long Context and Context Caching

> This notebook will show you how to use <u>Gemini 1.5</u> model's long context window. It will also teach you context caching to reduce operational costs associated with long context. 

The use case for this notebook is interesting because **long context is absolutely essential** to its success. In just a few minutes of running this notebook you will have an AI model that is capable of **answering tough questions about biology using scientifically-backed values** which is enabled by a large context window. A [video](https://www.youtube.com/shorts/5DU3gpyfjCM) accompanies this notebook. 

Questions like: 
* How much energy do mitochondria make in a day?
* How many rubisco proteins in a leaf?
* How long would it take for a single E. coli cell to grow into a 1-gram colony under ideal conditions?
* How many mutations will be in this DNA after 10 germline generations?
* How much tuberculosis bacteria must have been present at time of transmission and at time of diagnosis?
* What's the speed of neuronal development during the growth of a fruit fly from larva to adult?

## Datatset

The _BioNumbers_ dataset is a `csv` file containing 10,281 rows of numbers that are important in biology. An example would be the average thickness of a strand of human hair. The columns are: "BNID", "Properties", "Organism" ,"Value/Range" and "Units". It is from an ongoing 17-year long effort to solve the problem of quantifying biological parameters across a wide range of organisms, cellular processes and molecular interactions. An AI that could reason accurately across this broad dataset would become an invaluable resource for the field of biology. 

## Use case

Specifically, passing this dataset into long context allows Gemini to:
* Answer tough questions about biology
* Anchor the assumptions it makes with scientifically-backed numbers  
* Extend this reasoning to multimodal prompts with various scenarios

## How Long Context Enables This Use Case

Without long context, this use case would not be possible. The _BioNumbers_ dataset, spanning 10,281 rows and over 380,000 tokens, presents a challenge for traditional AI models limited by shorter context windows typically less than 128,000 tokens. With Gemini 1.5’s unprecedented long-context capacity of up to 2 million tokens, the model can ingest an entire corpus at once. In this way, long context uniquely enables this use case through (i) **robust end-to-end modelling**, (ii) **sophisticated prompting** and (iii) **enhanced user-friendliness**. 

(i) Robust end-to-end modeling eliminates the cascading errors and complexities inherent in multi-stage systems like retrieval-augmented generation (RAG). RAG systems, which attempt to address short context windows by first retrieving relevant information, are susceptible to errors at any stage that propagate and worsen downstream results. For example, an initial retrieval error in a RAG system dealing with BioNumbers might miss crucial biological values, leading to incorrect contextual representation and ultimately, a flawed final answer. Long-context retrieval avoids this by processing the entire dataset at once. This leads to greater trust in the model's output, particularly for critical biological calculations where minimizing sources of error is paramount.

(ii) Sophisticated prompting, enabled by the long-context capacity, allows for advanced techniques such as adding a system instruction. This is not merely a quality improvement but a fundamental shift in the model's capabilities. The ability to provide the model with examples and guide its reasoning process directly within the context of the entire dataset leads to more accurate and nuanced calculations. This is especially vital in biological numeracy where the interconnectedness of data requires complex reasoning, for instance, relating cell membrane size to permeability. The sophistication of the prompting allows for a deeper exploration of these relationships, going beyond simple keyword matching to true understanding.

(iii) Enhanced user-friendliness is crucial for the broader adoption and usability of the system. The intuitive nature of long-context interaction means that biologists without specialized training can easily leverage the system. This direct interaction avoids the complexities of separate retrieval tools and specialized queries, making the system more accessible to a wider range of users and fostering deeper biological intuition. If a system offers great potential but requires expert knowledge to operate, then that potential remains largely unrealized.

My old science teacher once told me biology was his least favorite subject because there were simply "too many things to remember." Looking back, I don't blame him; there really are a lot of things biologists need to remember. Long context enables us to leverage the reasoning capabilities of models and combine that with scientifically-verified biological numeracy, giving us the confidence to tackle complex biological questions that were previously too time-consuming to solve. Existing methods addressing short context windows are only workarounds; only long context allows the depth of reasoning crucial for sophisticated biological calculations. I'm sure by the end of this notebook, you will agree!

## Prerequisites

The most common issues people run into when getting started with long context on Kaggle are:

1. **Not having a phone-verified Kaggle account**: Without phone verification, critical features such as internet access in notebooks are disabled, leading to connection errors when trying to use external data or APIs.

2. **Creating a notebook without phone verification**: Even if you verify your phone number later, any notebooks created before verification will remain permanently internet-disabled. To resolve this, you need to either copy and edit the notebook or create a new one after verifying your phone number to ensure internet access is enabled.

By addressing these issues early, you can avoid common hurdles and get started smoothly with long-context workflows on Kaggle.

![](https://i.imgur.com/3XDHA2r.png)

## Import

```python
import google.generativeai as genai
from google.generativeai import caching
from IPython.display import display, Markdown, Video, Image
import pandas as pd
import time
import datetime
from kaggle_secrets import UserSecretsClient
```

## Set up your API key

To run the following cell, your API key must be stored it in a Kaggle User Secret named `GEMINI_API_KEY`. If you don't already have an API key, go to Google AI Studio and [create](https://aistudio.google.com/app/apikey) one for free. If you're not sure how to create a Kaggle User Secret, here is a [guide](https://www.kaggle.com/discussions/product-feedback/114053) you can read. Note you will have to be a verified Kaggle user to run this cell.

```python
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
```

## Create your model

In this notebook, we will be using `gemini-1.5-flash`, however you can also use `gemini-1.5-pro` if you want even more performance.

```python
MODEL_NAME = "models/gemini-1.5-flash-002"
model = genai.GenerativeModel(model_name=MODEL_NAME)
```

## Upload dataset

Use `upload_file` to upload larger text, image and audio data for your prompt or context.

```python
csv_path = "/kaggle/input/bionumbers-dataset/BioNumbers_Nov2024.csv"
csv_file = genai.upload_file(path=csv_path)

df = pd.read_csv(csv_path)
df.head()
```

## Prepare system instruction

System instructions are a type of sophisticated prompting technique that allows you to steer the behaviour of the model. By setting the system instruction, you are giving the model additional context to understand the task, provide more customized responses, and adhere to guidelines over the user interaction. 

In this case, we want the model to reference the values in its context when possible, as well as guide itself to an accurate answer within standard biological constraints. To aid independent verification, we want it to provide hyperlinks to the BioNumbers database entry from which the model is sourcing values from.

```python
system_instruction = """
You are a highly specialized biology calculator with access to a detailed database of biological numbers (BioNumbers) formatted as a CSV with the following columns: BNID, properties, organism, value/range, and units. Your tasks are:

Primary Data Usage: When answering questions, prioritize using the available data in the database and reference the correct BNID ID when citing any value.

Selecting Values: If multiple values are available, choose the most contextually relevant one and explain your reasoning for the choice.

Calculations: If a question involves calculations, explicitly use the provided data and reference BNIDs for all values used, including hyperlinks.

When Data is Missing or Incomplete:

- If no exact or relevant data exists in the database, provide a solution by making reasonable assumptions and estimates based on biological principles, ensuring that the solution is likely within the correct order of magnitude. Clearly state the assumptions made and provide justifications.
- Always prefer providing a worked solution, even if it involves assumptions, over stating that more information is needed.

Assumptions and Context: Clearly explain any assumptions made, contextualizing them within the problem. Bias towards actionability and problem-solving rather than precision when necessary.

Order-of-Magnitude Accuracy: Focus on delivering answers that are biologically plausible within the correct order of magnitude rather than hyper-precision, especially when data or constraints are unclear.

Accuracy and Clarity: Maintain accuracy when quoting data, calculations, and BNIDs. When making assumptions or estimates, prioritize transparency and clarity to ensure practical insights.

Markdown Output Requirement: All responses must be formatted in markdown-readable text to ensure clarity and usability. Use headings, lists, links, and code blocks appropriately. Ensure that the final output is plain text that can be copied and pasted directly into a code editor or markdown viewer.

Each BNID reference must include a hyperlink to its page on the BioNumbers website in the following format: [BNID <BNID value>](https://bionumbers.hms.harvard.edu/bionumber.aspx?id=<BNID value>).

Your goal is to provide biologically relevant, actionable solutions with clear reasoning, leveraging database values whenever possible while confidently bridging gaps with assumptions to deliver coherent, worked answers. Avoid unsupported or vague statements.
"""
```

# Send your first long-context prompt

**Important**: The following is a long-context prompt as the total input greatly exceeds the typical context window of 32,000 - 128,000 tokens. You don't need to do anything extra to access the larger context window of the Gemini API.

Define question

```python
question = "How much energy do mitochondria make in a day?"
```

The `generate_content` function can be used to prompt the model.

```python
response = model.generate_content([csv_file, system_instruction, question])
```

Model response can be accessed using the `text` attribute

```python
answer = response.text
```

Display output

```python
markdown_output = f"""
**Question:** {question}

**Answer:** {answer}
"""

display(Markdown(markdown_output))
```

You can inspect token usage through `usage_metadata`. Note that the prompt size tells you the model is using long context.

```python
response.usage_metadata
```

## Compare with baseline model

**Important**: The following code showcases how a longer context window unlocks new capabilities. Without the context, the model cannot be expected to answer the question. However, with the context, <u>Gemini 1.5</u> is able to answer by reasoning across the CSV provided in the prompt to unlock new abilities.

Define question

```python
question = "How many rubisco proteins in a leaf?"
```

Generate answer from baseline model with no context

```python
baseline_response = model.generate_content([question])
baseline_answer = baseline_response.text
```

Generate answer from long context model

```python
improved_response = model.generate_content([csv_file, system_instruction, question])
improved_answer = improved_response.text
```

Display responses

```python
markdown_output = f"""
**Question:** {question}

---

**Baseline Answer:**  
{baseline_answer}

---

**Long-Context Answer:**  
{improved_answer}
"""

display(Markdown(markdown_output))
```

# Reasoning with long context

Let's see if the model can tell us about itself.

```python
question = "What kind of information do you have in your long context and what kind of questions can you answer?"

response = model.generate_content([csv_file, system_instruction, question])
answer = response.text
markdown_output = f"""
**Question:** {question}

**Answer:** {answer}
"""

display(Markdown(markdown_output))
```

## Catching (human) errors

Another great use case is for catching errors in your calculation, demonstrating how long context enables the model to take on a copilot role for tough biological questions.

```python
question = """
Do you agree with my reasoning for the following question of how long would it take for a single E. coli cell to grow into a 1-gram colony under ideal conditions?

A: I'm going to assume we start with a single E. coli cell, that each bacterium is roughly 1mg and that it takes about 10 minute for the cell to double. So we would need 1000 cells to be 1 gram. Therefore the answer is how many replications does it take to double to 1000 cells from one cell. I think the way to do this is let x be that value and therefore: 2^{x-1}=1000. If you solve for x that gives you roughly 11. So it would take 11 replication times 10 minutes each so 110 minutes by my calculation.
"""

response = model.generate_content([csv_file, system_instruction, question])
answer = response.text
markdown_output = f"""
**Question:** {question}

**Answer:** {answer}
"""

display(Markdown(markdown_output))
```

## Clinical

Beyond the biology laboratory, it is useful to see how long context reasoning allows it to generalise to adjacent fields like the clinic.

```python
pdf_path = "/kaggle/input/bionumbers-dataset/prompts/case_report.pdf"
uploaded_pdf = genai.upload_file(path=pdf_path)
```

Note the model doesn't always force the use of its context, which is actually a positive reflection on its discernment.

```python
question = "How much tuberculosis bacteria must have been present at time of transmission and at time of diagnosis?"

response = model.generate_content([csv_file, system_instruction, question, uploaded_pdf])
answer = response.text
markdown_output = f"""
**Question:** {question}

**Answer:** {answer}
"""

display(Markdown(markdown_output))
```

## Genomic

Genomic analysis is increasingly becoming an essential topic in biology.

```python
fasta_path = "/kaggle/input/bionumbers-dataset/prompts/chromosome21_5093499-5094536.fa"

with open(fasta_path, 'r') as file:
    genome_segment = file.read().replace('\n', '')

print(genome_segment)
```

```python
question = "How many mutations will be in this DNA after 10 germline generations?"

response = model.generate_content([csv_file, system_instruction, genome_segment, question])
answer = response.text
markdown_output = f"""
**Question:** {question}

**Answer:** {answer}
"""

display(Markdown(markdown_output))
```

# Multimodal prompting

## Image

One of the strengths of long context using the Gemini 1.5 family is that it is able to natively understand multimodal inputs. Here, we demonstrate how it can recognise and reason across images.

```python
image_path = "/kaggle/input/bionumbers-dataset/prompts/cells.jpg"
uploaded_image = genai.upload_file(path=image_path)
Image(image_path)
```

```python
question = "How much energy do the cells in the image consume every hour?"

response = model.generate_content([csv_file, system_instruction, uploaded_image, question])
answer = response.text
markdown_output = f"""
**Question:** {question}

**Answer:** {answer}
"""

display(Markdown(markdown_output))
```

## Video

In a world where video is becoming a dominant media type, you can be reassured that the model can use video to prompt deeper reasoning from its large context window.

```python
video_path = "/kaggle/input/bionumbers-dataset/prompts/yeast.mp4"
uploaded_video = genai.upload_file(path=video_path)
Video(video_path, embed=True)
```

```python
question = "Estimate the total amount of CO2 produced by the yeast throughout the course of the video."

response = model.generate_content([csv_file, system_instruction, uploaded_video, question])
answer = response.text
markdown_output = f"""
**Question:** {question}

**Answer:** {answer}
"""

display(Markdown(markdown_output))
```

# Using context caching

Context caching is great for when need to ask a number of questions to the same context. In this case, all our questions are directed towards the exact same spreadsheet. Contet caching assists with this case, and can be more efficient by avoiding the need to pass the same tokens through the model for each new request. 

We already have the context downloaded to this notebook, so we will start by creating a `CachedContent` object to specify the prompt we want to use. This includes theh file and other fields you want to cache. We can also provide it with the same `system_instruction` that we have been using throughout this notebook.

```python
bio_cache = caching.CachedContent.create(
    model=MODEL_NAME,
    system_instruction=(
        system_instruction
    ),
    contents=[csv_file],
)

bio_cache
```

Once we have a `CachedContent` object, we can update the expiry time to keep it alive while we need it.

```python
bio_cache.update(ttl=datetime.timedelta(hours=1))
bio_cache
```

Next, we initialize a `GenerativeModel` using `from_cached_content`.

```python
bio_model = genai.GenerativeModel.from_cached_content(cached_content=bio_cache)
```

Then, we can generate content as you would with a directly instantiated model object (as we have been using previously).

```python
question = "What's the speed of neuronal development during the growth of a fruit fly from larva to adult?"

response = bio_model.generate_content([(question)])
answer = response.text

markdown_output = f"""
**Question:** {question}

**Answer:** {answer}
"""

display(Markdown(markdown_output))
```

## Deleting the cache

The cache has a small recurring storage cost so by default it is only saved for an hour. 

Even so, if you don't need the cache anymore, it is good practice to delete it proactively.

```python
print(bio_cache.name)
bio_cache.delete()
```

# Acknowledgements

- [BioNumbers](https://bionumbers.hms.harvard.edu/aboutus.aspx) team and all its contributors must be acknowledged. Without their dedication to biological numeracy, this dataset would not exist.
- Ron Milo and Rob Phillips for their book [_CELL BIOLOGY BY THE NUMBERS_](https://book.bionumbers.org/) which is an incredibly practical guide to biological numeracy.
- The `google-gemini/cookbook` team which created very helpful [quickstart guides](https://github.com/google-gemini/cookbook/tree/main/quickstarts) for using the Gemini API.
- Laura Deming's blog [post](https://ldeming.posthaven.com/understanding-biology-quickly) for showing me the value of thought experiments in biology

# References

- Milo, R., Jorgensen, P., Moran, U., Weber, G., & Springer, M. (2010). BioNumbers--the database of key numbers in molecular and cell biology. Nucleic Acids Research, 38(Database issue), D750–D753. https://doi.org/10.1093/nar/gkp889
- Benaissa, E., Bahalou, M. H., Safi, Y., Bssaibis, F., Benlahlou, Y., Chadli, M., Maleb, A., & Elouennass, M. (2021). Primary tuberculosis of the parotid gland: A forgotten diagnosis about a case! Clinical Case Reports, 9(5), e03954. https://doi.org/10.1002/ccr3.3954
- bccoer. (n.d.). Stoma and guard cells in succulent xerophyte leaf [Image]. Marked with CC0 1.0. Retrieved from https://creativecommons.org/publicdomain/zero/1.0/?ref=openverse
- Muschall, R. (2010, May 12). Yeast growth [Video]. YouTube. https://www.youtube.com/watch?v=hJyFGYPyHbY
- Genome Reference Consortium. (2013). GRCh38: Genome Reference Consortium Human Build 38. NCBI Assembly Database. Retrieved from https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.26/

# Next Steps

## Useful API references:
You can find the official documentation on long context [here](https://ai.google.dev/gemini-api/docs/long-context). For more information about context caching, check its [API reference](https://ai.google.dev/gemini-api/docs/caching).
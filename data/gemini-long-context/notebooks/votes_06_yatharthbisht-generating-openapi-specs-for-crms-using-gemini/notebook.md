# Generating Openapi specs for CRMs using Gemini

- **Author:** Yatharth bisht
- **Votes:** 73
- **Ref:** yatharthbisht/generating-openapi-specs-for-crms-using-gemini
- **URL:** https://www.kaggle.com/code/yatharthbisht/generating-openapi-specs-for-crms-using-gemini
- **Last run:** 2024-11-30 23:19:53.417000

---

## Preface <a class="anchor"  id="chapter1"></a>

**A brief overview of my work**



Below in the flowchart, you will find the overall workflow of my notebook, and how I utilized Gemini 1.5's various features to achieve my goal of fully automating the generation of OpenAPI specs for CRMs **just from the user's prompts!!**


<a href="https://ibb.co/C9Bt5vG"><img src="https://i.ibb.co/vd1Bxmn/Openapi-spec-generator-final-graph-1-900.png" alt="Openapi-spec-generator-final-graph-1-900" border="0"></a>

## Introduction <a class="anchor"  id="chapter1"></a>

***Context has come a "long" way with Gemini 1.5 🌌✨***



Conventional LLMs have been utilizing about a 100k tokens for way too long. That might sound sufficient, but let's be honest, not anymore. With data growing every second at an exponentially large rate , it only makes sense that the **abillity of LLMs to "memorize" that data** should grow too🧠; and that is exactly what Gemini 1.5 brings to the table.

Due to the context cap of its predecessors and other LLMs, concepts like RAG(Retrieval Augmented Generation) that utilize a vector database to store a corpus of information were relied on extensively, but not anymore! 

With Gemini 1.5's whopping 2 million max token capacity, who even needs a third party vector store? When all the relevant context can directly be saved in memory!!📚

You can see below exactly how significant of a milestone this is , where a comparison is drawn between conventional LLMs and Gemini 1.5:



<a href="https://ibb.co/JnPVGcr"><img src="https://i.ibb.co/wy1VPdp/Screenshot-2024-11-30-203818.png" alt="Screenshot-2024-11-30-203818" border="0"></a>



{[reference](https://www.youtube.com/watch?v=WCw1xBREoWw)}



And that is exactly what I aim to explore through this notebook! I am determined to squeeze out every last bit of utillity that the 2 million token capacity provides and use it on a very novel and essential usecase.

********************************************************************************

## Part 1: Exploring Online Documentation Q&A using Gemini's Huge Context Window and the Concept of Context Caching <a class="anchor"  id="chapter1"></a>

 

In the first part of this notebook, we explore how to leverage Gemini's long-context feature, which allows prompting with up to 2 million tokens, to ask questions about virtually any documentation available online. 🌐



This capability is particularly valuable as many online documentations lack an integrated AI assistant to address general queries. ❓ With Gemini's vast context window, we can directly embed large portions of documentation into the prompt, enabling users to extract meaningful answers without needing to navigate complex pages manually. 🧠✨



Additionally, by utilizing context caching, we can efficiently manage and process this vast amount of information. Context caching ensures that frequently accessed sections of the documentation are readily available, reducing processing times and enhancing the user experience. ⏩



This approach offers a unique advantage by:



Simplifying Navigation: No need to scroll through lengthy documents for answers. 📜



Enhancing Precision: Ensuring that specific, contextually relevant answers are retrieved. 🎯✅



Saving Time: Quick access to accurate information, even in large and detailed documentations. ⏳💡



Gemini's long-context capability, combined with context caching, opens new possibilities for online documentation Q&A, making technical and non-technical information far more accessible and user-friendly. 🚀🔍

```python
###Below is the implementation of a simple webcrawler , that is responsible for extracting the URLs of literally every page of that webpage





import urllib.request

import urllib.error

from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup

from queue import Queue

import json

class SimpleCrawler:

    def __init__(self, root_url):

        self.root_url = root_url.rstrip("/")

        self.base_host = urlparse(root_url).netloc

        self.visited = set()

        self.to_visit = Queue()

        self.to_visit.put(root_url)

        self.crawler_links = {}

        self.modified_urls = []



    def crawl(self):

        while not self.to_visit.empty():

            url = self.to_visit.get()

            if url in self.visited:

                continue

            reduced_url = url.replace(self.root_url, "", 1) or "/"

            self.modified_urls.append(reduced_url)

            #print(f"Crawling: {url}")

            self.crawler_links[url] = []

            self.visited.add(url)

            links = self.fetch_links(url)

            for link in links:

                if urlparse(link).netloc == self.base_host and link.startswith(url):

                    self.crawler_links[url].append(link)

                    #print(f"  Found relevant link: {link}")

                    if link not in self.visited:

                        self.to_visit.put(link)



    def fetch_links(self, url):

        try:

            response = urllib.request.urlopen(url)

            if response.info().get_content_type() != "text/html":

                return []

            html_content = response.read().decode("utf-8")

            soup = BeautifulSoup(html_content, "html.parser")

            links = [urljoin(url, a.get("href")) for a in soup.find_all("a", href=True)]

            return links

        except urllib.error.URLError as e:

            print(f"Error fetching {url}: {e}")

            return []



    def save_results(self):

        with open("crawler_with_found_links_new_.json", "w") as f:

            json.dump(self.crawler_links, f, indent=4)

        print("Saved crawler_with_found_links_new.json")

        with open("crawler_only_urls_new.json", "w") as f:

            json.dump(self.modified_urls, f, indent=4)

        print("Saved crawler_only_urls_new.json")



def crawl_and_save_results(url):

    crawler = SimpleCrawler(url)

    crawler.crawl()

    crawler.save_results()

    return ("crawler_with_found_links_new.json", "crawler_only_urls_new.json")
```

**Reason for using openapi's documentation**





Using openapi's documentation not only serves as an example of how we can question whole documentations available online ,but it will also serve as a stepping stone towards building a proper openapi specefication, which is the main goal of the notebook .

```python
paths1, paths2 = crawl_and_save_results("https://learn.openapis.org/")

print("Generated files")
```

```python
import json

with open('/kaggle/working/crawler_with_found_links_new_.json', 'r') as file:

    data = json.load(file)

all_urls = []

for key, value in data.items():

    if isinstance(value, list):

        all_urls.extend(value)

#print(all_urls)
```

```python
all_urls = list(set(all_urls))

print(len(all_urls))
```

```python
##in the below method we are successfully extracting text from every webpage of a particular documentation



import nest_asyncio

import asyncio

import aiohttp

nest_asyncio.apply()



async def fetch_text(session, url):

    try:

        async with session.get(url) as response:

            response_text = await response.text()

            soup = BeautifulSoup(response_text, "html.parser")

            body_text = soup.find("body").get_text()

            return body_text.strip()

    except Exception as e:

        print(f"Error fetching {url}: {e}")

        return ""



async def get_text_from_urls(url_list):

    async with aiohttp.ClientSession() as session:

        tasks = [fetch_text(session, url) for url in url_list]

        return await asyncio.gather(*tasks)



url_list = all_urls

text_list = await get_text_from_urls(url_list)

#print(text_list)
```

```python
def write_list_to_file(strings, filename="output.txt"):

    try:

        with open(filename, "w") as f:

            for string in strings:

                f.write(string + "\n")  # Add a newline after each string



    except Exception as e:

        print(f"An error occurred: {e}")





my_strings = text_list

filename = "openapi_cache_documentation.txt"



try:

    write_list_to_file(my_strings, filename)

    print(f"Strings successfully written to {filename}")



except Exception as e:  # Catch any potential errors during file operations

    print(f"An error occurred while writing to the file: {e}")
```

```python
import os

import google.generativeai as genai

from google.generativeai import caching

import datetime

import time
```

```python
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("GEMINI_API_KEY")
```

```python
genai.configure(api_key=secret_value_0)
```

## Context Caching🗄️ <a class="anchor"  id="chapter1"></a>



Context Caching, as discussed before, is a feature provided by the Gemini API that provides the service of saving or "caching" a certain piece of context (usually large) so that it is not processed again and again, saving us a lot of money in the long run.



This cache lasts for a predefined amount of time, referred to as "Time To Live{TTL}"

during which we can make references to that text without having to worry about overhead💰.



In my implementation , I have used context caching in a way that processes and caches the entirety of the openAPI documentation , which comes to about over 400 thousand tokens . It would have been a financial nightmare to even consider repeatedly prompting all of these tokens, but thanks to context caching we only process it once. Once it has been added to cache we will proceed to ask it questions regarding literally anything . For all we know , we are asking questions to the smartest person who happens to be an OpenAPI expert 💫



The following flow represents the working of context caching , and how all the various components are processed:





<a href="https://ibb.co/K9C1QQZ"><img src="https://i.ibb.co/HrmJMMv/Screenshot-2024-11-30-203717.png" alt="Screenshot-2024-11-30-203717" border="0"></a>



{[reference](https://medium.com/google-cloud/vertex-ai-context-caching-with-gemini-189117418b67)}

```python
path_your_file = "/kaggle/input/context-for-openapi-generation-using-gemini/Long_Context_Openapi_Yatharth/openapi_cache_documentation.txt"

text_file = genai.upload_file(path=path_your_file)

while text_file.state.name == 'PROCESSING':

  print('Waiting for text file to be processed.')

  time.sleep(2)

  text_file = genai.get_file(text_file.name)



print(f'Text processing complete: {text_file.uri}')
```

```python
cache = caching.CachedContent.create(

    model='models/gemini-1.5-flash-001',

    display_name='openapi_documentation', # used to identify the cache

    system_instruction=(

        'You are a master of documentation analysis, and your job is to answer '

        'the user\'s query based on the documentaton text file you have access to.'

    ),

    contents=[text_file],

    ttl=datetime.timedelta(minutes=5),

)
```

```python
model = genai.GenerativeModel.from_cached_content(cached_content=cache)
```

**We will be using  the question answering facillity to extract some rules and regulations** that our openapi spec generation agents will be following in the later stages of this project

```python
response = model.generate_content([(

    "You need to generate a complete set of step by step instructions for an agent that specializes in generating an openapi specefication given 2 openapi specs, given that the resulting ,generated spec has the capabillities of performing actions of both the scripts given to it as input"

                  "You should also consider all the rules the agent should follow and the precautionary measures to take"

               "You should treat the output as direct instructions to the agent and format it accordingly. Please refrain from using unnecessary headers or footers ,titles ,etc")])

print(response.usage_metadata)
```

```python
merge=response.text

print(merge)
```

```python
regs= model.generate_content([(

            "Give my AI agent specializing in generating openapi specefications a detailed, step by step guide as to how it can generate an openapi spec given some json responses as context ,for a particular CRM and user preference"

              "You should also consider all the rules the agent should follow and the precautionary measures to take"

               "You should treat the output as direct instructions to the agent and format it accordingly. Please refrain from using unnecessary headers or footers ,titles ,etc")])

print(regs.usage_metadata)

print(regs.text)
```

```python
final_instruct=regs.text
```

```python
print(final_instruct)
```

```python
corr= model.generate_content([(

            "Give my AI agent specializing in correcting and checking the format of openapi specefications a detailed list of precautions it sholud keep in mind , errors to look out for and some common mistakes"

              "You should also consider all of the script and look out for any mistakes "

               "You should treat the output as direct instructions to the agent and format it accordingly. Please refrain from using unnecessary headers or footers ,titles ,etc")])

print(corr.usage_metadata)

print(corr.text)
```

```python
corrections_conduct=corr.text
```

*Please note that the above 3 generated answers will serve as instructions for agents in the future as we dive deeper.*

*Feel free to ask any questions of your own in the cell below*

```python
qtion="Enter sample question here "

qtions = model.generate_content([(f"You have been asked the following question : {qtion} please answer it to the best of your abillity ")])

print (qtions.text)
```

**Why was context caching necessary?**



Notice how the prompt size in this scenario was extremely high (it almost crashed my notebook when I tried to print it😅). It not only would have been impossible to pass such a large prompt without Gemini's Long context facilities, it would be incredibly costly. In this scenario, since we are repeatedly prompting the same piece of information, the whole documentation was passed through only once, saving both time⏰ and money💸



In the next section we will discuss how we will utilize the retrieved information to the max!!!💫

## PART 2 Generating OpenAPI Specifications using Gemini 🚀 <a class="anchor"  id="chapter1"></a>



Using all the information about OpenAPI that we've gained so far from our online doc expert , it is now time to put that to good use. The objective of this whole agentic framework in the first place was to generate directly useable openAPI specefications, which will be done using numerous agents working with each other , powered by Gemini.



***Motivation*** 📖



In the digital world APIs are as crucial as the air we breathe or the water we drink, enabling seamless integrations between applications. OpenAPI specs are critical for documenting those essential APIs, providing a standardized , machine-readable format that defines API endpoints, methods, request/response structures, and much more. Howeverm , manually creating OpenAPI specs from extensive API documentation is a time-consuming venture. There are multiple agents available on the internet that RETRIEVE information from OpenAPI specs, but not so much that cater to their generation. This is where my project comes in , saving businesses their valuable time⏳ and money💰.



***Why OpenAPI*** ⚠️



API documentation for platforms like CRMs (Customer Relationship Management tools) often spans thousands of lines of text, detailing numerous endpoints, parameters, and workflows. Manually converting this documentation into OpenAPI specs demands significant developer time and expertise. Moreover, inconsistencies in documentation formats and the sheer volume of text increase the likelihood of errors, delays, and missed opportunities for automation. These inefficiencies translate into slow adoption of APIs in critical workflows.



<a href="https://ibb.co/KmN6gGQ"><img src="https://i.ibb.co/1bsZNnp/Screenshot-2024-11-30-201836.png" alt="Screenshot-2024-11-30-201836" border="0"></a>



{[reference](https://swagger.io/docs/specification/v3_0/basic-structure/)}



***How Gemini’s Long Context Window Solves the Problem*** ✨



My project aims to leverage Gemini 1.5's groundbreaking context window not only to handle an entire documentation at once, but later on, as you will see, handle entire sitemaps and numerous examples as well . All of which add to the credibillity of this venture.💯

This collection of agents can generate precise, ready-to-use OpenAPI specs in YAML or JSON format. This eliminates the manual effort involved in parsing and converting documentation, reducing errors and accelerating API integration. ✅

```python
user_pref="I want to build a openapi spec, using copper developer api for fetching a lead by ID, creating a new lead and create people in BULK " ## only used as reference ,feel free to experiment  

link="https://developer.copper.com/"
```

```python
paths1, paths2 = crawl_and_save_results(link)

print("Generated files")
```

```python
with open("/kaggle/working/crawler_with_found_links_new_.json", 'r') as file:

    data = json.load(file)



all_urls = []

for key, value in data.items():

    if isinstance(value, list):

        all_urls.extend(value)

all_urls = list(set(all_urls))#for unique entries only

#print(all_urls)
```

```python
len(all_urls)
```

## Part 2(a): Comparing RAG (Retrieval Augmented Generation) based response with In-Context Generation <a class="anchor"  id="subsection1"></a>



Ever since Gemini 1.5's introduction of large context windows, there has been pressing concerns regarding RAG(Retrieval Augmented Generation)🌐. Is it really true that in-context retrieval , (a technique only possible with an LLM that has a substantially large memory) renders RAG obsolete📊? Let us find that out in the following section , where we give both RAG and ICG the task for extracting relevant links from the sitemap of the whole documentation link (in this case, copper CRM) and see which performs better in terms of ease of use , accuracy of response and speed.🖥️



Below we can see an implementation of in-context retrieval , how the context is passed along with the prompt, directly to the LLM ; and recieves accurate retrieved response.



<a href="https://ibb.co/QDD1Vx5"><img src="https://i.ibb.co/LNN2ysG/Screenshot-2024-11-30-203627.png" alt="Screenshot-2024-11-30-203627" border="0"></a>



{[reference](https://aclanthology.org/2023.tacl-1.75.pdf)}

```python
!pip install llama_index

!pip install llama_index.embeddings.huggingface

!pip install llama_index.llms.gemini
```

```python
## we will be using a simple llama-index based agentic RAG as our contender for team RAGs



from llama_index.core import (

    VectorStoreIndex,

    SimpleDirectoryReader,

    StorageContext,

    Settings,

    load_index_from_storage

)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.node_parser import SentenceSplitter

from llama_index.llms.gemini import Gemini

import os

import warnings

warnings.filterwarnings('ignore')

import os

from dotenv import load_dotenv



load_dotenv()





def main_match(user_req):



    file_path = "/kaggle/working/crawler_only_urls_new.json"



    Settings.llm  = Gemini(model="models/gemini-1.5-flash", api_key=secret_value_0)

    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)



    reader = SimpleDirectoryReader(input_files=[file_path])

    documents = reader.load_data()

    nodes = Settings.node_parser.get_nodes_from_documents(documents, show_progress=True)



    vector_index = VectorStoreIndex.from_documents(documents, node_parser=nodes)

    vector_index.storage_context.persist(persist_dir="storage_mini")

    storage_context = StorageContext.from_defaults(persist_dir="storage_mini")

    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()



    q6 = f"""

    You are a system that is adept at evaluating and retrieving links from a json file,

    and those links will be retrieved on the basis of the user's specifications and requests,

    i.e., you will receive a user request (for the purpose of making an openapi spec) and you will evaluate that request,

    each word of it. Then, you will use this information to retrieve possible matches from a json file containing links from specific documentation which might contain the information the user is looking for.

    The user request is as follows: {user_req}

    YOUR FINAL OUTPUT SHOULD ONLY CONTAIN A LIST OF THE MATCHES AND NOTHING ELSE, ABSOLUTELY NOTHING ELSE, AND THESE MATCHES SHOULD BE TRACEABLE AND FOUND DIRECTLY, AS IS, IN THE GIVEN JSON FILE. MAKE SURE TO ADHERE TO THESE INSTRUCTIONS.

    """

    resp6 = query_engine.query(q6)

    print(resp6)

    print(type(resp6))

    return resp6



output= main_match("I want to build a openapi spec, using copper developer api for fetching a lead by ID, creating a new lead and create people in BULK")

list_resp6 = [item.strip() for item in str(output).strip("[]").split("\n")]



print(list_resp6)

#print(str(output))
```

```python
#print(type(list_resp6[len(list_resp6)-3]))

new_list = eval(list_resp6[len(list_resp6)-3])

#print(new_list)
```

```python
def remove_duplicates(input_list):

    seen = set()

    result = []

    for item in input_list:

        if item not in seen:

            result.append(item)

            seen.add(item)

    return result

def get_links_for_modified_urls(modified_urls, root_url, filename="crawler_with_found_links_new_.json"):



    file_path = "/kaggle/working/crawler_with_found_links_new_.json"

    with open(file_path, "r") as f:

        crawler_data = json.load(f)



    all_found_links = []

    for url in modified_urls:

        full_url = root_url.rstrip("/") + url

        if full_url in crawler_data:

            all_found_links.extend([full_url] + crawler_data[full_url])

        else:

            print(f"Error: {full_url} not found in the data.")



    all_found_links = remove_duplicates(all_found_links)

    print("length of total links : " , len(all_found_links))



    print (all_found_links)

    return all_found_links

RAG_links=get_links_for_modified_urls(new_list,"https://developer.copper.com/")

print(RAG_links) ## the final output acquired through RAG
```

```python
## now for in context retrieval by gemini derived links

print("content")

def join_strings(strings):

    return '\n'.join(strings)



# Example usage

strings = all_urls

corpus = join_strings(strings)

#print(corpus)
```

```python
#The above  output will be sent to gemini directly as prompt in order for it to identify all the relevant links

genai.configure(api_key=secret_value_0)



# Create the model

generation_config = {

  "temperature": 1,

  "top_p": 0.95,

  "top_k": 40,

  "max_output_tokens": 8192,

  "response_mime_type": "text/plain",

}



model = genai.GenerativeModel(

  model_name="gemini-1.5-pro",

  generation_config=generation_config,

  system_instruction="You are an excellent and unbiased text parser. You can efficiently and effectively extract smaller pieces of text from a large corpus , given the prompt for it. Your task here would be to extract links related to a particular prompt and return them. ",

)



chat_session = model.start_chat(

  history=[

    {

      "role": "user",

      "parts": [

          corpus

      ],

    },

  ]

)



response = chat_session.send_message(f"from the above given URLs, extract the URLs that might contain the contexts related to the following user specefication : {user_pref} \nPLEASE NOTE THAT YOU SHOULD RETURN AS MANY LINKS AS POSSIBLE, GIVEN THAT THEY ARE CONTEXTUALLY RELEVENT , AND MOST OF ALL DO NOT AT ANY COST RETURN ANYTHING APART FROM THE URLs THEMSELVES\n",

)

#print(response.text)
```

```python
print(response.usage_metadata)
```

```python
ICR_corpus=response.text

# ICR stands for in context retrieval



ICR_links = ICR_corpus.splitlines()

ICR_links.pop()

print(ICR_links)
```

In-Context Generation proved to be significantly easier and more user-friendly. It required no integration with external databases or APIs, did'nt involve installing additional dependencies , and took less time since it did not have to create a vector store; streamlining the process and saving valuable time ⏳. With Gemini's expanded context window, ICG could maintain a high fidelity to the provided data, ensuring that minimal information was lost. 🛡️

```python
### now that we have recieved all the links from which we need to extract text, we will proceed with the ICR_links
```

```python
## extracting text from all the relevant web-pages ,which will serve as the corpus from which we derive our context from

import nest_asyncio

import asyncio

import aiohttp

from bs4 import BeautifulSoup

nest_asyncio.apply()



async def fetch_text(session, url):

    try:

        async with session.get(url) as response:

            response_text = await response.text()

            soup = BeautifulSoup(response_text, "html.parser")

            body_text = soup.find("body").get_text()

            return body_text.strip()

    except Exception as e:

        print(f"Error fetching {url}: {e}")

        return ""



async def get_text_from_urls(url_list):

    async with aiohttp.ClientSession() as session:

        tasks = [fetch_text(session, url) for url in url_list]

        return await asyncio.gather(*tasks)



url_list = ICR_links

text_list = await get_text_from_urls(url_list)

#print(text_list)
```

```python
### The agent below is responsible for extracting JSON/HTTP 200 response from the text corpus, which will help us identify and extract all the relevant data(in this case json response) which will be further used as context for spec building



genai.configure(api_key=secret_value_0)

def extract_https200_codes(text_input):

    generation_config = {

    "temperature": 1,

    "top_p": 0.95,

    "top_k": 40,

    "max_output_tokens": 8192,

    "response_mime_type": "text/plain",

    }



    model = genai.GenerativeModel(

    model_name="gemini-1.5-flash",

    generation_config=generation_config,

    system_instruction=

                        """

                    "You are a content extraction assistant tasked with identifying and extracting JSON response codes from large blocks of text. "

                    "Given a block of text, your role is to:\n\n"

                    "- Extract only the JSON response codes mentioned in the text.\n\n"

                    "Your output should consist solely of the JSON response codes found, with no additional text, explanations, or formatting and not any headings / headers like 'here's what i found'etc.\n\n"

                    "If no JSON response codes are found in the text, simply return 'NOT_FOUND' and nothing else."

                    "for some context , the desired output may look something like : "



                                        {

                      "people": [

                        {

                          "name": "My Contact",

                          "emails": [

                            {

                            "email": "mycontact_1233@noemail.com",

                            "category": "work"

                            }

                          ],

                          "address": {

                            "street": "123 Main Street",

                            "city": "Savannah",

                            "state": "Georgia",

                            "postal_code": "31410",

                            "country": "United States"

                          },

                          "phone_numbers": [

                            {

                            "number": "415-123-45678",

                            "category": "mobile"

                            }

                          ]

                        }

                      ]

                    }



                    "NOTICE ONLY THE FORMAT AND NOT THE CONTENT OF TEXT "

                    "MAKE SURE ALL AND EVERY BIT OF TEXT THAT IS REMOTELY RELATED TO JSON FORMAT IS RETRIEVED FROM THE ENTIRETY OF THE TEXT AND NOTHING IS LEFT OUT "

                    """,

    )



    chat_session = model.start_chat(

    history=[



    ]

    )

    a= f"Analyze the following text and return only the HTTPS 200 codes and the JSON response codes:\n\n{text_input}"



    response = chat_session.send_message(a)

    return response.text
```

```python
def filter_list(input_list):

    keywords = ["YOUR_TOKEN_HERE", "NOT_FOUND"]

    return [item for item in input_list if not any(keyword in item for keyword in keywords)]

def remove_duplicates(input_list):

    seen = set()

    result = []

    for item in input_list:

        if item not in seen:

            result.append(item)

            seen.add(item)

    return result

def jresp_list(text_list):

    http_codes=[]

    for i in text_list:

        http_codes.append(extract_https200_codes(i))

        time.sleep(5)

    http_codes= filter_list(http_codes)

    http_codes= remove_duplicates(http_codes)

    return http_codes



jresp=jresp_list(text_list)
```

```python
#print(type(jresp))
```

```python
### a similar agent to the above ,tasked with extracting CURL queries from the text corpus, which will be further converted to JSON responses in order to bolster our context and makr our agents more robust.

def extract_curl(text_input):

    generation_config = {

    "temperature": 1,

    "top_p": 0.95,

    "top_k": 40,

    "max_output_tokens": 8192,

    "response_mime_type": "text/plain",

    }



    model = genai.GenerativeModel(

    model_name="gemini-1.5-flash",

    generation_config=generation_config,

    system_instruction=

                        """

                    "You are a content extraction assistant tasked with identifying and extracting Curl commands from large blocks of text. "

                    "Given a block of text, your role is to:\n\n"

                    "- Extract only the Curl commands present in the text.\n\n"



                A good example of a Curl Command would be :

                curl --location --request POST "https://api.copper.com/developer_api/v1/leads/{{example_leadconvert_id}}/convert" \

                --header "X-PW-AccessToken: YOUR_TOKEN_HERE" \

                --header "X-PW-Application: developer_api" \

                --header "X-PW-UserEmail: YOUR_EMAIL_HERE" \

                --header "Content-Type: application/json" \

                --data "{

                \"details\":{

                    \"person\":{

                    \"name\":\"John Doe\"

                    },

                    \"opportunity\":{

                    \"name\":\"Demo Project\",

                    \"pipeline_id\":213214,

                    \"pipeline_stage_id\":12345,

                    \"monetary_value\":1000

                    }

                }

                }"





                    "Your output should consist solely of the Curl commands found, with no additional text, explanations, or formatting and not any headings / headers like 'here's what i found'etc\n\n"

                    "If no Curl commands are found in the text, simply return 'NOT_FOUND' and nothing else."

                    "MAKE SURE ALL AND EVERY BIT OF TEXT THAT IS REMOTELY RELATED TO CURL QUERY IS RETRIEVED FROM THE ENTIRETY OF THE TEXT AND NOTHING IS LEFT OUT "

                    """,

    )



    chat_session = model.start_chat(

    history=[



    ]

    )

    a=f"Analyze the following text and return only the Curl commands:\n\n{text_input}"



    response = chat_session.send_message(a)

    return response.text

def filter_list(input_list):

    keywords = ["NOT_FOUND"]

    return [item for item in input_list if not any(keyword in item for keyword in keywords)]

def remove_duplicates(input_list):

    seen = set()

    result = []

    for item in input_list:

        if item not in seen:

            result.append(item)

            seen.add(item)

    return result

def curlresp_list(text_list):

    curl_codes=[]

    for i in text_list:

        curl_codes.append(extract_curl(i))

        time.sleep(5)

    curl_codes= filter_list(curl_codes)

    curl_codes= remove_duplicates(curl_codes)

    return curl_codes



curl=curlresp_list(text_list)
```

```python
# Simple agent responsible for the conversion of CURL queries to JSON Response

def convert_curl_to_http_response(curl_command):

    generation_config = {

    "temperature": 1,

    "top_p": 0.95,

    "top_k": 40,

    "max_output_tokens": 8192,

    "response_mime_type": "text/plain",

    }



    model = genai.GenerativeModel(

    model_name="gemini-1.5-flash",

    generation_config=generation_config,

    system_instruction=



"""You are a JSON response generation agent designed to interpret and respond to curl API requests. Your task is to analyze each curl query, understand its purpose, and craft an appropriate JSON response that the API might return. Your responses should be accurate representations of what the API would likely send back, including default values, typical metadata, and inferred details from the request body. Use placeholders (e.g., {example_id}) where specific IDs or timestamps might be variable.



                              When creating JSON responses, ensure they contain all relevant fields in a structured format. For example, if a lead conversion request includes details about a person and an opportunity, your response should reflect those entities fully, with fields like id, name, status, and relevant timestamps. Always infer and include logical values for fields not specified in the request."""

                              """

                              Example conversion :

                                 Given Curl Query :

                                 curl --location --request POST "https://api.copper.com/developer_api/v1/leads/{{example_leadconvert_id}}/convert" \

                                    --header "X-PW-AccessToken: YOUR_TOKEN_HERE" \

                                    --header "X-PW-Application: developer_api" \

                                    --header "X-PW-UserEmail: YOUR_EMAIL_HERE" \

                                    --header "Content-Type: application/json" \

                                    --data "{

                                    \"details\":{

                                        \"person\":{

                                        \"name\":\"John Doe\"

                                        },

                                        \"opportunity\":{

                                        \"name\":\"Demo Project\",

                                        \"pipeline_id\":213214,

                                        \"pipeline_stage_id\":12345,

                                        \"monetary_value\":1000

                                        }

                                    }

                                    }"

                             Generated JSON response:

                             {

                                "id": "{{example_leadconvert_id}}",

                                "status": "Converted",

                                "converted_to": {

                                    "person": {

                                    "id": 987654,

                                    "name": "John Doe",

                                    "first_name": "John",

                                    "last_name": "Doe",

                                    "email": null,

                                    "phone_numbers": [],

                                    "address": null,

                                    "socials": [],

                                    "tags": [],

                                    "custom_fields": [],

                                    "date_created": 1672158444,

                                    "date_modified": 1672158444

                                    },

                                    "opportunity": {

                                    "id": 123456,

                                    "name": "Demo Project",

                                    "pipeline_id": 213214,

                                    "pipeline_stage_id": 12345,

                                    "monetary_value": 1000,

                                    "status": "Open",

                                    "date_created": 1672158444,

                                    "date_modified": 1672158444,

                                    "close_date": null,

                                    "tags": [],

                                    "custom_fields": []

                                    }

                                },

                                "original_lead": {

                                    "id": "{{example_leadconvert_id}}",

                                    "name": "Original Lead Name",

                                    "status": "Converted",

                                    "customer_source_id": 331242,

                                    "date_created": 1672157444,

                                    "date_modified": 1672158444,

                                    "date_last_contacted": null

                                }

                              }





                        "MAKE ABSOLUTELY SURE THAT YOUR OUTPUT IS SYNTACTICALLY CORRECT ,ACCURATE AND COMPLETE.ANY AND ALL ERRORS ARE TO BE AVOIDED ENTIRELY"

                        "Your output should consist solely of the converted commands found, with no additional text, explanations, or formatting and not any headings / headers like 'here's what i found'etc\n\n"

                    """,

    )



    chat_session = model.start_chat(

    history=[



    ]

    )

    a=f"Convert the following Curl command to a JSON response format:\n\n{curl_command}"



    response = chat_session.send_message(a)

    return response.text



def convert(curl_codes):

    curl2http=[]

    for i in curl_codes:

        a=convert_curl_to_http_response(i)

        curl2http.append(a)

        time.sleep(5)

    return curl2http



converted=convert(curl)
```

```python
##COMBINED TEXT CORPUS : which will serve as main input to our baseline openapi generator

jresp.extend(converted)

combined_corpus="\n".join(jresp)

#print(combined_corpus)
```

## Part 2 {b}: Many-Shot Prompting <a class="anchor"  id="subsection1"></a>





Many-shot prompting is a concept derived from a recent research study conducted by Google. It is an extended version of few-shot prompting 🧠, where instead of just 5-6 rough examples we are providing our LLM with literally hundereds or thousands of such examples. This is also possible only because of a large context window, otherwise the main prompt would be lost among all the vast number of examples. ✨



In general, many-shot prompts can include hundreds or even thousands of examples. However, I explored a different approach here. 🚀



The dataset being used as "prompts" is sourced from the [APIs Guru OpenAPI directory](https://github.com/APIs-guru/openapi-directory).It consists of 10 pairs of JSON inputs and their corresponding OpenAPI responses, which are inherently large in size. 📂 These pairs serve as the example "shots" for the model. Just 10 pairs ? It might look like a small amount, but due to the nature of OpenAPI specs, they can be pretty huge, and pretty huge they are ! . The total token count of all these 10 examples combined is over a whopping 400 thousand!. This not only catches all the possible scenarios where an OpenAPI spec may vary, it also adds an element of coherence to it. 



To maintain the difference between actual instructions and examples ,I devised a strategy that involved not passing these examples or "shots" in the system prompt itself . Instead, these prompts are passed as responses in the history of the model , where the model assumes that these are outputs generated previously by the model itself , which promotes incremental learning and avoids cluttering the system prompt in a way that the original instruction is lost.



This strategy ensures that:



-->The system prompt remains focused, allowing the main instruction to stand out clearly. 🌟



-->The benefits of many-shot prompting are fully utilized, guiding the model with high-quality, contextual examples. 📊



-->Gemini’s long context window efficiently handles large datasets while maintaining accuracy and coherence. ✅



-->By leveraging this approach, the example pairs act as a powerful foundation, enabling the model to generate precise outputs while keeping the overall system design streamlined and effective. 🖥️✨

In the below graph we can clearly see just how much of an impact many shot prompting can have on output quality!



<a href="https://ibb.co/tJNQ3y8"><img src="https://i.ibb.co/CHXmWxh/Screenshot-2024-11-30-205421.png" alt="Screenshot-2024-11-30-205421" border="0"></a>





{[reference](https://arxiv.org/pdf/2404.11018)}

```python
import os

def extract_text_from_files(directory):

    text_list = []

    try:

        for file_name in os.listdir(directory):

            if file_name.endswith('.txt'):

                file_path = os.path.join(directory, file_name)

                with open(file_path, 'r', encoding='utf-8') as file:

                    content = file.read()

                    text_list.append(content)



        #print(f"Successfully extracted text from {len(text_list)} files.")

        return text_list



    except Exception as e:

        print(f"An error occurred: {e}")

        return []
```

```python
path = "/kaggle/input/context-for-openapi-generation-using-gemini/Long_Context_Openapi_Yatharth/long_ctxt_10/openapi_txt"

openapi_context = extract_text_from_files(path)

#print(openapi_context)
```

```python
path = "/kaggle/input/context-for-openapi-generation-using-gemini/Long_Context_Openapi_Yatharth/long_ctxt_10/json_txt"

json_context = extract_text_from_files(path)

#print(json_context)
```

```python
history_new=[

]



for i in range(0,10):

  user=    {

    "role": "user",

    "parts": [

      json_context[i],

    ],

  }

  history_new.append(user)

  model =     {

    "role": "model",

    "parts": [

        openapi_context[i],

    ],

  }

  history_new.append(model)  ##new history established , essentailly making the agent think it is already adept at the task it is assigned
```

```python
#print(history_new)
```

```python
## the agent below builds up our primary respone , i.e. the first iteration of the openapi specs that we will be generating.

def build_baseline_openapi_spec(combined_corpus, user_specifications):

    generation_config = {

    "temperature": 1,

    "top_p": 0.95,

    "top_k": 40,

    "max_output_tokens": 8192,

    "response_mime_type": "text/plain",

    }

    model = genai.GenerativeModel(

    model_name="gemini-1.5-flash",

    generation_config=generation_config,

    system_instruction=

                      f"""You are an assistant tasked with creating a baseline OpenAPI specification.

                              You will be provided with multiple JSON response objects (OpenAPI specifications) as context,

                              and user specifications outlining requirements for the API.

                              Your role is to:

                              - Analyze all provided scripts to understand the existing API functionalities.

                              - Identify common elements and best practices from the scripts.

                              - Incorporate the user specifications into the baseline OpenAPI spec in YAML format.

                              - Produce a baseline OpenAPI specification that includes the common elements.

                              - Make sure your output is syntactically correct and contextually viable, and should adhere to all the customer's demands from the provided scripts and meets the user's specifications.

                              "Follow the given instructions below to the letter in order to generate the best possible scenario":

                              {final_instruct}

                              Your output should be the baseline OpenAPI specification and nothing else, with no additional text, explanations, or formatting and not any headings / headers like 'here's what i found' etc."""

    )



    chat_session = model.start_chat(

    history=history_new

    )

    a=f"Here are the Json response scripts to use as context:\n\n{combined_corpus}\n\nUser specifications:\n\n{user_specifications}"



    response = chat_session.send_message(a)

    return response.text ,response.usage_metadata



zeroshot,tokens=build_baseline_openapi_spec(combined_corpus,user_pref)
```

```python
print(tokens)
```

**We can see how "many shot prompting" lives up to its name, as with an overwhelmingly large 400 thousand and over tokens it still manages to generate a coherent and accurate openapi specs!!**

```python
print(zeroshot)
```

```python
## The agent below is responsible for reducing hallucinations and mis-steps as much as possible, as it iterates through each and every element of context , further refining and confirming whether the openapi spec being generated really does contain everything the user asked for



def merge_http200_scripts(script1, script2):

    generation_config = {

    "temperature": 1,

    "top_p": 0.95,

    "top_k": 40,

    "max_output_tokens": 8192,

    "response_mime_type": "text/plain",

    }

    ###########################

    #### add merging instructions here later to improve prompt quality

    ###########################



    model = genai.GenerativeModel(

    model_name="gemini-1.5-flash",

    generation_config=generation_config,

    system_instruction=

                      f"""

                      "You are a content merging assistant tasked with combining two HTTP 200 scripts into a single, more elaborate OpenAPI specification. "

                    "Given two "scripts, your role is to:\n\n"

                    "- Understand all aspects of both scripts.\n"

                    "- Identify functionalities in script2 that are not explained or covered in script1.\n"

                    "- Feel free to ignore script 2 if it is not in accordance to the user defined preference for their openapi specefication , which is : {user_pref}, but please do it very judiciously and with care, because we dont want any loss of context"

                    "- Add those functionalities to script1, extending and elongating it appropriately.\n\n"

                    "The output should be a single, merged OpenAPI specification (HTTP 200 script) that combines the features of both scripts, "

                    "with all unique elements from script2 added to script1 where appropriate.\n\n"

                    "You need to follow the given instructions below to the letter and dont skip on anything:"

                    {merge}

                    "Your output should be the merged OpenAPI specification and nothing else.with no additional text, explanations, or formatting and not any headings / headers like 'here's what i found'etc\n\n"""""

    )



    chat_session = model.start_chat(

    history=[



    ]

    )

    a=f""" Merge the following two HTTP 200 scripts into a single, more elaborate OpenAPI specification:\n\n"

                    "Script 1:\n"

                    f"{script1}\n\n"

                    "Script 2:\n"

                    f"{script2}"

                    """



    response = chat_session.send_message(a)

    return response.text , response.usage_metadata

overall=zeroshot

for i in jresp:

    temp,tokenz=merge_http200_scripts(overall,i)

    overall=temp

    print(tokenz)

    time.sleep(5)

print ("done")
```

```python
##This is the final agent ,responsible for making sure that the output generated by the previous agent is syntactically correct or not.

##While making this project I came across numerous times when the final output seemed to contain everything but was not syntactically correct, one such response is used here as example . This ensures that the final output of this whole project is delivered without errors.

def corrections(spec):

    generation_config = {

    "temperature": 1,

    "top_p": 0.95,

    "top_k": 40,

    "max_output_tokens": 8192,

    "response_mime_type": "text/plain",

    }

   ##################

   ### add correction instructions here , dont add many shot example and explain why it was added in generation of zeroth sol but not here, it is primarily



    ### due to the fact that it ensures the original spec that is being corrected does not get lost with all the examples

#####################

    model = genai.GenerativeModel(

    model_name="gemini-1.5-pro",

    generation_config=generation_config,

    system_instruction=

                        f"""

                        Objective:\n"

                        "Your task is to review, correct, and refine OpenAPI specifications to ensure they are in a format "

                        "that is directly compatible with Postman and other API testing tools. Your goal is to produce a clean, "

                        "structured, and fully functional OpenAPI specification in YAML format, retaining all original content without any loss of information.\n\n"



                        "Follow the given instructions below to the letter while making a decision as to how to proceed :"

                        {corr}



                        "Final Deliverable:\n"

                        "The output should be a refined, fully compatible OpenAPI specification in YAML format that:\n"

                        "- Is immediately usable in Postman or similar tools without errors.\n"

                        "- Is clean, well-structured, and easy to read.\n"

                        "- Preserves all content from the original specification, without any loss or alteration of intent."



                        Heres some context for you to follow where you can Identify how an appropriate openapi spec should look like ,please learn from it :



                        Following is some context you should keep in mind :

                        openapi: "3.0.0"

                        info:

                        version: 1.0.0

                        title: Swagger Petstore

                        license:

                            name: MIT

                        servers:

                        - url: http://petstore.swagger.io/v1

                        paths:

                        /pets:

                            get:

                            summary: List all pets

                            operationId: listPets

                            tags:

                                - pets

                            parameters:

                                - name: limit

                                in: query

                                description: How many items to return at one time (max 100)

                                required: false

                                schema:

                                    type: integer

                                    format: int32

                            responses:

                                200:

                                description: An paged array of pets

                                headers:

                                    x-next:

                                    description: A link to the next page of responses

                                    schema:

                                        type: string

                                content:

                                    application/json:

                                    schema:

                                        $ref: "#/components/schemas/Pets"

                                default:

                                description: unexpected error

                                content:

                                    application/json:

                                    schema:

                                        $ref: "#/components/schemas/Error"

                            post:

                            summary: Create a pet

                            operationId: createPets

                            tags:

                                - pets

                            responses:

                                201:

                                description: Null response

                                default:

                                description: unexpected error

                                content:

                                    application/json:

                                    schema:

                                        $ref: "#/components/schemas/Error"

                        /pets/(petId):

                            get:

                            summary: Info for a specific pet

                            operationId: showPetById

                            tags:

                                - pets

                            parameters:

                                - name: petId

                                in: path

                                required: true

                                description: The id of the pet to retrieve

                                schema:

                                    type: string

                            responses:

                                200:

                                description: Expected response to a valid request

                                content:

                                    application/json:

                                    schema:

                                        $ref: "#/components/schemas/Pets"

                                default:

                                description: unexpected error

                                content:

                                    application/json:

                                    schema:

                                        $ref: "#/components/schemas/Error"

                        components:

                        schemas:

                            Pet:

                            required:

                                - id

                                - name

                            properties:

                                id:

                                type: integer

                                format: int64

                                name:

                                type: string

                                tag:

                                type: string

                            Pets:

                            type: array

                            items:

                                $ref: "#/components/schemas/Pet"

                            Error:

                            required:

                                - code

                                - message

                            properties:

                                code:

                                type: integer

                                format: int32

                                message:

                                type: string

                    """,

    )



    chat_session = model.start_chat(

    history=[

        {

        "role": "user",

        "parts": [

            """Correct the following OpenAPI specification and make sure to return only the corrected openapi spec and nothing else ,no headings or comments like ,  'heres what i found ':

                    The spec:

                    openapi: 3.0.3

                    info:

                    title: Pharmaceutical Shop API

                    description: API for managing products, orders, and customers in a pharmaceutical shop.

                    version: 1.0.0

                    servers:

                    - url: https://api.pharmashop.com/v1

                        description: Production server

                    - url: https://sandbox.api.pharmashop.com/v1

                        description: Sandbox server for testing

                    paths:

                    /products:

                        get:

                        summary: Retrieve a list of products

                        description: Fetches a list of all available pharmaceutical products.

                        responses:

                            '200':

                            description: A list of products.

                            content:

                                application/json:

                                schema:

                                    type: array

                                    items:

                                    $ref: '#/components/schemas/Product'

                        post:

                        summary: Add a new product

                        description: Adds a new pharmaceutical product to the inventory.

                        requestBody:

                            description: Product to add

                            required: true

                            content:

                            application/json:

                                schema:

                                $ref: '#/components/schemas/Product'

                        responses:

                            '201':

                            description: Product created successfully.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Product'

                    /products/{productId}:

                        get:

                        summary: Retrieve a product by ID

                        description: Fetches details of a specific product by its ID.

                        parameters:

                            - name: productId

                            in: path

                            required: true

                            description: ID of the product to retrieve

                            schema:

                                type: string

                        responses:

                            '200':

                            description: Product details retrieved successfully.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Product'

                            '404':

                            description: Product not found.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Error'

                        put:

                        summary: Update a product by ID

                        description: Updates the details of a specific product by its ID.

                        parameters:

                            - name: productId

                            in: path

                            required: true

                            description: ID of the product to update

                            schema:

                                type: string

                        requestBody:

                            description: Updated product information

                            required: true

                            content:

                            application/json:

                                schema:

                                $ref: '#/components/schemas/Product'

                        responses:

                            '200':

                            description: Product updated successfully.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Product'

                            '404':

                            description: Product not found.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Error'

                        delete:

                        summary: Delete a product by ID

                        description: Removes a specific product from the inventory by its ID.

                        parameters:

                            - name: productId

                            in: path

                            required: true

                            description: ID of the product to delete

                            schema:

                                type: string

                        responses:

                            '204':

                            description: Product deleted successfully.

                            '404':

                            description: Product not found.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Error'

                    /orders:

                        get:

                        summary: Retrieve a list of orders

                        description: Fetches a list of all customer orders.

                        responses:

                            '200':

                            description: A list of orders.

                            content:

                                application/json:

                                schema:

                                    type: array

                                    items:

                                    $ref: '#/components/schemas/Order'

                        post:

                        summary: Create a new order

                        description: Places a new order for products.

                        requestBody:

                            description: Order to create

                            required: true

                            content:

                            application/json:

                                schema:

                                $ref: '#/components/schemas/Order'

                        responses:

                            '201':

                            description: Order created successfully.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Order'

                    /orders/{orderId}:

                        get:

                        summary: Retrieve an order by ID

                        description: Fetches details of a specific order by its ID.

                        parameters:

                            - name: orderId

                            in: path

                            required: true

                            description: ID of the order to retrieve

                            schema:

                                type: string

                        responses:

                            '200':

                            description: Order details retrieved successfully.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Order'

                            '404':

                            description: Order not found.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Error'

                        put:

                        summary: Update an order by ID

                        description: Updates the details of a specific order by its ID.

                        parameters:

                            - name: orderId

                            in: path

                            required: true

                            description: ID of the order to update

                            schema:

                                type: string

                        requestBody:

                            description: Updated order information

                            required: true

                            content:

                            application/json:

                                schema:

                                $ref: '#/components/schemas/Order'

                        responses:

                            '200':

                            description: Order updated successfully.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Order'

                            '404':

                            description: Order not found.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Error'

                        delete:

                        summary: Delete an order by ID

                        description: Cancels a specific order by its ID.

                        parameters:

                            - name: orderId

                            in: path

                            required: true

                            description: ID of the order to delete

                            schema:

                                type: string

                        responses:

                            '204':

                            description: Order deleted successfully.

                            '404':

                            description: Order not found.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Error'

                    /customers:

                        get:

                        summary: Retrieve a list of customers

                        description: Fetches a list of all registered customers.

                        responses:

                            '200':

                            description: A list of customers.

                            content:

                                application/json:

                                schema:

                                    type: array

                                    items:

                                    $ref: '#/components/schemas/Customer'

                        post:

                        summary: Register a new customer

                        description: Adds a new customer to the system.

                        requestBody:

                            description: Customer to register

                            required: true

                            content:

                            application/json:

                                schema:

                                $ref: '#/components/schemas/Customer'

                        responses:

                            '201':

                            description: Customer registered successfully.

                            content:

                                application/json:

                                schema:

                                    $ref: '#/components/schemas/Customer'

                    /customers/{customerId}:

                        get:

                        summary: Retrieve a customer by ID

                        description: Fetches details of a specific customer by their ID.

                        parameters:

                            - name: customerId

                            in: path

                            required: true

                            description:

                    ::contentReference[oaicite:0]{index=0}





                    """,

        ],

        },

        {

        "role": "model",

        "parts": [

            """

                                openapi: "3.0.0"

                                info:

                                title: Pharmaceutical Shop API

                                version: 1.0.0

                                description: API for managing products, orders, and customers in a pharmaceutical shop.

                                servers:

                                - url: https://api.pharmashop.com/v1

                                    description: Production server

                                - url: https://sandbox.api.pharmashop.com/v1

                                    description: Sandbox server for testing

                                paths:

                                /products:

                                    get:

                                    summary: Retrieve a list of products

                                    operationId: listProducts

                                    tags:

                                        - Products

                                    responses:

                                        '200':

                                        description: A list of products.

                                        content:

                                            application/json:

                                            schema:

                                                type: array

                                                items:

                                                $ref: "#/components/schemas/Product"

                                        default:

                                        description: unexpected error

                                        content:

                                            application/json:

                                            schema:

                                                $ref: "#/components/schemas/Error"

                                    post:

                                    summary: Add a new product

                                    operationId: addProduct

                                    tags:

                                        - Products

                                    requestBody:

                                        description: Product to add

                                        required: true

                                        content:

                                        application/json:

                                            schema:

                                            $ref: "#/components/schemas/Product"

                                    responses:

                                        '201':

                                        description: Product created successfully.

                                        content:

                                            application/json:

                                            schema:

                                                $ref: "#/components/schemas/Product"

                                        default:

                                        description: unexpected error

                                        content:

                                            application/json:

                                            schema:

                                                $ref: "#/components/schemas/Error"

                                /products/{productId}:

                                    get:

                                    summary: Retrieve a product by ID

                                    operationId: getProductById

                                    tags:

                                        - Products

                                    parameters:

                                        - name: productId

                                        in: path

                                        required: true

                                        description: ID of the product to retrieve

                                        schema:

                                            type: string

                                    responses:

                                        '200':

                                        description: Product details retrieved successfully.

                                        content:

                                            application/json:

                                            schema:

                                                $ref: "#/components/schemas/Product"

                                        default:

                                        description: unexpected error

                                        content:

                                            application/json:

                                            schema:

                                                $ref: "#/components/schemas/Error"

                                components:

                                schemas:

                                    Product:

                                    type: object

                                    required:

                                        - id

                                        - name

                                        - price

                                    properties:

                                        id:

                                        type: string

                                        name:

                                        type: string

                                        price:

                                        type: number

                                        format: float

                                        description:

                                        type: string

                                        category:

                                        type: string

                                    Error:

                                    type: object

                                    required:

                                        - code

                                        - message

                                    properties:

                                        code:

                                        type: integer

                                        format: int32

                                        message:

                                        type: string

                                """,

        ],

        },

    ]

    )

    a=f"Correct the following OpenAPI specification; AND MAKE SURE YOUR RESPONSE DOES NOT CONTAIN ANYTHING APART FROM THE OPENAPI SPECEFICATION, NO HEADINGS , COMMENTS , FOOTNOTES , WHAT YOU DID ETC, SIMPLY THE YAML SCRIPT OF OPENAPI SPEC:\n{spec}\n please make sure that the spec you generate is in accordance with the user defined preference which is : {user_pref}"



    response = chat_session.send_message(a)

    return response.text, response.usage_metadata



final_output,tokens= corrections(overall)
```

## FINAL OUTPUT✨✨  <a class="anchor"  id="chapter1"></a>

```python
print(tokens)
```

```python
print(final_output)
```

## Conclusion  <a class="anchor"  id="chapter1"></a>

*We can see that we have now successfully generated openapi specs based on user preferences, ready to be put to use. Below we can see how it is accepted  in postman*



<a href="https://ibb.co/DWrKy1J"><img src="https://i.ibb.co/yP4s1N9/Screenshot-2024-11-30-205516.png" alt="Screenshot-2024-11-30-205516" border="0"></a>

## Future Scope and Prospects for OpenAPI Spec Generation Agent   <a class="anchor"  id="chapter1"></a>

**1. CRM Integration:** Expand support for popular CRMs (Salesforce, HubSpot, Zoho, etc.) with auto-detection of API endpoints and workflows, making spec generation more CRM-specific and efficient.



**2. Design-First API Development:** Introduce a visual design interface for collaborative, real-time API design, helping teams iterate on specifications before coding begins.



**3. Automated Testing and Validation:** Integrate with testing tools (e.g., ReadyAPI, Postman) to automatically generate and run tests from OpenAPI specs, ensuring consistency between documentation and live APIs.



**4. Code Generation & SDKs:** Enhance support for auto-generating server stubs, client SDKs, and boilerplate code in multiple languages (Java, Python, Node.js, etc.), speeding up development.



**5. AI-Powered API Optimization:** Leverage AI to suggest API design improvements, security enhancements, and performance optimizations based on best practices.



**6. Enhanced Collaboration:** Integrate with GitHub, GitLab, Jira, and Slack for version control, change tracking, and seamless teamwork on API specs.



**7. Microservices & Distributed Systems:** Support API documentation for microservices, service-to-service communication, and event-driven architectures, improving management of distributed systems.



**8. Real-Time Analytics:** Integrate usage analytics tools to track API performance, user behavior, and help identify areas for improvement.
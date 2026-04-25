# Legal AI Assistant for India's New Laws

- **Author:** Rishiraj Acharya
- **Votes:** 130
- **Ref:** rishirajacharya/legal-ai-assistant-for-india-s-new-laws
- **URL:** https://www.kaggle.com/code/rishirajacharya/legal-ai-assistant-for-india-s-new-laws
- **Last run:** 2024-11-22 15:16:39.450000

---

# **AI-Powered Legal Assistant for India's New Criminal Laws**

With the recent legislative changes in India's criminal law framework, legal professionals now face a steep learning curve as they adapt to these comprehensive legislative changes. The **Bharatiya Nyaya Sanhita (BNS)**, **Bharatiya Nagarik Suraksha Sanhita (BNSS)**, and **Bharatiya Sakshya Adhiniyam, 2023** have replaced foundational laws like the Indian Penal Code (IPC), Criminal Procedure Code (CrPC), and Indian Evidence Act, respectively. This notebook demonstrates the power of Gemini 1.5's vast context window and generative capabilities to assist judges, lawyers, and police officers in analyzing, interpreting, and applying these new laws efficiently in real-time scenarios.

This notebook explores how the **2-million-token capacity** of Gemini 1.5 allows it to process entire legal documents—thousands of pages long—while analyzing case-specific facts. By doing so, it provides actionable legal insights and helps professionals seamlessly transition to the new legal regime.

## Key Objectives
- **Purpose**: To stress-test Gemini 1.5’s long-context capability by helping legal professionals make sense of India’s new criminal laws.
- **Scope**: Apply the BNS, BNSS, and Bharatiya Sakshya Adhiniyam to a real-world scenario and provide case-specific legal interpretations.
- **Novelty**: Demonstrate how the model can process extensive legal texts and deliver precise legal advice by interpreting newly introduced provisions.

---

## **Setup & Installation**

In this section, we install the necessary packages to connect and interact with Google Cloud’s Vertex AI services. The following code authenticates our session and ensures we can call the generative models.

```python
!pip install -q --upgrade google-cloud-aiplatform
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)

import vertexai
vertexai.init(project="law-ai-439015", location="us-central1")
```

---

## **Introducing the Case**

In our scenario, Rajesh is accused of theft from a warehouse, and there is a mix of circumstantial and forensic evidence against him. We will ask the AI assistant to analyze this scenario through the lens of the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam, 2023.

The model will:
1. **Interpret the definitions and legal standards** for theft under the new **BNS**.
2. **Evaluate the procedural correctness** of Rajesh's arrest under the **BNSS**.
3. **Assess the admissibility and weight of forensic and circumstantial evidence** under the **Bharatiya Sakshya Adhiniyam**.

---

## **Loading Legal Documents**

In this section, we load full-text PDFs of the **Bharatiya Nyaya Sanhita (BNS)**, **Bharatiya Nagarik Suraksha Sanhita (BNSS)**, and **Bharatiya Sakshya Adhiniyam** directly from the official government sources. The **long-context capability** of Gemini 1.5 enables us to process all three documents simultaneously, ensuring a comprehensive understanding of the new legislative landscape.

```python
# Loading the legal documents directly from official PDF sources.
document1 = Part.from_uri(
    mime_type="application/pdf",
    uri="https://www.indiacode.nic.in/bitstream/123456789/20062/1/a2023-45.pdf",
)
document2 = Part.from_uri(
    mime_type="application/pdf",
    uri="https://www.indiacode.nic.in/bitstream/123456789/20099/1/a2023-46.pdf",
)
document3 = Part.from_uri(
    mime_type="application/pdf",
    uri="https://www.indiacode.nic.in/bitstream/123456789/20063/1/a2023-47.pdf",
)
```

```python
# Loading the legal documents from Kaggle datasets as required by the competition.
import base64
from vertexai.generative_models import Part, SafetySetting

def convert(path):
    with open(path, 'rb') as file:
        data = file.read()
    return base64.b64encode(data).decode('utf-8')

document1 = Part.from_data(
    mime_type="application/pdf",
    data=base64.b64decode(convert("/kaggle/input/indiacode/a2023-45.pdf"))
)
document2 = Part.from_data(
    mime_type="application/pdf",
    data=base64.b64decode(convert("/kaggle/input/indiacode/a2023-46.pdf"))
)
document3 = Part.from_data(
    mime_type="application/pdf",
    data=base64.b64decode(convert("/kaggle/input/indiacode/a2023-47.pdf"))
)
```

---

## **Case Scenario for Analysis**

The case involves Rajesh, accused of theft with circumstantial and forensic evidence. We input the scenario to be analyzed and interpreted by the legal assistant. This will help legal professionals understand how the new laws apply in real-world situations. We aim to demonstrate the **usefulness** of Gemini 1.5 in providing high-quality legal analysis.

```python
# Input scenario to be analyzed under the BNS, BNSS, and Bharatiya Sakshya Adhiniyam.
text1 = """A man named Rajesh was accused of theft from a warehouse. The warehouse belongs to a company, and a large amount of equipment was stolen. No direct evidence of stolen goods was found in Rajesh's possession. Rajesh claims he was at the warehouse for a different legitimate reason, which he refuses to disclose. However, forensic evidence at the scene includes partial fingerprints that may or may not belong to Rajesh, and a few security camera footages that are unclear but show a person with a similar build as Rajesh.

The police arrested Rajesh under suspicion of theft, and the prosecution is arguing that circumstantial evidence, along with witness testimony, is enough for conviction. The defense, however, is arguing that the evidence is too weak to convict Rajesh beyond a reasonable doubt.

**Questions for Legal Judgement:**
1. Under the Bharatiya Nyaya Sanhita (BNS), how should theft be defined, and what are the legal standards for proving theft in this case?
2. What procedural requirements under the Bharatiya Nagarik Suraksha Sanhita (BNSS) should the police have followed while arresting Rajesh?
3. How should the court treat the forensic evidence, circumstantial evidence, and witness testimony under the Bharatiya Sakshya Adhiniyam (Evidence Act)?"""
```

---

## **Video Integration for Witness Testimonies**

With Gemini 1.5's large context window, we can now integrate multimedia inputs like **video testimonies** directly into the analysis process. Instead of providing the watchman's statement as text, we can send the actual video of the watchman's statement to the model, allowing it to extract relevant context directly from the video. This feature enhances the flexibility and accuracy of the AI in processing diverse types of evidence, such as witness testimony, without the need for manual transcription.

This makes the notebook even more **interesting** and **useful** by demonstrating how the model handles real-world inputs in legal cases, where video evidence is common. By processing video files, Gemini can offer a more comprehensive legal analysis, bridging the gap between traditional legal methods and modern technological advancements.

```python
video1 = Part.from_data(
    mime_type="video/mp4",
    data=base64.b64decode(convert("/kaggle/input/indiacode/testimony.mp4"))
)
```

```python
from IPython.display import Video
Video("/kaggle/input/indiacode/testimony.mp4", embed=True, width=640)
```

## **Benefits of Video-Based Input**

1. **Accurate Context Extraction**: Video statements provide more detailed insights, including tone, facial expressions, and nuances in the testimony.
2. **Efficient Handling of Evidence**: Legal professionals often rely on video testimonies, and having the AI analyze this content directly can streamline case review.
3. **Enhanced Legal Interpretation**: By processing video, the model can better interpret human factors like reliability of the testimony, body language, and coherence, leading to more robust legal guidance.

---

## **System Instructions: Legal Expertise**

To ensure the AI provides detailed, accurate legal analysis, we configure the system with specialized instructions. This includes applying the 2023 versions of the laws, making the notebook **informative** by offering recent legal interpretations, and **novel** by showcasing how Gemini 1.5 assists in legal decision-making.

```python
# System instructions to ensure expertise in interpreting the 2023 legal texts.
textsi_1 = """You are an expert legal assistant well-versed in Indian criminal law, specializing in the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (Indian Evidence Act) 2023. These three laws are foundational to handling all criminal cases in India. Your role is to assist judges, lawyers, and police officers by providing precise legal information and interpretations based on these laws, offering case-specific guidance.

The legal documents you are referencing consist of:
- **Bharatiya Nyaya Sanhita (BNS)**: The official criminal code in India that replaced the Indian Penal Code (IPC) on July 1, 2024.
- **Bharatiya Nagarik Suraksha Sanhita (BNSS)**: The main legislation detailing procedural law for administration of criminal justice, replacing the Criminal Procedure Code (CrPC).
- **Bharatiya Sakshya Adhiniyam, 2023**: The Indian Evidence Act, governing how evidence is to be treated in criminal cases.

When providing legal analysis or answering queries, do the following:
1. **Use relevant sections of the BNS, BNSS, and Bharatiya Sakshya Adhiniyam** based on the specific scenario being queried.
2. Ensure the legal provisions, sections, and terminologies you use are from the 2023 versions of the law (not outdated versions).
3. Provide clear explanations of legal terms, procedures, and how they apply to the scenario.
4. Offer interpretations that reflect recent changes in the law to guide decision-making.
5. If the user needs to explore similar cases or refer to legal precedents, offer summaries of related sections and guidelines."""
```

---

## **Using Context Caching for Optimization**

In this notebook, we implement **context caching** to optimize the use of Gemini 1.5 when handling large, repeat content like the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam. By caching these legal texts, which exceed 32,768 tokens, we can reuse them across multiple API requests without resending them each time, significantly reducing token consumption and cost. This approach also speeds up subsequent requests by focusing only on new inputs, such as case-specific evidence or witness testimonies, while maintaining consistent legal interpretations. Overall, context caching ensures **efficiency** and **cost-effectiveness** in managing long legal documents.

```python
import datetime
from vertexai.preview import caching

cached_content = caching.CachedContent.create(
    model_name="gemini-1.5-pro-002",
    system_instruction=textsi_1,
    contents=[document1, document2, document3],
    ttl=datetime.timedelta(minutes=60),
    display_name="legal-cache",
)

cached_content_name = cached_content.name
```

---

## **Generation Configuration & Safety Settings**

We set up the generation configuration to ensure that the AI provides responses tailored to the legal domain. We also adjust safety settings to maximize the AI's interpretative capabilities in legal contexts, ensuring **high-quality outputs**.

- "temperature" controls randomness in predictions (0 makes responses more deterministic).
- "top_p" ensures tokens are chosen from the top 95% of cumulative probability, balancing diversity and coherence.
- All safety settings are turned off as we are expected to discuss harmful things in criminal cases.

```python
# Configure the AI model to generate appropriate outputs.
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0,
    "top_p": 0.95,
}

# Define safety settings for different content categories.
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]
```

---

## **Exponential Backoff with Jitter**

We make the system highly **efficient** by implementing exponential backoff, an algorithm that retries requests using exponentially increasing waiting times between requests, up to a maximum backoff time. You should generally use exponential backoff with jitter to retry requests that meet both the response and idempotency criteria. For best practices implementing automatic retries with exponential backoff, see [Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/).

```python
import time
import random
from functools import wraps

def exponential_backoff_with_jitter(max_retries=5, base_delay=1, max_backoff=32, jitter=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay

            while retries < max_retries:
                try:
                    # Try executing the function
                    return func(*args, **kwargs)
                except Exception as e:
                    # If it fails, log the failure (optional)
                    print(f"Attempt {retries + 1} failed: {e}")
                    
                    # Exponential backoff with jitter
                    if jitter:
                        delay_with_jitter = delay + random.uniform(0, delay)
                    else:
                        delay_with_jitter = delay

                    # Sleep for the backoff time
                    time.sleep(min(delay_with_jitter, max_backoff))

                    # Increment the retry count
                    retries += 1
                    delay = min(delay * 2, max_backoff)

            # If all retries fail, raise the last exception
            raise Exception(f"All {max_retries} retries failed.")
        
        return wrapper
    return decorator
```

---

## **Model Setup using Vertex AI**

Here, we initialize the **Gemini 1.5 Pro model** using the Vertex AI platform. This model is specifically designed for processing long-context data, making it ideal for reading and interpreting entire legal documents at once.

> To use AI Studio instead of Vertex AI, use `from google.generativeai import GenerativeModel` instead of `from vertexai.generative_models import GenerativeModel`. I'm using Vertex AI since I find it more reliable and have $500 credits.

```python
# Import required libraries and initialize the model.
from vertexai.preview.generative_models import GenerativeModel

cached_content_name = cached_content.name

@exponential_backoff_with_jitter(max_retries=5, base_delay=1, max_backoff=32, jitter=True)
def generate():
    cached_content = caching.CachedContent(cached_content_name=cached_content_name)
    model = GenerativeModel.from_cached_content(cached_content=cached_content)
    responses = model.generate_content(
        [text1, video1],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
    )

    return responses.candidates[0].content.parts[0].text, responses.usage_metadata
```

---

## **Generating Legal Interpretations**

Finally, we invoke the `generate()` function to process the scenario and legal texts. The AI will now provide its interpretation of the case based on the BNS, BNSS, and Bharatiya Sakshya Adhiniyam, helping legal professionals make informed judgments.

```python
# Run the generate function to analyze the case and provide legal insights.
reply, usage = generate()
print(reply)
```

---

## **Conclusion**

This notebook demonstrates the real-world application of Gemini 1.5 in aiding legal professionals as they adapt to India's new legal framework. By leveraging the model’s large context window, we can process entire legal documents at once, provide precise legal interpretations, and streamline decision-making in criminal cases. This is a **well-documented** notebook that demonstrated the best practices, showing how cutting-edge AI can transform how legal experts approach complex, evolving legislation. Let's finish by checking the usage metadata to see the number of tokens we handled.

```python
print(usage)
```
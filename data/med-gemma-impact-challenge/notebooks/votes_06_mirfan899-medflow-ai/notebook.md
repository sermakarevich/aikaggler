# MedFlow AI

- **Author:** Muhammad Irfan
- **Votes:** 37
- **Ref:** mirfan899/medflow-ai
- **URL:** https://www.kaggle.com/code/mirfan899/medflow-ai
- **Last run:** 2026-02-12 07:29:03.120000

---

# 🧠 MedFlow AI — A multi-agent clinical intelligence system

MedFlow AI is a human-in-the-loop, agent-orchestrated clinical assistant built on Google MedGemma.
The system separates documentation, clinical reasoning, and decision-making to ensure safety, transparency, and medical compliance.

🧩 System Overview

MedFlow AI consists of two MedGemma-powered agents coordinated by an orchestrator and supervised by a licensed doctor.

* Agent 1: Clinical Documentation (SOAP: S, O, A)

* Agent 2: Plan Analysis, Labs & Lifestyle Guidance

* Doctor: Final authority and decision-maker

## Model Used

google/medgemma-1.5-4b-it

Chosen for:

* Strong medical language understanding

* Safe clinical summarization

* Structured output generation

```python
import os
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from transformers import pipeline
import torch
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")

login(token=hf_token)


MODEL_ID = "google/medgemma-1.5-4b-it"

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)
```

## 🤖 Agent 1 — SOAP Note Generator
📌 Purpose

Agent 1 converts raw patient input into a structured clinical SOAP note, focusing strictly on documentation — not decision-making.

This agent does NOT:

* Diagnose diseases

* Prescribe medications

* Suggest treatment plans

```python
from transformers import pipeline
import json
import re

# Initialize your model pipeline
# pipe = pipeline("text-generation", model="google/medgemma-7b", device=0)

def strip_json_fences(text: str) -> str:
    """Remove markdown-style ```json fences from model output."""
    return re.sub(r"```json|```", "", text).strip()

def normalize_soap(soap_note: dict) -> dict:
    """Ensure SOAP keys are present and unwrap assessment if nested."""
    return {
        "subjective": soap_note.get("subjective") or soap_note.get("S") or {},
        "objective": soap_note.get("objective") or soap_note.get("O") or {},
        "assessment": soap_note.get("assessment") or soap_note.get("A") or {}
    }

def run_agent_1(patient_info: dict, images: list = None, previous_prescriptions: dict = None) -> dict:
    """
    Agent-1:
    - Generates S/O/A from patient input
    - Includes all explicitly provided symptoms, vitals, medical history, age, gender
    - Handles images and lab/test summaries
    - Handles returning patients with prior prescriptions
    - Returns missing info and ambiguity flags
    """

    # Include previous prescriptions if provided
    if previous_prescriptions:
        patient_info["previous_prescriptions"] = previous_prescriptions

    patient_json = json.dumps(patient_info, indent=2)

    # ===== Build messages for MedGemma model =====
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": """
You are Agent 1 in the MedFlow AI system.

Your task is to generate a structured SOAP note (S/O/A) in JSON from patient input.

Rules:
- Subjective (S) must include ALL explicitly provided patient symptoms, duration, severity, age, gender, and medical history.
- Objective (O) must include ALL vitals, lab results, imaging summaries, and previous prescriptions (if any).
- Images or test reports must be summarized under Objective → imaging_findings or lab_results.
- Assessment must be a plain string summarizing all S + O in neutral language.
- Do NOT infer, diagnose, or recommend.
- SOAP sections must contain only facts explicitly provided in the input.
- Include missing information in `missing_information`.
- Include any ambiguous or low-confidence info in `flags`.
- If a section has no data, return an empty object `{}`.
- Return valid JSON ONLY. Do NOT nest assessment inside another object.
"""
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Generate the SOAP note in JSON from the following patient data:\n{patient_json}"
            }]
        }
    ]

    # Attach images if provided
    if images:
        for img in images:
            messages[1]["content"].append({"type": "image", "image": img})

    # ===== Call the model =====
    output = pipe(text=messages, max_new_tokens=800)
    assistant_text = output[-1]["generated_text"]
    # print(assistant_text[-1]["content"])  # Optional debug print

    raw_json = strip_json_fences(assistant_text[-1]["content"])
    # ===== Parse JSON safely =====
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return {
            "soap_note": {"subjective": {}, "objective": {}, "assessment": {}},
            "missing_information": ["Model output could not be parsed"],
            "flags": ["JSON parsing failure"]
        }

    # Normalize keys and unwrap assessment
    soap = normalize_soap(data)
    # Return final structured output
    return {
        "soap_note": soap,
        "missing_information": data.get("missing_information", []),
        "flags": data.get("flags", []),
        "previous_prescriptions": patient_info.get("previous_prescriptions", []),
        "lab_results": data.get("lab_results", [])
    }
```

## Example Usage

* Patient age & gender

* Symptoms

* Duration & severity

* Medical history

* Vitals (if available)

```python
# Optional: add image
from PIL import Image
import requests
image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

patient_input = {
    "age": 45,
    "gender": "Male",
    "symptoms": [
        "Chest discomfort",
        "Shortness of breath during exertion",
        "Fatigue"
    ],
    "duration": "2 weeks",
    "severity": "Moderate",
    "medical_history": ["Hypertension"],
    "medications": [],
    "vitals": {
        "blood_pressure": "145/90",
        "heart_rate": "92 bpm"
    }
}
```

### Example Usage

```python
soap_note = run_agent_1(patient_input, images=[image])
print("=== SOAP S/O/A ===")
print(json.dumps(soap_note, indent=2))
```

## 🤖 Agent 2 — Plan, Labs & Lifestyle Analyzer
📌 Purpose

Agent 2 evaluates the doctor-provided plan in the context of:

* Symptoms

* Assessment


Use MedGemma to provide supportive, explainable insights, not decisions.

```python
def run_agent_2(soap_note: dict, doctor_plan: dict, ethnicity: str = None) -> dict:
    """
    Agent-2:
    - Receives Agent-1 SOAP note
    - Receives doctor plan (medications, tests, follow-up)
    - Produces full SOAP note including Plan
    - Evaluates medication and test alignment
    - Suggests lifestyle, food, exercise, clothing, music, fragrance recommendations
    """

    soap_json = json.dumps(soap_note, indent=2)
    plan_json = json.dumps(doctor_plan, indent=2)

    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": """
You are Agent 2 in the MedFlow AI system.

Your role:
- Receive the SOAP note (S/O/A) from Agent 1.
- Receive the doctor's Plan (medications, lab tests, follow-up).
- Produce a complete SOAP note including:
    - Subjective
    - Objective
    - Assessment
    - Plan
- Analyze medication alignment with symptoms and assessment.
- Analyze lab/test relevance.
- Suggest lifestyle recommendations: food, exercise, clothing, music, fragrances.
- Highlight missing information or caution if relevant.
- Maintain clinical neutrality.
- Do NOT diagnose or prescribe new medications or tests.

Output format (JSON only):
1. soap_note: dictionary with keys:
    - subjective (dict, include S details from Agent-1)
    - objective (dict, include O details from Agent-1)
    - assessment (string, free text)
    - plan (dict, include medications, tests, follow-up)
- medication_review: alignment score (0-100) and rationale
- test_validation: relevance score (0-100) and rationale for each test
- lifestyle_recommendations: food, exercise, clothing, music, fragrance
- additional_notes: optional notes on patient context or missing info
- Percentages as numbers (0-100)
- Clearly indicate missing or uncertain information
"""
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    f"Analyze the following clinical data:\n\n"
                    f"SOAP Note from Agent-1:\n{soap_json}\n\n"
                    f"Doctor Plan:\n{plan_json}\n\n"
                    f"Patient Ethnicity: {ethnicity or 'Not provided'}\n\n"
                    "Return ONLY a JSON object in the format specified above."
                )
            }]
        }
    ]

    # Call the model
    output = pipe(text=messages, max_new_tokens=2000)
    assistant_text = output[-1]["generated_text"]
    # print(assistant_text[-1]["content"])

    # Parse JSON safely
    raw_json = strip_json_fences(assistant_text[-1]["content"])
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        data = {
            "soap_note": {},
            "medication_review": [],
            "test_validation": [],
            "lifestyle_recommendations": {},
            "additional_notes": ["Model output could not be parsed."]
        }

    return data
```

## 👨‍⚕️ Doctor-in-the-Loop (Critical Step)

Before any recommendations are made:

*  Doctor reviews SOAP (S/O/A)

*  Doctor adds Plan (P) manually

*  Doctor may prescribe:

* Medications

* Lab tests

* Follow-up instructions

👉 No AI output is patient-facing without doctor approval

## Example Usage

Doctor adds medications, lab tests and ethnicity for Soap note and AI suggestions.

```python
# ===== Doctor adds the plan =====
patient_ethnicity = "South Asian"
doctor_plan = {
    "medications": ["Omeprazole 20mg once daily"],
    "lab_tests": ["H. pylori test", "CBC"],
    "follow_up": "2 weeks"
}
```

```python
# Agent 2 generates the soap note and other recommendations.
agent2_output = run_agent_2(soap_note, doctor_plan, patient_ethnicity)
print(json.dumps(agent2_output, indent=2))
```

## Visualize JSON and PDF Generation
Paste the agent2_output in JSONCRACK to visualize it.

```python
!pip install reportlab -q
```

```python
import json
import os
from IPython.display import IFrame, display


def save_visualization_html(json_data: dict, output_file: str = "visualization.html"):
    """
    Saves an HTML file that embeds the JSON Crack widget and sends data via postMessage.
    Includes retries and a manual button to ensure data loading.
    """
    json_str = json.dumps(json_data)
    widget_url = "https://jsoncrack.com/widget"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MedFlow AI - JSON Visualization</title>
        <style>
            body, html {{ margin: 0; padding: 0; height: 100%; overflow: hidden; font-family: sans-serif; }}
            iframe {{ width: 100%; height: 100%; border: none; }}
            #controls {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: white;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                z-index: 100;
            }}
            button {{
                padding: 8px 16px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            button:hover {{ background-color: #0056b3; }}
        </style>
    </head>
    <body>
        <div id="controls">
            <button onclick="sendData()">Reload Graph Data</button>
        </div>
        <iframe id="jsoncrackEmbed" src="{widget_url}"></iframe>
        <script>
            const jsonData = {json_str};
            const iframe = document.getElementById("jsoncrackEmbed");
            
            const sendData = () => {{
                console.log("Sending data to JSON Crack...");
                iframe.contentWindow.postMessage({{
                    json: JSON.stringify(jsonData),
                    options: {{
                        theme: "light",
                        direction: "RIGHT"
                    }}
                }}, "*");
            }};

            // Retry mechanism
            let attempts = 0;
            const maxAttempts = 10;
            
            const attemptSend = () => {{
                if (attempts < maxAttempts) {{
                    sendData();
                    attempts++;
                    setTimeout(attemptSend, 1000); // Retry every 1s
                }}
            }};

            iframe.onload = () => {{
                console.log("Iframe loaded, starting send attempts...");
                attemptSend();
            }};
            
        </script>
    </body>
    </html>
    """
    
    with open(output_file, "w") as f:
        f.write(html_content)
    
    print(f"Visualization saved to {os.path.abspath(output_file)}")
    return os.path.abspath(output_file)

output_html_path = "agent2_output_graph.html"
save_visualization_html(agent2_output, output_html_path)

# Display in Kaggle notebook
display(IFrame(src=output_html_path, width="100%", height=700))
```

### Soap note pdf
SOAP note is generated. Can be modified according to the need.

```python
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from IPython.display import IFrame, display

def generate_professional_soap_pdf(soap_note, patient_input, output_path="soap_note_professional.pdf"):
    """
    Generate a professional PDF for a SOAP note with tables for vitals, labs, meds, imaging, and plan.
    Includes MedFlow AI header and patient info extracted from patient_input.
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=30)
    styles = getSampleStyleSheet()
    elements = []

    # Custom styles
    header_style = ParagraphStyle(name='HeaderStyle', parent=styles['Heading1'], fontSize=18, spaceAfter=10, alignment=1)
    section_style = ParagraphStyle(name='SectionStyle', parent=styles['Heading2'], fontSize=14, spaceAfter=6)
    normal_style = styles['Normal']

    # --- Header ---
    elements.append(Paragraph("MedFlow AI - SOAP Note", header_style))
    elements.append(Spacer(1, 12))

    # --- Patient Info ---
    patient_lines = []
    for key in ["name", "age", "gender"]:
        if key in patient_input:
            val = patient_input[key]
            patient_lines.append(f"<b>{key.capitalize()}:</b> {val}")
    
    if patient_lines:
        elements.append(Paragraph("Patient Information", section_style))
        elements.append(Paragraph("<br/>".join(patient_lines), normal_style))
        elements.append(Spacer(1, 12))

    # --- Helper Functions ---
    def format_value(val):
        if isinstance(val, dict):
            return ', '.join(f"{k}: {format_value(v)}" for k,v in val.items())
        elif isinstance(val, list):
            return ', '.join(format_value(v) for v in val)
        elif val is None:
            return "None"
        else:
            return str(val)

    def add_dict_table(title, data_dict):
        if not data_dict:
            return
        elements.append(Paragraph(title, section_style))
        table_data = [[Paragraph("<b>Field</b>", normal_style), Paragraph("<b>Value</b>", normal_style)]]
        for k, v in data_dict.items():
            table_data.append([Paragraph(str(k), normal_style), Paragraph(format_value(v), normal_style)])
        table = Table(table_data, hAlign='LEFT', colWidths=[150, 350], repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    # --- Add SOAP Sections ---
    add_dict_table("Subjective", soap_note.get("subjective", {}))
    add_dict_table("Objective", soap_note.get("objective", {}))
    add_dict_table("Assessment", {"Assessment": soap_note.get("assessment", "")})
    add_dict_table("Plan", soap_note.get("plan", {}))  # Plan as table

    # Build PDF
    doc.build(elements)
    print(f"Professional SOAP note PDF generated: {output_path}")
    return output_path

# --- Usage ---
pdf_path = generate_professional_soap_pdf(agent2_output["soap_note"], patient_input)

# Display inline in Kaggle
display(IFrame(src=pdf_path, width="100%", height=700))
```
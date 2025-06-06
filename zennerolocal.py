from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr
import requests
import json

# Initialize FastAPI app
app = FastAPI()

# Request schema for input (Call transcript)
class TranscriptInput(BaseModel):
    transcript: str

# Response schema for output (Fields with confidence scores)
class FieldOutput(BaseModel):
    field_name: str
    field_value: str
    confidence_score: float

class ExtractResponse(BaseModel):
    fields: list[FieldOutput]

# Define the endpoint URL and headers for interacting with the external model
url = "http://localhost:11434/api/generate"
headers = {'Content-Type': 'application/json'}

# History to manage prompts
history = []

def generate_response(prompt: str):
    """
    Function to generate response by interacting with the external model.
    Sends the prompt to the model and returns the result.
    """
    history.append(prompt)
    final_prompt = f"""
You are an AI that extracts data from a mortgage application conversation to fill out the Uniform Residential Loan Application (Form 1003).

Extract only the following fields (use exact names for "field_name"):

- Borrower Name
- Employer Name
- Loan Amount
- Property Address
- Current Address
- Social Security Number
- Annual Income
- Date of Birth
- Marital Status
- Number of Dependents
- Phone Number
- Email Address
- Loan Purpose
- Estimated Value of Property
- Assets
- Liabilities
- Monthly Housing Expenses
- Other Income
- Co-Borrower Name
- Co-Borrower Employer Name
- Co-Borrower Annual Income
- Co-Borrower Social Security Number

Return ONLY a valid JSON in this format:

{{
  "fields": [
    {{
      "field_name": "example",
      "field_value": "example value",
      "confidence_score": 0.95
    }}
  ]
}}

Rules:
- Do not include explanations, comments, markdown, or code blocks. No text outside JSON. JSON must start with '''{''' and end with '''}'''.
- Include only fields present in the transcript.
- Use confidence_score = 1.0 only if the answer is clear and explicitly stated.
- If you infer, guess, or the user is uncertain, lower confidence (0 < score < 1).
- If a user changes an answer, use their final answer.
- Output strictly valid JSON. No explanations or extra text.
- JSON must escape special characters (e.g., newline → \\n, quote → \\")

Transcript:
{prompt}
    """.join(history)

    data = {
        "model": "zenngpt1.0",
        "prompt": final_prompt,
        "stream": False
    }

    try:
        # Make POST request to the external model API
        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            # Parse the JSON response and return the "response" field
            data = json.loads(response.text)
            return data.get('response', '')
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Error from model: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

@app.post("/extract-fields", response_model=ExtractResponse)
def extract_fields(input_data: TranscriptInput):
    """
    Endpoint to accept a transcript and extract fields required for 1003 form.
    """
    transcript_text = input_data.transcript.strip()

    if not transcript_text:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    try:
        # Prepare the prompt to extract fields for the 1003 form
        prompt = f"""
        {transcript_text}
        """

        # Call the generate_response function to get the extracted fields
        response_text = generate_response(prompt)

        # Parse the response (assuming model returns a valid JSON structure)
        output_json = json.loads(response_text)

        # Extract and structure the fields
        fields = []
        for field in output_json.get('fields', []):
            fields.append(FieldOutput(**field))

        return {"fields": fields}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON. Please check the prompt or model output.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")

examples = [
    ["Hi, I'm John Doe, I'd like a $250,000 loan for 123 Main St."],
    ["My name is Jane Smith, applying for $300,000 on 456 Oak Ave."],
]
interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter transcript here..."),
    outputs=gr.JSON(),
    title="Mortgage Application AI",
    theme="soft",
    description="Extract structured fields for the 1003 form.",
    css=".gr-button { background-color: #4CAF50; color: white; }",
    examples=examples
)
interface.launch()


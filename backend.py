from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests  # type: ignore # Use 'requests' to interact with Gemini AI's API
import json
import os

# Initialize FastAPI app
app = FastAPI()

# Set your Gemini AI API key here (replace with your actual Gemini API key)
GEMINI_API_KEY = ""  # Replace this with your actual API key from Gemini AI

# Request schema
class TranscriptInput(BaseModel):
    transcript: str

# Response schema
class FieldOutput(BaseModel):
    field_name: str
    field_value: str
    confidence_score: float

class ExtractResponse(BaseModel):
    fields: List[FieldOutput]

@app.post("/extract-fields", response_model=ExtractResponse)
def extract_fields(input_data: TranscriptInput):
    transcript_text = input_data.transcript.strip()

    if not transcript_text:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    try:
        # Create prompt to extract fields
        prompt = f"""
        Extract fields from the following mortgage application call transcript for 1003 Form.
        Return a JSON array with 'field_name', 'field_value', and 'confidence_score' between 0 and 1.

        Transcript:
        {transcript_text}
        """

        # Call Gemini AI API (use requests to make the POST request)
        url = "https://api.gemini.com/v1/chat/completions"  # Replace with the correct URL for Gemini AI's API
        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gemini-model",  # Replace with the correct model name if different
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

        # Make the POST request to Gemini AI
        response = requests.post(url, headers=headers, json=payload)

        # Check for errors in the response
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Error from Gemini AI: {response.text}")

        response_data = response.json()

        # Get the output text from the response
        output_text = response_data['choices'][0]['message']['content'].strip()

        # Parse model's output (assuming the model returns valid JSON array)
        output_json = json.loads(output_text)

        # Create a list of FieldOutput based on the parsed JSON
        fields = []
        for field in output_json:
            fields.append(FieldOutput(**field))

        # Return the fields in the response
        return ExtractResponse(fields=fields)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON. Please check the prompt or model output.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")
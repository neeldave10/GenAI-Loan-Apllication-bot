🏦 GenAI Loan Application Bot
An end-to-end GenAI-powered application that automates the extraction of mortgage loan application data (Form 1003) from customer-agent call transcripts using locally hosted LLMs.

🚀 Project Overview
This project addresses the challenge of extracting structured data from unstructured and ambiguous customer service call transcripts. It uses a locally hosted LLM (Mistral via Ollama) to identify and extract key fields required for mortgage applications, such as:

Borrower Name

Loan Amount

Social Security Number

Property Address

Employer Information

Income

And more (~23 fields total)

The system supports both UI interaction and REST API integration, returning results as structured JSON with confidence scores.

🔧 Tech Stack
Frontend: Gradio UI

Backend: FastAPI

LLM: Mistral model hosted locally via Ollama

Prompt Engineering: Custom prompts to guide extraction and ensure JSON formatting

Deployment: Localhost with RESTful API support

🧠 Features
✅ Extracts mortgage fields from free-form text

✅ Provides confidence scores (0–1) for each value

✅ Handles ambiguity, corrections, and edge cases

✅ Validates and parses LLM output as structured JSON

✅ Works offline without external LLM APIs (OpenAI, Gemini)

📬 API Endpoint
POST /extract-fields

Request:
json
Copy
Edit
{
  "transcript": "Hi, my name is John Doe. I’d like to apply for a mortgage loan..."
}
Response:
json
Copy
Edit
{
  "fields": [
    {
      "field_name": "Borrower Name",
      "field_value": "John Doe",
      "confidence_score": 0.95
    },
    {
      "field_name": "Loan Amount",
      "field_value": "$250,000",
      "confidence_score": 0.90
    }
  ]
}
🖥️ UI Features
Text input for transcript

Submit button

Structured result display with CSS-styled output

Error handling and user-friendly messaging

📊 Example Scenarios Covered
Complete applications

Missing or partial data

Contradictory or corrected info

Fast or jumbled speech

Refusals due to privacy concerns

🛠️ Challenges & Learnings
Explored OpenAI and Gemini but opted for Ollama to avoid cost

Managed long inference times with prompt optimization

Successfully integrated FastAPI, Gradio, and Mistral for a seamless offline experience

🔮 Future Enhancements
Optimize model size or quantization for speed

Add audio-to-text (speech recognition) pipeline

Enable GPU acceleration for faster inference

Improve caching and error handling

Consider fine-tuning on domain-specific data

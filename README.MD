# Mini AI-Based Vulnerability Analysis System

This project is a mini AI-powered security analysis system that detects software vulnerabilities using vector embeddings, Pinecone, and a local Meta LLaMA GGUF model.

It analyzes vulnerable code snippets or scan outputs and generates a structured security vulnerability report.

## Architecture Overview

User Input (Code / Scan Output)
→ Embedding Model (SentenceTransformers)
→ Pinecone Vector Database (Similarity Search)
→ Relevant Vulnerability Context
→ Local LLaMA GGUF Model (Reasoning)
→ Structured Vulnerability Report (JSON)

This follows a Retrieval-Augmented Generation (RAG) approach.


## Technologies Used

- **Python**
- **SentenceTransformers** (Embeddings)
- **Pinecone** (Vector Database)
- **Meta LLaMA 2 GGUF** (Local LLM via llama-cpp)
- **FastAPI** (Optional bonus support)


## Supported Vulnerability Types

- SQL Injection
- Cross-Site Scripting (XSS)
- Command Injection
- Insecure Dependencies
- Open Ports / Misconfiguration


## Project Structure

vulnscan-mini/
├── app.py
├── requirements.txt
├── README.md
├── sample_input.txt
├── data/
│ └── vulnerabilities.txt
└── models/
└── llama-2-7b-chat.gguf


## LLaMA Model Setup

This project uses a local Meta LLaMA GGUF model via `llama-cpp-python`.

Due to size and licensing restrictions, the model file is **not included** in this repository.

Download the model from:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

Place the downloaded `.gguf` file inside the `models/` directory.


## Setup Instructions

### 1. Create Virtual Environment
python -m venv venv
venv\Scripts\activate

# 2. Install Dependencies
pip install -r requirements.txt

### 3. Set Pinecone API Key
setx PINECONE_API_KEY "your_pinecone_api_key"

## Environment Variables

This project uses environment variables to securely manage secrets.

### Pinecone API Key

Set the Pinecone API key before running the application.

**Windows**
setx PINECONE_API_KEY "your_pinecone_api_key"


### 4. How to Run
python app.py


### 5.Example Input
user = input()
query = "SELECT * FROM users WHERE name = '" + user + "'"


### 6. Example Output
{
  "vulnerability": "SQL Injection",
  "severity": "High",
  "description": "User input is concatenated into SQL queries without sanitization.",
  "impact": "Attackers can read or modify database records.",
  "remediation": [
    "Use parameterized queries",
    "Validate user input",
    "Use prepared statements or ORM"
  ]
}

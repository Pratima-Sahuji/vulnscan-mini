from llama_cpp import Llama
import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")

INDEX_NAME = "vuln-index"

MODEL_PATH = "./models/llama-2-7b-chat.gguf"

pc = Pinecone(api_key=PINECONE_API_KEY)
print("Pinecone client initialized:", pc is not None)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


index = pc.Index(INDEX_NAME)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded")

with open("data/vulnerabilities.txt", "r") as f:
    documents = f.readlines()

vectors = []

for i, doc in enumerate(documents):
    embedding = embedder.encode(doc).tolist()
    vectors.append({
        "id": f"doc-{i}",
        "values": embedding,
        "metadata": {"text": doc}
    })

index.upsert(vectors=vectors)
print("Vulnerability knowledge indexed")


def retrieve_security_context(user_input, top_k=2):
    """
    Retrieves relevant vulnerability context from Pinecone
    using vector similarity search.
    """
    query_embedding = embedder.encode(user_input).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    context = "\n".join(
        match["metadata"]["text"] for match in results["matches"]
    )

    return context


llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,         
    n_threads=6,         
    n_batch=128,        
    temperature=0.0,    
    verbose=False        
)


print("LLaMA model loaded")


def analyze_with_llama(user_input, security_context):
    system_prompt = (
        "You are a strict JSON generation engine for cybersecurity analysis. "
        "You must output ONLY valid JSON. "
        "No explanations. No questions. No extra text."
    )

    user_prompt = f"""
Context:
{security_context}

Input:
{user_input}

Return ONLY this JSON object:

{{
  "vulnerability": "SQL Injection | XSS | Command Injection | Insecure Dependency | Misconfiguration",
  "severity": "Low | Medium | High | Critical",
  "description": "One sentence description",
  "impact": "One sentence impact",
  "remediation": ["step 1", "step 2", "step 3"]
}}
"""

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=300
    )

    return response["choices"][0]["message"]["content"]




def analyze_input(user_input):
    context = retrieve_security_context(user_input)
    return analyze_with_llama(user_input, context)


if __name__ == "__main__":
    test_input = "SELECT * FROM users WHERE name = '" + "USER_INPUT" + "'"

    result = analyze_input(test_input)

    print("\n--- Vulnerability Report ---\n")
    print(result)



#optional 

api = FastAPI()

class AnalyzeRequest(BaseModel):
    input: str

@api.post("/analyze")
def analyze(req: AnalyzeRequest):
    result = analyze_input(req.input)
    return json.loads(result)
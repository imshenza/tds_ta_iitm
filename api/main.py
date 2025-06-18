import os
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import requests
from dotenv import load_dotenv
from typing import Optional, List
import torch

# Load environment variables
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or restrict if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)


DB_PATH = "data/chunks.db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"



# Lazy load model
MODEL = None

# Request schema
class Query(BaseModel):
    question: str
    image: Optional[str] = None

# Response schema
class QueryResponse(BaseModel):
    answer: str
    relevant_links: List[str]

def load_model():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def search_chunks(query, k=5):
    load_model()

    # Load all chunks from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, content, source FROM chunks")
    rows = cursor.fetchall()
    conn.close()

    contents = [row[1] for row in rows]
    embeddings = MODEL.encode(contents, convert_to_tensor=True)
    query_embedding = MODEL.encode(query, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, embeddings, top_k=k)[0]

    top_chunks = [{
        "id": rows[hit['corpus_id']][0],
        "content": rows[hit['corpus_id']][1],
        "source": rows[hit['corpus_id']][2]
    } for hit in hits]

    return top_chunks

def generate_answer_openrouter(query, context_chunks):
    context_text = "\n\n".join([chunk['content'] for chunk in context_chunks])
    prompt = f"""Answer the question based on the following context:\n\n{context_text}\n\nQuestion: {query}"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You're a helpful teaching assistant for the IIT Madras Data Science program."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Failed to get response from GPT: {str(e)}"

@app.post("/query", response_model=QueryResponse)
def query_route(query: Query):
    top_chunks = search_chunks(query.question, k=5)
    answer = generate_answer_openrouter(query.question, top_chunks)
    links = list(set(chunk["source"] for chunk in top_chunks))
    return {
        "answer": answer,
        "relevant_links": links
    }

# Local dev run
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sentence_transformers import util
import os
import uvicorn

# Load saved SentenceTransformer model
model = joblib.load("similarity_model.joblib")

# Initialize FastAPI app
app = FastAPI()

# Define request body structure
class TextPair(BaseModel):
    text1: str
    text2: str

# Define POST endpoint
@app.post("/")
async def get_similarity(data: TextPair):
    try:
        # Encode input texts
        embeddings = model.encode([data.text1, data.text2])
        # Calculate cosine similarity
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return {"similarity score": round(score, 4)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # fallback to 8000 for local dev
    uvicorn.run("app:app", host="0.0.0.0", port=port)
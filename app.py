from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
from sentence_transformers import util
import os
import uvicorn

# Load the pre-trained model
model = joblib.load("similarity_model.joblib")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (optional but good for testing with frontend or Postman)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body schema
class TextPair(BaseModel):
    text1: str
    text2: str

# Root route for testing server
@app.get("/")
async def root():
    return JSONResponse(content={"message": "Semantic Similarity API is running. Use POST /api"})

# Main similarity route
@app.post("/api")
async def get_similarity(data: TextPair):
    try:
        # Encode input texts using the loaded model
        embeddings = model.encode([data.text1, data.text2])
        # Calculate cosine similarity
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return {"similarity score": round(score, 4)}
    except Exception as e:
        return {"error": str(e)}

# Uvicorn entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Required for Railway
    uvicorn.run("app:app", host="0.0.0.0", port=port)
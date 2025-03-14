from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import csv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Initialize FastAPI
app = FastAPI(title="FAISS Caption Similarity API")
logger = logging.getLogger("fastapi")
# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample captions dataset
# Sample captions dataset
captions = []

# Read the CSV file and populate the dictionary
csv_file = "captions.txt"  # Replace with your actual file path

with open(csv_file, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row

    for row in reader:
        image, caption = row
        captions.append(caption)

captions = list(set(captions))
# Convert captions into embeddings
caption_embeddings = model.encode(captions, normalize_embeddings=True)
caption_embeddings = np.array(caption_embeddings).astype("float32")

# Create FAISS index (using Inner Product for cosine similarity)
index = faiss.IndexFlatIP(caption_embeddings.shape[1])
index.add(caption_embeddings)

# Request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# Search API endpoint
@app.post("/search")
def search_similar_captions(request: QueryRequest):
    query_embedding = model.encode([request.query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_embedding, request.top_k)
    
    results = [{"caption": captions[idx], "score": float(scores[0][i])} for i, idx in enumerate(indices[0])]
    logger.info(f"similar captions for context: {results}")
    return {"query": request.query, "results": results}

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the FAISS Caption Similarity API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8802)
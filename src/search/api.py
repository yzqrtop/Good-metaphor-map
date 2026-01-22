from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Initialize FastAPI application
app = FastAPI()

# Initialize encoder
encoder = SentenceTransformer("jinaai/jina-embeddings-v3")

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 5

# Define response model
class SearchResult(BaseModel):
    id: str
    text: str
    score: float

# Simulate Milvus search function
def milvus_search(vec, k=5):
    """Simulate Milvus search"""
    # In actual projects, should implement interaction with Milvus here
    # Currently return simulated results
    return [
        {"id": f"doc_{i}", "text": f"Result {i}", "score": 0.9 - i * 0.1}
        for i in range(k)
    ]

# Search endpoint
@app.post("/search", response_model=list[SearchResult])
async def search(req: QueryRequest):
    """Search for similar metaphors"""
    # Generate query vector
    vec = encoder.encode([req.query])
    
    # Search in Milvus
    results = milvus_search(vec, k=req.k)
    
    return results

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
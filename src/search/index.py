import json
from pathlib import Path
import argparse
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path):
    """Load JSONL file"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                continue
    return data


def insert_to_milvus(records, embs):
    """Insert records and embedding vectors into Milvus"""
    # In actual projects, should implement interaction with Milvus here
    # Currently only print information
    print(f"Preparing to insert {len(records)} records into Milvus")
    print(f"Embedding vector dimension: {embs.shape if hasattr(embs, 'shape') else 'unknown'}")
    # Actual insertion logic


def build_index(gold_path: Path):
    """Build vector index"""
    # Initialize encoder
    encoder = SentenceTransformer("jinaai/jina-embeddings-v3")
    
    # Load gold dataset
    records = load_jsonl(gold_path)
    
    print(f"Loaded {len(records)} records")
    
    # Generate embedding vectors
    texts = [f"{r.get('E', '')} {r.get('V', '')}" for r in records]
    embs = encoder.encode(texts)
    
    print(f"Generated {len(embs)} embedding vectors")
    
    # Insert into Milvus
    insert_to_milvus(records, embs)
    
    print("Index construction completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=Path, required=True, dest="gold_path", help="Gold dataset path")
    args = parser.parse_args()
    
    # 构建索引
    build_index(args.gold_path)
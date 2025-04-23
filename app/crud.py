from sqlalchemy.orm import Session
from app import models, schemas
import numpy as np
from typing import List
from sqlalchemy import func, cast
from pgvector.sqlalchemy import Vector
from app.vector_utils import DocumentProcessor
from typing import List, Dict

# Initialize the processor once
document_processor = DocumentProcessor()

# Define wrapper functions to maintain compatibility with existing code
def split_text_into_chunks(text: str, max_tokens: int = 500) -> List[str]:
    """Wrapper to maintain original function name but use the new processor"""
    chunks = document_processor.split_text_into_chunks(text, max_tokens=max_tokens)
    # Return just the content parts to match the original function's return type
    return [chunk.content for chunk in chunks]

def generate_embeddings_for_chunks(chunks: List[str]) -> List[np.ndarray]:
    """Wrapper to maintain original function name but use the new processor"""
    return [document_processor.get_embedding(chunk) for chunk in chunks]

def get_embedding(text: str) -> np.ndarray:
    """Wrapper to maintain original function name but use the new processor"""
    return document_processor.get_embedding(text)

def create_document(db: Session, doc: schemas.DocumentCreate):
    try:
        # Create document record
        db_doc = models.Document(title=doc.title)
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)

        # Process content (no cleaning needed if DB is properly configured)
        chunks = split_text_into_chunks(doc.content)
        embeddings = generate_embeddings_for_chunks(chunks)

        # Store chunks
        for chunk, embedding in zip(chunks, embeddings):
            db_chunk = models.DocumentChunk(
                content=chunk, 
                embedding=embedding, 
                document_id=db_doc.id
            )
            db.add(db_chunk)

        db.commit()
        return db_doc
        
    except Exception as e:
        db.rollback()
        print(f"Error storing document: {str(e)}")
        raise


def semantic_search(db: Session, query: str, top_k: int = 5) -> List[Dict[str, str]]:
    print(f"🔍 Searching for: {query}")
    
    query_embedding = get_embedding(query)
    
    if len(query_embedding) != 384:
        raise ValueError(f"Embedding length is {len(query_embedding)}, expected 384.")

    print(f"Query embedding: {query_embedding[:10]}...")  # Shortened for readability

    # Cast the embedding properly
    query_embedding_vector = cast(query_embedding, Vector(384))

    score = func.cosine_distance(
        models.DocumentChunk.embedding,
        query_embedding_vector
    ).label("score")

    results = (
        db.query(models.DocumentChunk, score)
        .order_by(score.desc())
        .limit(top_k)
        .all()
    )

    print("🔎 Results:", [
        {
            "content": chunk.content,
            "score": float(similarity_score)
        }
        for chunk, similarity_score in results
    ])

    search_results = []
    for chunk, similarity_score in results:
        search_results.append({
            "content": chunk.content,
            "score": float(similarity_score)
        })

    return search_results
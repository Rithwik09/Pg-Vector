from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# def get_embedding(text: str):
#     embedding = model.encode(text)
#     norm = np.linalg.norm(embedding)
#     return embedding / norm if norm != 0 else embedding
def get_embedding(text: str):
    embedding = model.encode(text, normalize_embeddings=False)  # disable internal normalization
    embedding = np.array(embedding)  # just in case it's a list
    norm = np.linalg.norm(embedding)

    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Embedding norm is zero or invalid. Cannot normalize.")
    
    return embedding / norm
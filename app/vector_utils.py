# from sentence_transformers import SentenceTransformer

# def get_model():
#     return SentenceTransformer('all-MiniLM-L6-v2')

# def get_embedding(text: str):
#     model = get_model()
#     return model.encode([text])[0]


from sentence_transformers import SentenceTransformer

# Load model globally to avoid reloading on every request
model = SentenceTransformer("all-MiniLM-L6-v2")  # You can replace with any other model

def get_embedding(text: str) -> list[float]:
    return model.encode(text).tolist()

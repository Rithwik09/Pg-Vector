from sqlalchemy.orm import Session
from app import models, schemas
from app.vector_utils import get_embedding

def create_note(db: Session, note: schemas.NoteCreate):
    embedding = get_embedding(note.content)
    db_note = models.Note(content=note.content, embedding=embedding)
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    return db_note

def semantic_search(db: Session, query: str):
    query_embedding = get_embedding(query)
    return (
        db.query(models.Note)
        .order_by(models.Note.embedding.cosine_distance(query_embedding))
        .limit(10)
        .all()
    )

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app import db, crud, schemas

app = FastAPI()

def get_db():
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

@app.post("/notes", response_model=schemas.NoteOut)
def add_note(note: schemas.NoteCreate, db: Session = Depends(get_db)):
    try:
        print(f"📝 Received note: {note}")
        return crud.create_note(db, note)
        
        return {"message": "Note added"}    
    except Exception as e:
        print("Error while adding note:", e)
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/search", response_model=list[schemas.NoteOut])
# def search_notes(q: str, db: Session = Depends(get_db)):
#     try : 
#         print(f"🔍 Searching for: {q}")
#         return crud.semantic_search(db, q)
#     except Exception as e:
#         print("Error while searching:", e)
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/search", response_model=list[schemas.NoteOut])
def search_notes(search: schemas.SearchQuery, db: Session = Depends(get_db)):
    try:
        print(f"🔍 Searching for: {search.query}")
        return crud.semantic_search(db, search.query)
    except Exception as e:
        print("Error while searching:", e)
        raise HTTPException(status_code=500, detail=str(e))

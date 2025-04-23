from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app import db, crud, schemas
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import SearchResult

app = FastAPI()
# Your frontend origin
origins = [
    "http://192.168.1.25:8080",
    "http://localhost:8080",  # Optional, good to keep for dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Don't use ["*"] if you set allow_credentials=True
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Dependency
def get_db():
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

@app.post("/documents", response_model=schemas.DocumentBase)
def upload_document(
    title: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    from app.pdf_parser import extract_text_from_pdf
    import tempfile

    print(f"📄 Uploading document: {title}")

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    # Extract text from PDF
    content = extract_text_from_pdf(tmp_path)

    # Create a schema object manually
    from app.schemas import DocumentCreate
    doc_schema = DocumentCreate(title=title, content=content)

    return crud.create_document(db, doc_schema)

# Semantic search
@app.post("/search", response_model=list[SearchResult])
def search_chunks(search: schemas.SearchQuery, db: Session = Depends(get_db)):
    try:
        print(f"🔍 Searching for: {search.query}")
        return crud.semantic_search(db, search.query)
    except Exception as e:
        print("❌ Error during search:", e)
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel
from typing import Optional


# Used in search
class SearchQuery(BaseModel):
    query: str


# For chunk return
class DocumentChunkBase(BaseModel):
    content: str

    class Config:
        orm_mode = True


# For upload
class DocumentCreate(BaseModel):
    title: str
    content: str  # full raw text from PDF


class DocumentBase(BaseModel):
    title: str

    class Config:
        orm_mode = True


# ✅ For search results
class SearchResult(BaseModel):
    document_title: Optional[str] = None
    chunk_id: Optional[str] = None
    content: str
    score: float

    class Config:
        orm_mode = True

from pydantic import BaseModel

class SearchQuery(BaseModel):
    query: str
    
class NoteCreate(BaseModel):
    content: str

class NoteOut(BaseModel):
    id: int
    content: str

    class Config:
        orm_mode = True

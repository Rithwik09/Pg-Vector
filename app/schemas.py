from pydantic import BaseModel

class NoteCreate(BaseModel):
    content: str

class NoteOut(BaseModel):
    id: int
    content: str

    class Config:
        orm_mode = True

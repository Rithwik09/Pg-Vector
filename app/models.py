from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Note(Base):   
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    embedding = Column(Vector(384), nullable=False)

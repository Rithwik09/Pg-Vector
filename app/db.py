# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, declarative_base

# DATABASE_URL = "postgresql://postgres:admin@localhost:5432/semantic_notes"

# engine = create_engine(DATABASE_URL)
# try:
#     with engine.connect() as connection:
#         print("Connection to the database stabilized.")
# except Exception as e:
#     print("Could not connect to the database:", e)
  
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, Note  # Ensure Note is imported!
import os

# Replace with your actual database credentials
DB_USER = "admin_user"
DB_PASSWORD = "dbuser123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "semantic_vector"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

print("🔗 Connecting to DB at:", engine.url)

try:
    Base.metadata.create_all(bind=engine)
    print("✅ All tables created (if not already).")
except Exception as e:
    print("❌ Failed to create tables:", e)

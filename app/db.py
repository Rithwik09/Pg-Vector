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


# DB_USER = "pgvector"
# DB_PASSWORD = "Ranucle!#139"
# DB_HOST = "pgvector-test-instance-1.cncu6qecijbq.us-east-2.rds.amazonaws.com"
# DB_PORT = "5432"
# DB_NAME = "pgvector-test"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base
import os

DB_USER = "admin_user"
DB_PASSWORD = "dbuser123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "test_db"

# Modified connection string with UTF-8 support
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?client_encoding=utf8"

# Configure engine with pool_pre_ping to handle connection issues
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=True  # Enable SQL logging for debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

print("🔗 Connecting to DB at:", engine.url)

try:
    Base.metadata.create_all(bind=engine)
    print("✅ All tables created (if not already).")
except Exception as e:
    print("❌ Failed to create tables:", e)
    raise
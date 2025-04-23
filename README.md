# Creating a new Environment

python -m venv env

    Activate the environment : 
    source venv/Scripts/activate

    deactivate the environment : 
    deactivate

# Install dependencies
pip install -r requirements.txt


uvicorn app.main:app --reload


# running the conda env first 
 open user profile OR where you have created a conda env with local_vector file 


 # before installing setup db


# file structure 

 BACK-END/
├── app/
│   ├── __init__.py
│   ├── crud.py          # DB insert/retrieve functions (PDF chunks, embeddings, etc.)
│   ├── db.py            # DB connection setup (should include pgvector)
│   ├── main.py          # FastAPI entrypoint
│   ├── models.py        # Pydantic/ORM models for DB tables
│   ├── schemas.py       # Request/Response validation schemas
│   └── vector_utils.py  # Embedding generation + similarity search logic
├── venv/                # Virtual environment
├── README.md
└── requirements.txt
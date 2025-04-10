# Creating a new Environment

python -m venv env

    Activate the environment : 
    source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt


uvicorn app.main:app --reload

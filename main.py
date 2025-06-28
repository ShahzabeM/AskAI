# --- Core Libraries & Imports ---
from fastapi import FastAPI, HTTPException  # FastAPI app & error handling
from pydantic import BaseModel              # Used for validating incoming request data
from openai import OpenAI                   # OpenAI SDK to call GPT
from dotenv import load_dotenv              # Loads environment variables from a .env file
import os                                   # For accessing environment variables
from sqlalchemy import create_engine, Column, Integer, Text, TIMESTAMP  # SQLAlchemy DB setup
from sqlalchemy.ext.declarative import declarative_base                 # For creating DB models
from sqlalchemy.orm import sessionmaker                               # For DB sessions
from typing import List                    # Type hinting for lists

# --- DATABASE CONFIGURATION ---

# Connection string to local Postgres database
# DATABASE_URL = "postgresql://zabe:@localhost:5432/askai"

from urllib.parse import quote_plus
load_dotenv() 


DATABASE_USER = os.getenv("DATABASE_USER")  # e.g., 'postgres'
DATABASE_PASSWORD = quote_plus(os.getenv("DATABASE_PASSWORD"))  # Encodes special chars safely
DATABASE_HOST = os.getenv("DATABASE_HOST")  # e.g., 'askai-db-server.postgres.database.azure.com'
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME")  # e.g., 'askai'

DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# Create DB engine and session factory
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Base class for creating models
Base = declarative_base()

# Define the structure of the "conversation_history" table
class Conversation(Base):
    __tablename__ = "conversation_history"  # Table name in PostgreSQL

    id = Column(Integer, primary_key=True, index=True)   # Unique ID
    user_input = Column(Text, nullable=False)            # What the user asked
    ai_response = Column(Text, nullable=False)           # What GPT responded
    created_at = Column(TIMESTAMP, server_default="now()")  # Timestamp when row is added

# Create the table if it doesn’t already exist
Base.metadata.create_all(bind=engine)

# --- OPENAI SETUP ---

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Initialize OpenAI client

# --- FASTAPI APP ---

# Create the FastAPI app instance
app = FastAPI()

# Define what kind of input the /ask endpoint should expect
class Prompt(BaseModel):
    user_input: str  # Incoming JSON must have a string field called `user_input`

# Root route to verify the server is running
@app.get("/")
def read_root():
    return {"message": "Welcome to AskAI"}

# POST /ask — Accepts user input and returns AI-generated response
@app.post("/ask")
def ask_ai(prompt: Prompt):
    try:
        # Call the OpenAI API with system + user messages
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.user_input}
            ]
        )
        ai_message = response.choices[0].message.content  # Extract generated response text

        # Save the interaction to the database
        db = SessionLocal()
        new_convo = Conversation(user_input=prompt.user_input, ai_response=ai_message)
        db.add(new_convo)
        db.commit()
        db.close()

        return {"response": ai_message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# GET /history — Returns a list of all past conversations
@app.get("/history")
def get_conversation_history():
    try:
        db = SessionLocal()
        conversations = db.query(Conversation).order_by(Conversation.id.desc()).all()
        db.close()

        # Format each conversation into a dictionary
        return [
            {
                "id": convo.id,
                "user_input": convo.user_input,
                "ai_response": convo.ai_response,
                "created_at": convo.created_at
            }
            for convo in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# DELETE /delete/{convo_id} — Deletes a specific conversation by ID
@app.delete("/delete/{convo_id}")
def delete_conversation(convo_id: int):
    db = SessionLocal()
    try:
        # Find the conversation with the matching ID
        convo = db.query(Conversation).filter(Conversation.id == convo_id).first()

        # If it doesn't exist, return a 404 error
        if convo is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Delete and commit
        db.delete(convo)
        db.commit()

        return {"message": f"Conversation with ID {convo_id} deleted successfully."}
    
    except Exception as e:
        db.rollback()  # Undo changes if something fails
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        db.close()  # Always close the DB session
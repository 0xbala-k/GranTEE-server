import json
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine, Column, String, JSON, Integer, Float
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from web3.auto import w3
from eth_account.messages import encode_defunct

# Database setup using SQLAlchemy
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ORM model for a user
class User(Base):
    __tablename__ = "users"
    wallet_address = Column(String, primary_key=True, index=True)
    data = Column(JSON)

# ORM model for a scholarship
class Scholarship(Base):
    __tablename__ = "scholarships"
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    max_amount_per_applicant = Column(Float, nullable=False)
    deadline = Column(String, nullable=False)
    applicants = Column(Integer, nullable=False, default=0)
    description = Column(String, nullable=False)
    requirements = Column(JSON, nullable=False)  # Stored as a list of strings

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="GranTEE Server")

# Configure CORS to allow requests from http://localhost:5173
origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Allowed origins
    allow_credentials=True,
    allow_methods=["*"],        # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],        # Allow all headers
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic model for request body when storing/updating user data
class UserData(BaseModel):
    wallet_address: str  # the user's wallet address
    signature: str       # signature produced by signing the message with the wallet's private key
    data: str            # the user's personal JSON data as string (name, age, socials, etc.)

# Endpoint to create or update a user's profile
@app.post("/user")
def create_or_update_user(user_data: UserData, db: Session = Depends(get_db)):
    # Recover the address from the signed message using the data string as the message
    encoded_message = encode_defunct(text=user_data.data)
    print("encoded_message", encoded_message)
    try:
        recovered_address = w3.eth.account.recover_message(encoded_message, signature=user_data.signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signature verification error: {e}")

    if recovered_address.lower() != user_data.wallet_address.lower():
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        # Convert the string data to a JSON/dictionary object
        json_data = json.loads(user_data.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON data: {e}")

    # Check if the user exists; if so, update, otherwise create new
    user = db.query(User).filter(User.wallet_address == user_data.wallet_address).first()
    if user:
        user.data = json_data
    else:
        user = User(wallet_address=user_data.wallet_address, data=json_data)
        db.add(user)

    db.commit()
    return {"message": "User data saved", "data": json_data}

# Endpoint to retrieve a user's profile
# For security, this endpoint also requires a valid signature.
@app.get("/user/{wallet_address}")
def get_user(wallet_address: str, signature: str, db: Session = Depends(get_db)):
    # Verify the signature using the wallet_address (converted to lower case) as the message
    encoded_message = encode_defunct(text=wallet_address.lower())
    try:
        recovered_address = w3.eth.account.recover_message(encoded_message, signature=signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signature verification error: {e}")

    if recovered_address.lower() != wallet_address.lower():
        raise HTTPException(status_code=401, detail="Invalid signature")
    user = db.query(User).filter(User.wallet_address == wallet_address).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"wallet_address": wallet_address, "data": user.data}

# Pydantic model for Scholarship data
class ScholarshipData(BaseModel):
    id: str
    title: str
    maxAmountPerApplicant: float
    deadline: str
    applicants: int
    description: str
    requirements: List[str]

# Helper function to convert a Scholarship ORM object to a dictionary
def scholarship_to_dict(s: Scholarship) -> dict:
    return {
        "id": s.id,
        "title": s.title,
        "maxAmountPerApplicant": s.max_amount_per_applicant,
        "deadline": s.deadline,
        "applicants": s.applicants,
        "description": s.description,
        "requirements": s.requirements,
    }

# Endpoint to upload (create) scholarship data (no authentication required)
@app.post("/scholarship")
def create_scholarship(scholarship_data: ScholarshipData, db: Session = Depends(get_db)):    
    # Check if the scholarship already exists
    scholarship = db.query(Scholarship).filter(Scholarship.id == scholarship_data.id).first()
    if scholarship:
        raise HTTPException(status_code=400, detail="Scholarship already exists")

    new_scholarship = Scholarship(
        id=scholarship_data.id,
        title=scholarship_data.title,
        max_amount_per_applicant=scholarship_data.maxAmountPerApplicant,
        deadline=scholarship_data.deadline,
        applicants=scholarship_data.applicants,
        description=scholarship_data.description,
        requirements=scholarship_data.requirements,
    )
    
    db.add(new_scholarship)
    
    db.commit()
    
    return {"message": "Scholarship created", "scholarship": scholarship_data.dict()}

# Endpoint to get a single scholarship by its id (no authentication required)
@app.get("/scholarship/{scholarship_id}")
def get_scholarship(scholarship_id: str, db: Session = Depends(get_db)):
    scholarship = db.query(Scholarship).filter(Scholarship.id == scholarship_id).first()
    if not scholarship:
        raise HTTPException(status_code=404, detail="Scholarship not found")
    return {"scholarship": scholarship_to_dict(scholarship)}
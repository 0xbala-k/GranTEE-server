import json
import os
from dotenv import load_dotenv
from llms.gemini import GeminiLLM
from rag.rag_api import RAGSimulator
import datetime
import uuid

from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, Column, String, JSON, Integer, Float
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from web3.auto import w3
from eth_account.messages import encode_defunct
import asyncio

# Import our custom modules
from agents.router import Router
from agents.question_answering import process_question
from agents.decision_making import process_decision
from agents.aggregator import VotingAggregator
from agents.decision_making import DecisionState


# Load environment variables
load_dotenv()

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

# ORM model for a scholarship application
class ScholarshipApplicationRecord(Base):
    __tablename__ = "applications"
    id = Column(String, primary_key=True, index=True)  # Auto-generated ID
    wallet_address = Column(String, index=True)
    scholarship_id = Column(String, index=True)
    student_data = Column(JSON)
    result = Column(JSON)  # Store the complete decision result
    decision = Column(String)  # "approved" or "rejected"
    confidence = Column(Float)
    created_at = Column(String)  # ISO format date string

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
    data: str # the user's personal JSON data as string (name, age, socials, etc.)

# Pydantic models for the new endpoints
class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class ScholarshipDecisionRequest(BaseModel):
    student_data: Dict[str, Any]
    scholarship_criteria: Dict[str, Any]
    query: Optional[str] = None

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

# Endpoint for homepage
@app.get("/")
async def root():
    return {"message": "Welcome to GranTEE Server with Q/A and Decision Making capabilities using a voting-based aggregation system"}

# Endpoint to create or update a user's profile
@app.post("/user")
def create_or_update_user(user_data: UserData, db: Session = Depends(get_db)):
    # Recover the address from the signed message
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
@app.get("/user/{wallet_address}")
def get_user(wallet_address: str, signature: str, db: Session = Depends(get_db)):
    # Verify the signature
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

# Endpoint to upload (create) scholarship data
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
    
    return {"message": "Scholarship created", "scholarship": scholarship_to_dict(new_scholarship)}

# Endpoint to get a single scholarship by ID
@app.get("/scholarship/{scholarship_id}")
def get_scholarship(scholarship_id: str, db: Session = Depends(get_db)):
    scholarship = db.query(Scholarship).filter(Scholarship.id == scholarship_id).first()
    if not scholarship:
        raise HTTPException(status_code=404, detail="Scholarship not found")
    return {"scholarship": scholarship_to_dict(scholarship)}

# Endpoint to get all scholarships
@app.get("/scholarships")
def get_scholarships(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    scholarships = db.query(Scholarship).offset(skip).limit(limit).all()
    return {"scholarships": [scholarship_to_dict(s) for s in scholarships]}

# Endpoint to update a scholarship
@app.put("/scholarship/{scholarship_id}")
def update_scholarship(scholarship_id: str, scholarship_data: ScholarshipData, db: Session = Depends(get_db)):
    scholarship = db.query(Scholarship).filter(Scholarship.id == scholarship_id).first()
    if not scholarship:
        raise HTTPException(status_code=404, detail="Scholarship not found")
    
    # Update scholarship attributes
    scholarship.title = scholarship_data.title
    scholarship.max_amount_per_applicant = scholarship_data.maxAmountPerApplicant
    scholarship.deadline = scholarship_data.deadline
    scholarship.applicants = scholarship_data.applicants
    scholarship.description = scholarship_data.description
    scholarship.requirements = scholarship_data.requirements
    
    db.commit()
    
    return {"message": "Scholarship updated", "scholarship": scholarship_to_dict(scholarship)}

# Endpoint to delete a scholarship
@app.delete("/scholarship/{scholarship_id}")
def delete_scholarship(scholarship_id: str, db: Session = Depends(get_db)):
    scholarship = db.query(Scholarship).filter(Scholarship.id == scholarship_id).first()
    if not scholarship:
        raise HTTPException(status_code=404, detail="Scholarship not found")
    
    db.delete(scholarship)
    db.commit()
    
    return {"message": "Scholarship deleted"}

# Helper function to convert a ScholarshipApplicationRecord to a dictionary
def application_to_dict(app: ScholarshipApplicationRecord) -> dict:
    return {
        "id": app.id,
        "wallet_address": app.wallet_address,
        "scholarship_id": app.scholarship_id,
        "decision": app.decision,
        "confidence": app.confidence,
        "created_at": app.created_at
    }

# Endpoint to get user's scholarship applications
@app.get("/applications/{wallet_address}")
def get_user_applications(wallet_address: str, signature: str, db: Session = Depends(get_db)):
    # Verify the signature using the wallet_address as the message
    encoded_message = encode_defunct(text=wallet_address.lower())
    try:
        recovered_address = w3.eth.account.recover_message(encoded_message, signature=signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signature verification error: {e}")

    if recovered_address.lower() != wallet_address.lower():
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Get all applications for the user
    applications = db.query(ScholarshipApplicationRecord).filter(
        ScholarshipApplicationRecord.wallet_address == wallet_address
    ).all()
    
    # Get the corresponding scholarship details for each application
    result = []
    for app in applications:
        scholarship = db.query(Scholarship).filter(Scholarship.id == app.scholarship_id).first()
        if scholarship:
            app_dict = application_to_dict(app)
            app_dict["scholarship"] = scholarship_to_dict(scholarship)
            result.append(app_dict)
    
    return {"applications": result}

# Get a specific application
@app.get("/application/{application_id}")
def get_application(application_id: str, wallet_address: str, signature: str, db: Session = Depends(get_db)):
    # Verify the signature using the application_id as the message
    encoded_message = encode_defunct(text=application_id)
    try:
        recovered_address = w3.eth.account.recover_message(encoded_message, signature=signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signature verification error: {e}")

    if recovered_address.lower() != wallet_address.lower():
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Get the application
    application = db.query(ScholarshipApplicationRecord).filter(
        ScholarshipApplicationRecord.id == application_id,
        ScholarshipApplicationRecord.wallet_address == wallet_address
    ).first()
    
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Get the scholarship details
    scholarship = db.query(Scholarship).filter(Scholarship.id == application.scholarship_id).first()
    if not scholarship:
        raise HTTPException(status_code=404, detail="Scholarship not found")
    
    # Return the full application details including the decision result
    result = {
        "application": application_to_dict(application),
        "scholarship": scholarship_to_dict(scholarship),
        "student_data": application.student_data,
        "result": application.result
    }
    
    return result

# Model for scholarship application
class ScholarshipApplication(BaseModel):
    wallet_address: str
    scholarship_id: str
    student_data: Dict[str, Any]
    signature: str

# Endpoint to apply for a scholarship
@app.post("/apply")
async def apply_for_scholarship(application: ScholarshipApplication, db: Session = Depends(get_db)):
    # Verify the user's signature
    encoded_message = encode_defunct(text=json.dumps(application.student_data))
    try:
        recovered_address = w3.eth.account.recover_message(encoded_message, signature=application.signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signature verification error: {e}")

    if recovered_address.lower() != application.wallet_address.lower():
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Check if the scholarship exists
    scholarship = db.query(Scholarship).filter(Scholarship.id == application.scholarship_id).first()
    if not scholarship:
        raise HTTPException(status_code=404, detail="Scholarship not found")
    
    # Process the scholarship application through our decision making pipeline
    try:
        # Convert the scholarship to the expected format for the decision API
        scholarship_criteria = {
            "name": scholarship.title,
            "description": scholarship.description,
            "requirements": scholarship.requirements,
            "amount": scholarship.max_amount_per_applicant
        }
        
        # Process the decision
        result = await process_decision(application.student_data, scholarship_criteria)
        
        # Safely access vote_count and vote_breakdown
        vote_count = result.get("vote_count", {}) or {}
        vote_breakdown = result.get("vote_breakdown", {}) or {}
        
        # Add voting details
        result["vote_details"] = {
            "total_votes": result.get("total_votes", 0) or sum(vote_count.values()) if vote_count else 0,
            "approval_votes": vote_count.get("approve", 0),
            "rejection_votes": vote_count.get("reject", 0),
            "confidence": result.get("confidence", 0.0) or 0.0,
            "decision_threshold": 0.5,  # Majority vote
            "agent_breakdown": vote_breakdown.get("by_agent", {}),
            "source_breakdown": vote_breakdown.get("by_source", {})
        }
        
        # Update the applicant count
        scholarship.applicants += 1
        db.commit()
        
        # Generate a unique ID and timestamp for the application
        application_id = str(uuid.uuid4())
        current_time = datetime.datetime.now().isoformat()
        
        # Add timestamp to the result
        result["created_at"] = current_time
        
        # Save the application result to the database
        application_record = ScholarshipApplicationRecord(
            id=application_id,
            wallet_address=application.wallet_address,
            scholarship_id=application.scholarship_id,
            student_data=application.student_data,
            result=result,
            decision="approved" if result.get("decision", False) else "rejected",
            confidence=result.get("confidence", 0.0),
            created_at=current_time
        )
        db.add(application_record)
        db.commit()
        
        # Add application ID to the result
        result["application_id"] = application_id
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing application: {str(e)}")

# New endpoint for routing a query
@app.post("/query")
async def route_query(request: QueryRequest):
    """
    Route a query to either the question answering or decision making pipeline.
    Uses a voting system across multiple RAG sources for higher accuracy.
    """
    try:
        # Initialize the router
        router = Router()
        
        # Route the query
        route_result = await router.route_query(request.query, request.context)
        
        # Process the query based on the routing decision
        if route_result["route"] == "decision":
            # If it's a decision, extract student data and scholarship criteria
            student_data = route_result["student_data"]
            scholarship_criteria = route_result["scholarship_criteria"]
            
            # Process the decision
            result = await process_decision(student_data, scholarship_criteria)
            
            # Add voting information
            result["vote_details"] = {
                "total_votes": sum(result.get("vote_count", {}).values()),
                "approval_votes": result.get("vote_count", {}).get("approve", 0),
                "rejection_votes": result.get("vote_count", {}).get("reject", 0)
            }
        else:
            # If it's a question, process it with the Q/A pipeline
            result = await process_question(request.query)
        
        # Add the routing information to the result
        result["routed_to"] = route_result["route"]
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Endpoint for direct question answering
@app.post("/question")
async def ask_question(request: QueryRequest):
    """
    Process a question directly through the Q/A pipeline.
    Uses a voting system across multiple RAG sources for higher accuracy.
    """
    try:
        result = await process_question(request.query)
        
        # Add extra context about the voting system
        result["voting_info"] = {
            "description": "Results are aggregated from multiple RAG sources using a weighted voting system",
            "confidence": result.get("confidence", 0.0),
            "sources_count": len(result.get("sources", []))
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Endpoint for direct decision making
@app.post("/decision")
async def make_decision(request: ScholarshipDecisionRequest):
    """
    Process a scholarship decision directly through the decision making pipeline.
    Uses a multi-agent voting system for more robust decisions.
    """
    try:
        result = await process_decision(request.student_data, request.scholarship_criteria)
        
        if result is None:
            raise HTTPException(status_code=500, detail="No result returned from decision process")
        
        # Safely access vote_count and vote_breakdown
        vote_count = result.get("vote_count", {}) or {}
        vote_breakdown = result.get("vote_breakdown", {}) or {}
        
        # Add voting details
        result["vote_details"] = {
            "total_votes": result.get("total_votes", 0) or sum(vote_count.values()) if vote_count else 0,
            "approval_votes": vote_count.get("approve", 0),
            "rejection_votes": vote_count.get("reject", 0),
            "confidence": result.get("confidence", 0.0) or 0.0,
            "decision_threshold": 0.5,  # Majority vote
            "agent_breakdown": vote_breakdown.get("by_agent", {}),
            "source_breakdown": vote_breakdown.get("by_source", {})
        }
        
        return result
    except Exception as e:
        # Log the full error for debugging
        import traceback
        print(f"Decision error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error making decision: {str(e)}")

def setup_models():
    """Set up three different analyzer models with different perspectives"""
    # Set up three different analyzer models with different perspectives
    analyzer_models = {
        "academic": GeminiLLM(model_name="gemini-1.5-pro"),  # Focus on academic metrics
        "holistic": GeminiLLM(model_name="gemini-1.5-pro"),  # Focus on whole-person evaluation
        "equity": GeminiLLM(model_name="gemini-1.5-flash")    # Focus on equity considerations
    }
    
    decision_model = GeminiLLM(model_name="gemini-2.0-flash-lite-preview-02-05")
    explainer_model = GeminiLLM(model_name="gemini-1.5-flash")
    
    # Set up RAG simulator
    rag_simulator = RAGSimulator()
    
    return analyzer_models, decision_model, explainer_model, rag_simulator

async def analyze_student_eligibility(state: DecisionState) -> DecisionState:
    """Analyze student eligibility using 3 agents x 3 RAG sources = 9 votes"""
    try:
        # Check if we have RAG results
        if not state.get("rag_results"):
            return {**state, "error": "No information available for analysis."}
        
        # Get all analyzer models
        analyzer_models, _, _, _ = setup_models()
        
        # Create a grid of 9 analysis tasks (3 agents × 3 RAG sources)
        analysis_tasks = []
        
        # For each agent perspective
        for agent_type, analyzer_model in analyzer_models.items():
            # For each RAG source
            for rag_result in state["rag_results"]:
                if rag_result.get("success", False):
                    system_prompt = _get_agent_system_prompt(agent_type)
                    
                    task = analyze_with_agent_and_rag(
                        agent_type=agent_type,
                        rag_data=rag_result, 
                        student_data=state["student_data"], 
                        scholarship_criteria=state["scholarship_criteria"],
                        analyzer_model=analyzer_model,
                        system_prompt=system_prompt
                    )
                    analysis_tasks.append(task)
        
        # Run all 9 analyses in parallel
        analysis_results = await asyncio.gather(*analysis_tasks)
        
        # If no successful analyses, return error
        if not analysis_results:
            return {**state, "error": "No successful analyses completed."}
        
        # Combine analyses using the aggregator
        aggregator = VotingAggregator()
        combined_analysis = aggregator.combine_analyses(
            [r["analysis"] for r in analysis_results]
        )
        
        # Return the combined analysis and individual votes
        return {
            **state, 
            "analysis": combined_analysis,
            "analysis_votes": analysis_results,
            "total_votes": len(analysis_results)  # Should be 9 when all succeed
        }
    except Exception as e:
        return {**state, "error": f"Error analyzing eligibility: {str(e)}"}

async def analyze_with_agent_and_rag(
    agent_type: str,
    rag_data: Dict[str, Any], 
    student_data: Dict[str, Any], 
    scholarship_criteria: Dict[str, Any],
    analyzer_model: GeminiLLM,
    system_prompt: str
) -> Dict[str, Any]:
    """Analyze using a specific agent type and RAG source"""
    try:
        # Extract the answer and sources
        answer = rag_data.get("answer", "")
        sources = rag_data.get("sources", [])
        rag_source = rag_data.get("source", "Unknown RAG")
        
        # Create context strings
        sources_context = ""
        for i, source in enumerate(sources[:3]):
            source_text = source.get("text", "")
            source_meta = source.get("metadata", {})
            source_score = source.get("score", 0)
            sources_context += f"Source {i+1} (Relevance: {source_score:.2f}): {source_text}\n"
            sources_context += f"Metadata: {json.dumps(source_meta)}\n\n"
            
        # Convert student and scholarship data to formatted strings
        student_context = json.dumps(student_data, indent=2)
        scholarship_context = json.dumps(scholarship_criteria, indent=2)
        
        # Create a prompt tailored to this agent's perspective
        prompt = f"""
        As the {agent_type.upper()} EVALUATOR, analyze this student's eligibility for the scholarship:
        
        STUDENT DATA:
        {student_context}
        
        SCHOLARSHIP CRITERIA:
        {scholarship_context}
        
        RAG INFORMATION:
        {answer}
        
        SOURCES:
        {sources_context}
        
        INSTRUCTIONS FOR {agent_type.upper()} EVALUATOR:
        {_get_agent_specific_instructions(agent_type)}
        
        Provide a detailed analysis of how well the student meets each criterion from your {agent_type} perspective.
        End with a preliminary YES or NO recommendation.
        """
        
        # Get the analysis from the model with the agent-specific system prompt
        analysis = await analyzer_model.generate(prompt, system_prompt=system_prompt)
        
        # Extract preliminary recommendation
        recommendation = "YES" if "YES" in analysis[-200:].upper() else "NO"
        
        return {
            "agent_type": agent_type,
            "rag_source": rag_source,
            "analysis": analysis,
            "recommendation": recommendation,
            "decision": recommendation == "YES",
            "confidence": max(0.6, min(0.9, rag_data.get("confidence", 0.0) * 1.2))
        }
    except Exception as e:
        return {
            "agent_type": agent_type,
            "rag_source": rag_source,
            "analysis": f"Error analyzing eligibility: {str(e)}",
            "recommendation": "NO",  # Default to NO on error
            "decision": False,
            "confidence": 0.0
        }

def _get_agent_specific_instructions(agent_type: str) -> str:
    """Get perspective-specific instructions for each agent type"""
    if agent_type == "academic":
        return """
        Focus on academic metrics:
        - Prioritize GPA and academic achievements
        - Evaluate academic trajectory and rigor of coursework
        - Consider test scores and academic honors
        - Assess academic potential for success
        """
    elif agent_type == "holistic":
        return """
        Focus on whole-person evaluation:
        - Consider extracurricular activities and leadership roles
        - Evaluate community service and impact
        - Assess personal challenges overcome
        - Consider unique talents or perspectives
        - Look at the personal essay and character qualities
        """
    elif agent_type == "equity":
        return """
        Focus on equity considerations:
        - Consider barriers the student may have overcome
        - Evaluate socioeconomic factors in context
        - Consider first-generation status if applicable
        - Assess potential for contribution to campus diversity
        - Consider access to resources in their educational background
        """
    else:
        return "Provide an objective analysis of the student's eligibility."

def _get_agent_system_prompt(agent_type: str) -> str:
    """Get perspective-specific system prompt for each agent type"""
    if agent_type == "academic":
        return "You are an academic evaluator who prioritizes scholastic achievement and metrics."
    elif agent_type == "holistic":
        return "You are a holistic evaluator who considers the whole person beyond just academic metrics."
    elif agent_type == "equity":
        return "You are an equity-focused evaluator who considers barriers students have overcome and background contexts."
    else:
        return "You are an objective scholarship evaluator."

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
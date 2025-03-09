# Server side (TEE) implementation for GranTEE

## Setup and run
### Local
1. Create python env
```bash
python -m venv venv
source venv/bin/activate
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Set up environment variables
```bash
cp .env.example .env
# Edit .env file and add your Google API key
```
4. Run server
```bash
uvicorn main:app --reload
```

## API Endpoints

### Intelligent Routing
This endpoint automatically determines whether to treat the query as a general question or a scholarship decision request.

**Endpoint:** `POST /query`

**Request:**
```json
{
  "query": "Your question or decision query here",
  "context": {
    "student_data": {
      "name": "John Doe",
      "gpa": 3.8,
      "background": "First-generation student"
    },
    "scholarship_criteria": {
      "minimum_gpa": 3.5,
      "preference": "First-generation students"
    }
  }
}
```
Note: The `context` field is optional for questions but recommended for decision queries.

### Direct Question Answering
Use this endpoint to directly ask questions without routing.

**Endpoint:** `POST /question`

**Request:**
```json
{
  "query": "What are the eligibility requirements for the Merit Scholarship?"
}
```

### Scholarship Decision Making
Use this endpoint to directly evaluate a student for scholarship eligibility.

**Endpoint:** `POST /decision`

**Request:**
```json
{
  "student_data": {
    "name": "Jane Smith",
    "gpa": 3.7,
    "background": "First-generation student",
    "activities": ["Student government", "Volunteer work"]
  },
  "scholarship_criteria": {
    "name": "Merit Scholarship",
    "minimum_gpa": 3.5,
    "preference": "Leadership experience",
    "amount": "$10,000"
  }
}
```

## Implementation Details

This system uses:

1. **LangGraph**: For creating the agent workflows for question answering and decision making
2. **Gemini LLMs**: Three different Gemini models for different tasks:
   - `gemini-1.5-flash`: For routing and lightweight tasks
   - `gemini-1.5-pro`: For detailed analysis and synthesis
   - `gemini-1.0-pro`: For generating student-friendly explanations
3. **Integrated RAG APIs**: Three external RAG APIs for comprehensive knowledge retrieval:
   - **Deep Search RAG** (cl-rag.onrender.com): Provides in-depth information with up to 15 sources
   - **Community Search RAG** (cl-rag-community.onrender.com): Provides community-based knowledge with up to 10 sources
   - **Fast Search RAG** (cl-rag-fast.onrender.com): Provides quick responses with up to 5 sources
4. **Multi-Agent Voting System**: Nine parallel votes (3 agent types × 3 RAG sources) for more robust decisions

### Agent Types
The system uses three types of agents to evaluate scholarship applications:
- **Academic Agent**: Focuses on academic metrics, GPA, and test scores
- **Holistic Agent**: Evaluates the whole person including extracurriculars and personal qualities
- **Equity Agent**: Considers barriers overcome, background contexts, and diversity contributions

The final decision is made through a weighted voting system that aggregates insights from all agents and RAG sources.

The system architecture separates concerns into different modules for easier maintenance and extension.

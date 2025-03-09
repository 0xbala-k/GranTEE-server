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
    "gpa": 3.9,
    "background": "First-generation college student",
    "activities": ["Student Government President", "Volunteer at Local Hospital"],
    "essay": "I am committed to using my education to serve underrepresented communities..."
  },
  "scholarship_criteria": {
    "name": "Merit Scholarship",
    "requirements": ["GPA above 3.5", "Demonstrated leadership experience"],
    "amount": 10000
  }
}
```

### Scholarship Management
The application includes a complete set of endpoints for managing scholarships.

#### Create a Scholarship
**Endpoint:** `POST /scholarship`

**Example Request:**
```json
{
  "id": "merit-2023",
  "title": "Merit Scholarship 2023",
  "maxAmountPerApplicant": 10000,
  "deadline": "2023-12-31",
  "applicants": 0,
  "description": "Scholarship for high-achieving students with leadership potential",
  "requirements": [
    "GPA above 3.5",
    "Demonstrated leadership experience",
    "Community service"
  ]
}
```

#### Get All Scholarships
**Endpoint:** `GET /scholarships`

**Query Parameters:**
- `skip` (optional): Number of records to skip for pagination (default: 0)
- `limit` (optional): Maximum number of records to return (default: 100)

#### Get a Specific Scholarship
**Endpoint:** `GET /scholarship/{scholarship_id}`

#### Update a Scholarship
**Endpoint:** `PUT /scholarship/{scholarship_id}`

Request body is the same as the create endpoint.

#### Delete a Scholarship
**Endpoint:** `DELETE /scholarship/{scholarship_id}`

#### Apply for a Scholarship
**Endpoint:** `POST /apply`

**Example Request:**
```json
{
  "wallet_address": "0x123abc...",
  "scholarship_id": "merit-2023",
  "student_data": {
    "name": "Jane Smith",
    "gpa": 3.9,
    "background": "First-generation college student",
    "activities": ["Student Government President", "Volunteer at Local Hospital"],
    "essay": "I am committed to using my education to serve underrepresented communities..."
  },
  "signature": "0xabcdef..." 
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

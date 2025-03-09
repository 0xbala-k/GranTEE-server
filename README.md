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

### Deployment to Render
For detailed instructions on deploying to Render.com, see the [DEPLOYMENT.md](DEPLOYMENT.md) file.

Quick steps:
1. Connect your repository to Render
2. Create a new Web Service with Docker runtime
3. Set the required environment variables (GOOGLE_API_KEY)
4. Deploy and access your API at the provided URL

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

### User Management Endpoints
- `POST /user` - Create or update a user profile
- `GET /user/{wallet_address}` - Get a user's profile

### Scholarship Management Endpoints
- `POST /scholarship` - Create a new scholarship
- `GET /scholarships` - Get all scholarships
- `GET /scholarship/{scholarship_id}` - Get a specific scholarship
- `PUT /scholarship/{scholarship_id}` - Update a scholarship
- `DELETE /scholarship/{scholarship_id}` - Delete a scholarship

### Application Endpoints
- `POST /apply` - Apply for a scholarship (requires wallet signature)
- `GET /applications/{wallet_address}` - Get all applications for a user
- `GET /application/{application_id}` - Get details of a specific application

For detailed request/response formats for all endpoints, see [sample_queries.md](sample_queries.md).

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
5. **Blockchain Integration**: Secure user and application management through wallet signatures

### Agent Types
The system uses three types of agents to evaluate scholarship applications:
- **Academic Agent**: Focuses on academic metrics, GPA, and test scores
- **Holistic Agent**: Evaluates the whole person including extracurriculars and personal qualities
- **Equity Agent**: Considers barriers overcome, background contexts, and diversity contributions

The final decision is made through a weighted voting system that aggregates insights from all agents and RAG sources.

The system architecture separates concerns into different modules for easier maintenance and extension.

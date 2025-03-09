# Sample Queries for GranTEE Server

This document contains sample queries for testing each of the endpoints in the GranTEE server.

## User Management Endpoints

### Create or Update User Profile

**Request to `POST /user`**
```bash
curl -X POST https://grantee-server.onrender.com/user \
  -H "Content-Type: application/json" \
  -d '{
    "wallet_address": "0x123abc...",
    "signature": "0xdef456...",
    "data": "{\"name\":\"John Doe\",\"age\":20,\"major\":\"Computer Science\",\"gpa\":3.8,\"background\":\"First-generation student\",\"activities\":[\"Chess Club\",\"Volunteer Tutor\"],\"achievements\":[\"Dean's List 2023\",\"Hackathon Winner\"]}"
  }'
```

**Expected Response:**
```json
{
  "message": "User data saved",
  "data": {
    "name": "John Doe",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.8,
    "background": "First-generation student",
    "activities": ["Chess Club", "Volunteer Tutor"],
    "achievements": ["Dean's List 2023", "Hackathon Winner"]
  }
}
```

### Get User Profile

**Request to `GET /user/{wallet_address}`**
```bash
curl "https://grantee-server.onrender.com/user/0x123abc...?signature=0xdef456..."
```

**Expected Response:**
```json
{
  "wallet_address": "0x123abc...",
  "data": {
    "name": "John Doe",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.8,
    "background": "First-generation student",
    "activities": ["Chess Club", "Volunteer Tutor"],
    "achievements": ["Dean's List 2023", "Hackathon Winner"]
  }
}
```

## Scholarship Management Endpoints

### Create a Scholarship

**Request to `POST /scholarship`**
```bash
curl -X POST https://grantee-server.onrender.com/scholarship \
  -H "Content-Type: application/json" \
  -d '{
    "id": "merit-2023",
    "title": "Merit Scholarship 2023",
    "maxAmountPerApplicant": 10000,
    "deadline": "2023-12-31",
    "applicants": 0,
    "description": "Scholarship for high achieving students with leadership potential",
    "requirements": [
      "GPA above 3.5",
      "Demonstrated leadership experience",
      "Community service"
    ]
  }'
```

**Expected Response:**
```json
{
  "message": "Scholarship created",
  "scholarship": {
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
}
```

### Get All Scholarships

**Request to `GET /scholarships`**
```bash
curl "https://grantee-server.onrender.com/scholarships"
```

**Expected Response:**
```json
{
  "scholarships": [
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
    },
    {
      "id": "need-based-2023",
      "title": "Need-Based Scholarship 2023",
      "maxAmountPerApplicant": 5000,
      "deadline": "2023-12-15",
      "applicants": 0,
      "description": "Financial assistance for students demonstrating financial need",
      "requirements": [
        "Demonstrated financial need",
        "GPA above 3.0",
        "Full-time enrollment"
      ]
    }
  ]
}
```

### Get a Specific Scholarship

**Request to `GET /scholarship/{scholarship_id}`**
```bash
curl "https://grantee-server.onrender.com/scholarship/merit-2023"
```

**Expected Response:**
```json
{
  "scholarship": {
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
}
```

### Update a Scholarship

**Request to `PUT /scholarship/{scholarship_id}`**
```bash
curl -X PUT https://grantee-server.onrender.com/scholarship/merit-2023 \
  -H "Content-Type: application/json" \
  -d '{
    "id": "merit-2023",
    "title": "Merit Scholarship 2023 (Updated)",
    "maxAmountPerApplicant": 12000,
    "deadline": "2024-01-15",
    "applicants": 0,
    "description": "Scholarship for exceptional students with leadership qualities",
    "requirements": [
      "GPA above 3.5",
      "Demonstrated leadership experience",
      "Community service",
      "Two recommendation letters"
    ]
  }'
```

**Expected Response:**
```json
{
  "message": "Scholarship updated",
  "scholarship": {
    "id": "merit-2023",
    "title": "Merit Scholarship 2023 (Updated)",
    "maxAmountPerApplicant": 12000,
    "deadline": "2024-01-15",
    "applicants": 0,
    "description": "Scholarship for exceptional students with leadership qualities",
    "requirements": [
      "GPA above 3.5",
      "Demonstrated leadership experience",
      "Community service",
      "Two recommendation letters"
    ]
  }
}
```

### Delete a Scholarship

**Request to `DELETE /scholarship/{scholarship_id}`**
```bash
curl -X DELETE "https://grantee-server.onrender.com/scholarship/merit-2023"
```

**Expected Response:**
```json
{
  "message": "Scholarship deleted"
}
```

### Apply for a Scholarship

**Request to `POST /apply`**
```bash
curl -X POST https://grantee-server.onrender.com/apply \
  -H "Content-Type: application/json" \
  -d '{
    "wallet_address": "0x123abc...",
    "scholarship_id": "merit-2023",
    "student_data": {
      "name": "Jane Smith",
      "gpa": 3.9,
      "background": "First-generation college student",
      "activities": ["Student Government President", "Volunteer at Local Hospital"],
      "essay": "I am committed to using my education to serve underrepresented communities..."
    },
    "signature": "0xdef456..."
  }'
```

**Expected Response:**
```json
{
  "decision": true,
  "confidence": 0.78,
  "explanation": "Your application for the Merit Scholarship has been APPROVED. Your GPA of 3.9 exceeds our requirement of 3.5, and your leadership role as Student Government President demonstrates the leadership qualities we seek. Your essay shows a strong commitment to community service, which aligns with our scholarship's values.",
  "vote_details": {
    "total_votes": 9,
    "approval_votes": 7,
    "rejection_votes": 2,
    "confidence": 0.78,
    "decision_threshold": 0.5,
    "agent_breakdown": {
      "academic": {
        "approve": 3,
        "reject": 0
      },
      "holistic": {
        "approve": 2,
        "reject": 1
      },
      "equity": {
        "approve": 2,
        "reject": 1
      }
    },
    "source_breakdown": {
      "deep_search": {
        "approve": 3,
        "reject": 0
      },
      "community_search": {
        "approve": 2,
        "reject": 1
      },
      "fast_search": {
        "approve": 2,
        "reject": 1
      }
    }
  },
  "application_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2023-09-15T14:30:45.123456"
}
```

### Get User's Applications

**Request to `GET /applications/{wallet_address}`**
```bash
curl "https://grantee-server.onrender.com/applications/0x123abc...?signature=0xdef456..."
```

**Expected Response:**
```json
{
  "applications": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "wallet_address": "0x123abc...",
      "scholarship_id": "merit-2023",
      "decision": "approved",
      "confidence": 0.78,
      "created_at": "2023-09-15T14:30:45.123456",
      "scholarship": {
        "id": "merit-2023",
        "title": "Merit Scholarship 2023",
        "maxAmountPerApplicant": 10000,
        "deadline": "2023-12-31",
        "applicants": 1,
        "description": "Scholarship for high-achieving students with leadership potential",
        "requirements": [
          "GPA above 3.5",
          "Demonstrated leadership experience",
          "Community service"
        ]
      }
    }
  ]
}
```

### Get Specific Application

**Request to `GET /application/{application_id}`**
```bash
curl "https://grantee-server.onrender.com/application/550e8400-e29b-41d4-a716-446655440000?wallet_address=0x123abc...&signature=0xdef456..."
```

**Expected Response:**
```json
{
  "application": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "wallet_address": "0x123abc...",
    "scholarship_id": "merit-2023",
    "decision": "approved",
    "confidence": 0.78,
    "created_at": "2023-09-15T14:30:45.123456"
  },
  "scholarship": {
    "id": "merit-2023",
    "title": "Merit Scholarship 2023",
    "maxAmountPerApplicant": 10000,
    "deadline": "2023-12-31",
    "applicants": 1,
    "description": "Scholarship for high-achieving students with leadership potential",
    "requirements": [
      "GPA above 3.5",
      "Demonstrated leadership experience",
      "Community service"
    ]
  },
  "student_data": {
    "name": "Jane Smith",
    "gpa": 3.9,
    "background": "First-generation college student",
    "activities": ["Student Government President", "Volunteer at Local Hospital"],
    "essay": "I am committed to using my education to serve underrepresented communities..."
  },
  "result": {
    "decision": true,
    "confidence": 0.78,
    "explanation": "Your application for the Merit Scholarship has been APPROVED. Your GPA of 3.9 exceeds our requirement of 3.5, and your leadership role as Student Government President demonstrates the leadership qualities we seek. Your essay shows a strong commitment to community service, which aligns with our scholarship's values.",
    "vote_details": {
      "total_votes": 9,
      "approval_votes": 7,
      "rejection_votes": 2,
      "confidence": 0.78,
      "decision_threshold": 0.5,
      "agent_breakdown": {
        "academic": {
          "approve": 3,
          "reject": 0
        },
        "holistic": {
          "approve": 2,
          "reject": 1
        },
        "equity": {
          "approve": 2,
          "reject": 1
        }
      }
    }
  }
}
```

## AI Decision and Query Endpoints

### General Query (Auto-routing)

**Request to `POST /query`**
```bash
curl -X POST https://grantee-server.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the eligibility requirements for the Merit Scholarship program?"
  }'
```

**Expected Response:**
```json
{
  "query": "What are the eligibility requirements for the Merit Scholarship program?",
  "response": "The Merit Scholarship requires a GPA above 3.5 and demonstrated leadership experience. The program provides $10,000 in financial support for qualified students. Additionally, applicants must demonstrate community service involvement.",
  "sources": [
    {
      "text": "Merit Scholarship requires GPA above 3.5, leadership experience and provides $10,000 in funding.",
      "score": 0.92,
      "metadata": {
        "Source": "Scholarship Database 2025",
        "filename": "Scholarship_Guide"
      }
    },
    {
      "text": "Merit Scholarship applicants must show evidence of community service and leadership roles.",
      "score": 0.85,
      "metadata": {
        "Source": "Scholarship Requirements Guide"
      }
    }
  ],
  "confidence": 0.89,
  "routed_to": "question"
}
```

### Direct Question Answering

**Request to `POST /question`**
```bash
curl -X POST https://grantee-server.onrender.com/question \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the eligibility requirements for the Merit Scholarship program?"
  }'
```

**Expected Response:**
```json
{
  "query": "What are the eligibility requirements for the Merit Scholarship program?",
  "response": "The Merit Scholarship requires a GPA above 3.5 and demonstrated leadership experience. The program provides $10,000 in financial support for qualified students. Additionally, applicants must demonstrate community service involvement.",
  "sources": [
    {
      "text": "Merit Scholarship requires GPA above 3.5, leadership experience and provides $10,000 in funding.",
      "score": 0.92,
      "metadata": {
        "Source": "Scholarship Database 2025",
        "filename": "Scholarship_Guide"
      }
    },
    {
      "text": "Merit Scholarship applicants must show evidence of community service and leadership roles.",
      "score": 0.85,
      "metadata": {
        "Source": "Scholarship Requirements Guide"
      }
    }
  ],
  "confidence": 0.89,
  "voting_info": {
    "description": "Results are aggregated from multiple RAG sources using a weighted voting system",
    "confidence": 0.89,
    "sources_count": 2
  }
}
```

### Scholarship Decision Making

**Request to `POST /decision`**
```bash
curl -X POST https://grantee-server.onrender.com/decision \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Expected Response:**
```json
{
  "decision": true,
  "confidence": 0.78,
  "explanation": "Your application for the Merit Scholarship has been APPROVED. Your GPA of 3.9 exceeds our requirement of 3.5, and your leadership role as Student Government President demonstrates the leadership qualities we seek. Your essay shows a strong commitment to community service, which aligns with our scholarship's values.",
  "vote_details": {
    "total_votes": 9,
    "approval_votes": 7,
    "rejection_votes": 2,
    "confidence": 0.78,
    "decision_threshold": 0.5,
    "agent_breakdown": {
      "academic": {
        "approve": 3,
        "reject": 0
      },
      "holistic": {
        "approve": 2,
        "reject": 1
      },
      "equity": {
        "approve": 2,
        "reject": 1
      }
    },
    "source_breakdown": {
      "deep_search": {
        "approve": 3,
        "reject": 0
      },
      "community_search": {
        "approve": 2,
        "reject": 1
      },
      "fast_search": {
        "approve": 2,
        "reject": 1
      }
    }
  }
}
``` 
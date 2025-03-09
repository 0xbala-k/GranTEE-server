# Sample Queries for GranTEE Server

This document contains sample queries for testing each of the endpoints in the GranTEE server.

## 1. General Query Endpoint (Auto-routing)

### Example 1: Question about scholarship requirements

**Request to `POST /query`**
```json
{
  "query": "What are the eligibility requirements for the Merit Scholarship program?"
}
```

**Expected Response:**
```json
{
  "query": "What are the eligibility requirements for the Merit Scholarship program?",
  "response": "The Merit Scholarship requires a GPA above 3.5 and demonstrated leadership experience. The program provides $10,000 in financial support for qualified students. To be eligible, students must maintain a minimum GPA of 3.0, be enrolled full-time, demonstrate financial need, show community involvement, and submit recommendation letters.",
  "sources": [
    {
      "text": "Merit Scholarship requires GPA above 3.5, leadership experience and provides $10,000 in funding.",
      "score": 0.92,
      "metadata": {
        "Source": "Scholarship Database 2025",
        "filename": "Scholarship_Guide",
        "last_updated": "2025-03-08"
      }
    },
    {
      "text": "Minimum GPA of 3.0",
      "score": 0.94,
      "metadata": {
        "Source": "Scholarship Requirements Guide",
        "filename": "Eligibility_Criteria",
        "last_updated": "2025-03-08"
      }
    }
  ],
  "error": null,
  "source": "question_answering",
  "routed_to": "question"
}
```

### Example 2: Decision-making query

**Request to `POST /query`**
```json
{
  "query": "Should John Doe, who has a 3.7 GPA and is the first in his family to attend college, receive the Merit Scholarship which requires at least a 3.5 GPA and leadership experience?"
}
```

**Expected Response:**
```json
{
  "student": "John Doe",
  "decision": "Approved",
  "explanation": "After careful review of your application for the Merit Scholarship, we are pleased to inform you that your application has been approved. Your GPA of 3.7 exceeds our minimum requirement of 3.5, and your status as a first-generation college student aligns with our goal of supporting diverse educational journeys. The committee was particularly impressed by your academic achievements and your pioneering role in your family's educational history. Congratulations on your achievement, and we look forward to supporting your academic journey.",
  "sources": [
    {
      "text": "Merit Scholarship requires GPA above 3.5, leadership experience and provides $10,000 in funding.",
      "score": 0.92,
      "metadata": {
        "Source": "Scholarship Database 2025",
        "filename": "Scholarship_Guide",
        "last_updated": "2025-03-08"
      }
    }
  ],
  "error": null,
  "source": "decision_making",
  "routed_to": "decision"
}
```

## 2. Direct Question Answering

### Example 1: Application process query

**Request to `POST /question`**
```json
{
  "query": "What steps are involved in the scholarship application process?"
}
```

**Expected Response:**
```json
{
  "query": "What steps are involved in the scholarship application process?",
  "response": "The scholarship application process involves several key steps: First, you need to submit an online application through our portal. You'll then need to provide your academic transcripts which should include your GPA and coursework history. Next, you'll submit financial information to determine need-based eligibility. An important part of the process is writing a personal statement that highlights your achievements, goals, and why you deserve the scholarship. Finally, selected candidates will interview with the selection committee. Make sure to complete all steps before the application deadline.",
  "sources": [
    {
      "text": "Step 1: Submit online application",
      "score": 0.93,
      "metadata": {
        "Source": "Application Process Guide",
        "filename": "How_To_Apply",
        "last_updated": "2025-03-08"
      }
    },
    {
      "text": "Step 2: Provide academic transcripts",
      "score": 0.91,
      "metadata": {
        "Source": "Application Process Guide",
        "filename": "How_To_Apply",
        "last_updated": "2025-03-08"
      }
    }
  ],
  "error": null,
  "source": "question_answering"
}
```

### Example 2: Scholarship history query

**Request to `POST /question`**
```json
{
  "query": "Can you tell me about the history and statistics of the scholarship program?"
}
```

**Expected Response:**
```json
{
  "query": "Can you tell me about the history and statistics of the scholarship program?",
  "response": "Our scholarship program has grown significantly over recent years. In 2022, we supported 150 students with a total of $1,500,000 in scholarship aid. This increased in 2023 to 175 recipients who received a total of $1,800,000 in financial support, representing a 20% increase in funding. The program has a strong track record of supporting students who go on to achieve academic and professional success, including graduates who have entered tech careers and others who have continued to graduate studies.",
  "sources": [
    {
      "text": "In 2022, the scholarship program supported 150 students with a total of $1,500,000.",
      "score": 0.89,
      "metadata": {
        "Source": "Annual Scholarship Report",
        "filename": "Historical_Data",
        "last_updated": "2025-03-08"
      }
    },
    {
      "text": "In 2023, the scholarship program supported 175 students with a total of $1,800,000.",
      "score": 0.86,
      "metadata": {
        "Source": "Annual Scholarship Report",
        "filename": "Historical_Data",
        "last_updated": "2025-03-08"
      }
    }
  ],
  "error": null,
  "source": "question_answering"
}
```

## 3. Direct Decision Making

### Example 1: Strong candidate evaluation

**Request to `POST /decision`**
```json
{
  "student_data": {
    "name": "Sarah Johnson",
    "gpa": 3.9,
    "background": "Low-income household",
    "activities": ["Student government president", "Volunteer at local hospital", "Math club tutor"],
    "essay": "My journey to college has been challenging as a low-income student, but I've maintained a high GPA while leading student government and volunteering 10 hours weekly."
  },
  "scholarship_criteria": {
    "name": "Merit Scholarship",
    "minimum_gpa": 3.5,
    "preference": "Leadership experience and community service",
    "amount": "$10,000"
  }
}
```

**Expected Response:**
```json
{
  "student": "Sarah Johnson",
  "decision": "Approved",
  "explanation": "We are delighted to inform you that your application for the Merit Scholarship has been approved. Your exceptional academic performance with a 3.9 GPA significantly exceeds our 3.5 minimum requirement. Your leadership as student government president demonstrates exactly the kind of leadership qualities we seek in recipients. Additionally, your commitment to community service through hospital volunteering and peer tutoring shows a dedication to helping others that aligns perfectly with our scholarship values. Your personal essay powerfully conveyed your determination to overcome financial challenges while maintaining academic excellence and service commitments. Congratulations on this well-deserved achievement - we look forward to supporting your continued success.",
  "sources": [
    {
      "text": "Merit Scholarship requires GPA above 3.5, leadership experience and provides $10,000 in funding.",
      "score": 0.92,
      "metadata": {
        "Source": "Scholarship Database 2025",
        "filename": "Scholarship_Guide",
        "last_updated": "2025-03-08"
      }
    },
    {
      "text": "Must show community involvement",
      "score": 0.88,
      "metadata": {
        "Source": "Scholarship Requirements Guide",
        "filename": "Eligibility_Criteria",
        "last_updated": "2025-03-08"
      }
    }
  ],
  "error": null,
  "source": "decision_making"
}
```

### Example 2: Borderline candidate evaluation

**Request to `POST /decision`**
```json
{
  "student_data": {
    "name": "Michael Chen",
    "gpa": 3.4,
    "background": "International student",
    "activities": ["Chess club", "Campus newspaper"],
    "essay": "As an international student, I've worked hard to adapt to a new educational system while maintaining good grades and participating in campus activities."
  },
  "scholarship_criteria": {
    "name": "Merit Scholarship",
    "minimum_gpa": 3.5,
    "preference": "Leadership experience",
    "amount": "$10,000"
  }
}
```

**Expected Response:**
```json
{
  "student": "Michael Chen",
  "decision": "Rejected",
  "explanation": "Thank you for applying to the Merit Scholarship program. After careful review of your application, we regret to inform you that we are unable to approve your scholarship at this time. The primary reason for this decision is that your current GPA of 3.4 falls slightly below our minimum requirement of 3.5. While we recognize your valuable contributions to the chess club and campus newspaper, our evaluation process places significant weight on meeting the minimum GPA threshold. We appreciate your hard work as an international student adapting to a new educational environment, which is certainly commendable. For future applications, we encourage you to focus on raising your GPA to meet the minimum requirement. You might also consider seeking leadership positions in your extracurricular activities, as our scholarship gives preference to demonstrated leadership experience. Please consider applying for our Need-based Aid program, which has different eligibility criteria and may be more suitable for your situation.",
  "sources": [
    {
      "text": "Merit Scholarship requires GPA above 3.5, leadership experience and provides $10,000 in funding.",
      "score": 0.92,
      "metadata": {
        "Source": "Scholarship Database 2025",
        "filename": "Scholarship_Guide",
        "last_updated": "2025-03-08"
      }
    },
    {
      "text": "Need-based Aid requires Financial need, academic standing and provides $15,000 in funding.",
      "score": 0.87,
      "metadata": {
        "Source": "Scholarship Database 2025",
        "filename": "Scholarship_Guide",
        "last_updated": "2025-03-08"
      }
    }
  ],
  "error": null,
  "source": "decision_making"
}
```

## 4. Mixed Context Query Example

**Request to `POST /query`**
```json
{
  "query": "What scholarships would be available for a student with a 3.6 GPA who is involved in community service?",
  "context": {
    "student_data": {
      "name": "Alex Rivera",
      "gpa": 3.6,
      "background": "Community college transfer",
      "activities": ["Volunteer at food bank", "Environmental activism"]
    }
  }
}
```

**Expected Response:**
```json
{
  "query": "What scholarships would be available for a student with a 3.6 GPA who is involved in community service?",
  "response": "Based on your profile with a 3.6 GPA and involvement in community service through food bank volunteering and environmental activism, you would be eligible for the Merit Scholarship which requires a GPA above 3.5 and values leadership experience. Your community service activities fulfill our requirement for community involvement. As a community college transfer, you might also qualify for Need-based Aid, which considers academic standing and financial need while offering up to $15,000 in support. I recommend applying to both programs and highlighting your volunteer work and environmental activism in your personal statement.",
  "sources": [
    {
      "text": "Merit Scholarship requires GPA above 3.5, leadership experience and provides $10,000 in funding.",
      "score": 0.92,
      "metadata": {
        "Source": "Scholarship Database 2025",
        "filename": "Scholarship_Guide",
        "last_updated": "2025-03-08"
      }
    },
    {
      "text": "Must show community involvement",
      "score": 0.88,
      "metadata": {
        "Source": "Scholarship Requirements Guide",
        "filename": "Eligibility_Criteria",
        "last_updated": "2025-03-08"
      }
    }
  ],
  "error": null,
  "source": "question_answering",
  "routed_to": "question"
}
``` 
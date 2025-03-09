from typing import Dict, Any, Optional, List
import json
from llms.gemini import GeminiLLM

class Router:
    """
    A router that determines whether a query should be handled by the question answering
    or decision making pipeline.
    """
    
    def __init__(self):
        """Initialize the router with a Gemini model"""
        # Use a lightweight Gemini model for routing decisions
        self.model = GeminiLLM(model_name="gemini-1.5-flash")
    
    async def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route the query to either the question answering or decision making pipeline.
        
        Args:
            query: The user's query
            context: Optional context that may include student data or scholarship criteria
            
        Returns:
            A dictionary indicating the routing decision and relevant data
        """
        # Determine if the query is about making a scholarship decision or a general question
        routing_decision = await self._determine_route(query, context)
        
        # Extract and structure the data based on the routing decision
        if routing_decision["route"] == "decision":
            # For decisions, extract student data and scholarship criteria
            student_data = context.get("student_data") if context else {}
            scholarship_criteria = context.get("scholarship_criteria") if context else {}
            
            # If there's no structured data in the context, try to extract it from the query
            if not student_data or not scholarship_criteria:
                extracted_data = await self._extract_decision_data(query)
                student_data = extracted_data.get("student_data", {})
                scholarship_criteria = extracted_data.get("scholarship_criteria", {})
            
            return {
                "route": "decision",
                "query": query,
                "student_data": student_data,
                "scholarship_criteria": scholarship_criteria
            }
        else:
            # For general questions, just pass the query
            return {
                "route": "question",
                "query": query
            }
    
    async def _determine_route(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Determine if the query is about making a scholarship decision or a general question.
        
        Args:
            query: The user's query
            context: Optional context that may include student data or scholarship criteria
            
        Returns:
            A dictionary with the routing decision
        """
        # Create a prompt for the model
        context_str = json.dumps(context) if context else "No additional context provided."
        
        prompt = f"""
        Determine if the following query is asking for a decision on a scholarship application or
        is a general question about scholarships, applications, or any other topic.
        
        USER QUERY: {query}
        
        CONTEXT: {context_str}
        
        INSTRUCTIONS:
        - If the query is asking for a decision on whether a specific student should receive a scholarship,
          respond with "DECISION"
        - If the query is asking general questions about scholarships, eligibility, or any other topic 
          without requesting a specific decision for a student, respond with "QUESTION"
        - Your response must be only the word "DECISION" or "QUESTION" with no additional text
        """
        
        # Get the routing decision from the model
        system_prompt = "You are a routing system that categorizes queries as either decision requests or general questions."
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        
        # Parse the response
        route = "decision" if "DECISION" in response.upper() else "question"
        
        return {"route": route}
    
    async def _extract_decision_data(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract student data and scholarship criteria from the query if they're embedded in it.
        
        Args:
            query: The user's query that may contain student information and scholarship criteria
            
        Returns:
            A dictionary containing extracted student data and scholarship criteria
        """
        prompt = f"""
        Extract student data and scholarship criteria from the following query.
        
        QUERY: {query}
        
        INSTRUCTIONS:
        - Extract any student information (name, age, GPA, background, achievements, etc.)
        - Extract any scholarship criteria mentioned (GPA requirements, background preferences, etc.)
        - Format the response as a JSON object with two main objects: "student_data" and "scholarship_criteria"
        - If certain information is not present, use empty objects or default values
        - Make reasonable inferences where data is implied but not explicitly stated
        
        Example response format:
        {{
            "student_data": {{
                "name": "John Doe",
                "gpa": 3.8,
                "background": "First-generation college student"
            }},
            "scholarship_criteria": {{
                "minimum_gpa": 3.5,
                "preference": "First-generation students"
            }}
        }}
        """
        
        # Get the extraction from the model
        system_prompt = "You are a data extraction system that parses unstructured text to extract structured information."
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        
        # Try to parse the response as JSON
        try:
            data = json.loads(response)
            return {
                "student_data": data.get("student_data", {}),
                "scholarship_criteria": data.get("scholarship_criteria", {})
            }
        except json.JSONDecodeError:
            # If the response isn't valid JSON, return empty data
            return {
                "student_data": {},
                "scholarship_criteria": {}
            } 
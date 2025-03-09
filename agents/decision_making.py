from typing import Dict, List, Any, Optional, Annotated
from typing_extensions import TypedDict
import json
import asyncio
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel

from llms.gemini import GeminiLLM
from rag.rag_api import RAGSimulator
from agents.aggregator import VotingAggregator

# Define the state for the decision-making agent
class DecisionState(TypedDict):
    student_data: Dict[str, Any]
    scholarship_criteria: Dict[str, Any]
    query: str
    rag_results: Optional[List[Dict[str, Any]]]
    aggregated_result: Optional[Dict[str, Any]]
    sources: Optional[List[Dict[str, Any]]]
    analysis: Optional[str]
    analysis_votes: Optional[List[Dict[str, Any]]]
    recommendation: Optional[str]
    decision: Optional[bool]
    decision_votes: Optional[Dict[str, Any]]
    confidence: Optional[float]
    explanation: Optional[str]
    error: Optional[str]

# Set up the models
def setup_models():
    # Use different Gemini models for different tasks
    analyzer_model = GeminiLLM(model_name="gemini-1.5-pro")
    decision_model = GeminiLLM(model_name="gemini-1.5-pro")
    explainer_model = GeminiLLM(model_name="gemini-1.5-flash")
    
    # Set up RAG simulator
    rag_simulator = RAGSimulator()
    
    return analyzer_model, decision_model, explainer_model, rag_simulator

# Define the agents/nodes for the graph
async def retrieve_information(state: DecisionState) -> DecisionState:
    """Retrieve information from RAG APIs"""
    try:
        # Get RAG simulator
        rag_simulator = RAGSimulator()
        
        # Create a query combining student data and scholarship criteria
        student_data = state.get("student_data", {}) or {}
        background = student_data.get("background", "")
        gpa = student_data.get("gpa", "unknown GPA")
        
        query = f"Scholarship evaluation for student with {gpa}, {background} background."
        
        # Call all RAG APIs with the query
        scholarship_criteria = state.get("scholarship_criteria", {}) or {}
        rag_results = await rag_simulator.call_all_rag_apis(
            query, 
            context={"student": student_data, "scholarship": scholarship_criteria}
        )
        
        # Check if all API calls were successful
        if not rag_results:
            return {**state, "error": "No RAG results returned"}
            
        all_successful = all(result.get("success", False) for result in rag_results)
        if not all_successful:
            errors = [result.get("error", "Unknown error") for result in rag_results if result.get("error")]
            if not any(result.get("success", False) for result in rag_results):
                return {**state, "error": f"Error retrieving information: {', '.join(errors)}"}
        
        # Use the voting aggregator to combine results
        aggregator = VotingAggregator()
        aggregated_result = aggregator.aggregate_rag_results(rag_results)
        
        # Update state with results and sources
        return {
            **state, 
            "rag_results": rag_results, 
            "aggregated_result": aggregated_result,
            "sources": aggregated_result.get("sources", []),
            "confidence": aggregated_result.get("confidence", 0.0)
        }
    except Exception as e:
        import traceback
        print(f"Error in retrieve_information: {str(e)}")
        print(traceback.format_exc())
        return {**state, "error": f"Error retrieving information: {str(e)}"}

async def analyze_single_rag(
    rag_data: Dict[str, Any], 
    student_data: Dict[str, Any], 
    scholarship_criteria: Dict[str, Any],
    analyzer_model: GeminiLLM
) -> Dict[str, Any]:
    """Analyze a single RAG source for eligibility"""
    try:
        # Extract the answer and sources
        answer = rag_data.get("answer", "")
        sources = rag_data.get("sources", [])
        
        # Create context strings
        sources_context = ""
        for i, source in enumerate(sources[:3]):  # Limit to top 3 sources
            source_text = source.get("text", "")
            source_meta = source.get("metadata", {})
            source_score = source.get("score", 0)
            sources_context += f"Source {i+1} (Relevance: {source_score:.2f}): {source_text}\n"
            sources_context += f"Metadata: {json.dumps(source_meta)}\n\n"
            
        # Convert student and scholarship data to formatted strings
        student_context = json.dumps(student_data, indent=2)
        scholarship_context = json.dumps(scholarship_criteria, indent=2)
        
        # Create a prompt for the model
        prompt = f"""
        Analyze this student's eligibility for the scholarship based on the requirements and student data:
        
        STUDENT DATA:
        {student_context}
        
        SCHOLARSHIP CRITERIA:
        {scholarship_context}
        
        RAG INFORMATION:
        {answer}
        
        SOURCES:
        {sources_context}
        
        Provide a detailed analysis of how well the student meets each criterion.
        Be objective and thorough in your analysis.
        End with a preliminary YES or NO recommendation.
        """
        
        # Get the analysis from the model
        system_prompt = "You are an objective scholarship analyzer that evaluates student eligibility based on strict criteria."
        analysis = await analyzer_model.generate(prompt, system_prompt=system_prompt)
        
        # Extract preliminary recommendation
        recommendation = "YES" if "YES" in analysis[-100:].upper() else "NO"
        
        return {
            "analysis": analysis,
            "recommendation": recommendation,
            "rag_source": rag_data.get("source", "Unknown RAG"),
            "confidence": max(0.6, min(0.9, rag_data.get("confidence", 0.0) * 1.2))  # Scale confidence to 0.6-0.9 range
        }
    except Exception as e:
        return {
            "analysis": f"Error analyzing eligibility: {str(e)}",
            "recommendation": "NO",  # Default to NO on error
            "rag_source": rag_data.get("source", "Unknown RAG"),
            "confidence": 0.0
        }

async def analyze_single_rag_with_agent(
    prompt: str,
    agent_type: str,
    rag_source: str,
    analyzer_model: GeminiLLM,
    system_prompt: str
) -> Dict[str, Any]:
    """
    Analyze a student's eligibility using a specific agent type and RAG source
    
    Args:
        prompt: The prompt to send to the model
        agent_type: The type of agent (academic, holistic, equity)
        rag_source: The source of the RAG data
        analyzer_model: The LLM model to use
        system_prompt: The system prompt for this agent type
        
    Returns:
        The analysis results
    """
    try:
        # Get the analysis from the model
        analysis = await analyzer_model.generate(prompt, system_prompt=system_prompt)
        
        # Extract preliminary recommendation from the analysis (look for YES/NO in the last 200 characters)
        last_part = analysis[-200:].upper()
        recommendation = "YES" if "YES" in last_part else "NO"
        
        # Calculate a confidence based on position of the YES/NO in the text (closer to end = higher confidence)
        # This is a simple heuristic that could be improved
        if "YES" in last_part:
            confidence = 0.6 + (last_part.find("YES") / 200 * 0.3)  # Scale to 0.6-0.9
        else:
            confidence = 0.6 + (last_part.find("NO") / 200 * 0.3) if "NO" in last_part else 0.6
            
        return {
            "agent_type": agent_type,
            "rag_source": rag_source,
            "analysis": analysis,
            "recommendation": recommendation,
            "decision": recommendation == "YES",
            "confidence": min(0.95, max(0.6, confidence)),  # Clamp between 0.6 and 0.95
            "reasoning": analysis[-200:] if len(analysis) > 200 else analysis
        }
    except Exception as e:
        import traceback
        print(f"Error in analyze_single_rag_with_agent: {str(e)}")
        print(traceback.format_exc())
        
        return {
            "agent_type": agent_type,
            "rag_source": rag_source,
            "analysis": f"Error analyzing eligibility: {str(e)}",
            "recommendation": "NO",  # Default to NO on error
            "decision": False,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}"
        }

async def analyze_student_eligibility(state: DecisionState) -> DecisionState:
    """Analyze student eligibility using 3 agents × 3 RAG sources = 9 votes"""
    try:
        # Check if we have RAG results
        if not state.get("rag_results"):
            return {**state, "error": "No information available for analysis."}
        
        # Import the agent-specific functions from main.py
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from main import _get_agent_system_prompt, _get_agent_specific_instructions
        
        # Define agent types
        agent_types = ["academic", "holistic", "equity"]
        
        # Get the analyzer model
        analyzer_model, _, _, _ = setup_models()
        
        # Create a grid of 9 analysis tasks (3 agents × 3 RAG sources)
        analysis_tasks = []
        
        # For each RAG source
        for i, rag_result in enumerate(state.get("rag_results", [])):
            if not rag_result.get("success", False):
                continue
                
            # For each agent perspective
            for agent_type in agent_types:
                # Get the agent-specific system prompt
                system_prompt = _get_agent_system_prompt(agent_type)
                
                # Create a prompt tailored to this agent's perspective
                sources_context = ""
                for j, source in enumerate(rag_result.get("sources", [])[:3]):
                    source_text = source.get("text", "")
                    source_meta = source.get("metadata", {})
                    source_score = source.get("score", 0)
                    sources_context += f"Source {j+1} (Relevance: {source_score:.2f}): {source_text}\n"
                    sources_context += f"Metadata: {json.dumps(source_meta)}\n\n"
                
                # Convert student and scholarship data to formatted strings
                student_context = json.dumps(state.get("student_data", {}), indent=2)
                scholarship_context = json.dumps(state.get("scholarship_criteria", {}), indent=2)
                
                # Get the answer from this RAG result
                answer = rag_result.get("answer", "")
                rag_source = f"RAG Source {i+1}: {rag_result.get('source', 'Unknown')}"
                
                # Create task
                prompt = f"""
                As the {agent_type.upper()} EVALUATOR, analyze this student's eligibility for the scholarship:
                
                STUDENT DATA:
                {student_context}
                
                SCHOLARSHIP CRITERIA:
                {scholarship_context}
                
                RAG INFORMATION FROM {rag_source}:
                {answer}
                
                SOURCES:
                {sources_context}
                
                INSTRUCTIONS FOR {agent_type.upper()} EVALUATOR:
                {_get_agent_specific_instructions(agent_type)}
                
                Provide a detailed analysis of how well the student meets each criterion from your {agent_type} perspective.
                End with a preliminary YES or NO recommendation.
                """
                
                # Create a task that runs asynchronously
                task = analyze_single_rag_with_agent(
                    prompt=prompt,
                    agent_type=agent_type,
                    rag_source=rag_source,
                    analyzer_model=analyzer_model,
                    system_prompt=system_prompt
                )
                
                analysis_tasks.append(task)
        
        # Run all tasks in parallel
        analysis_results = await asyncio.gather(*analysis_tasks)
        
        # If no successful analyses, return error
        if not analysis_results:
            return {**state, "error": "No successful analyses completed."}
        
        # Combine analyses using the aggregator
        combined_analysis = ""
        for result in analysis_results:
            combined_analysis += f"\n\n--- {result['agent_type'].upper()} EVALUATOR ON {result['rag_source']} ---\n\n"
            combined_analysis += result["analysis"]
        
        # Return the combined analysis and individual votes
        return {
            **state, 
            "analysis": combined_analysis,
            "analysis_votes": analysis_results,
            "total_votes": len(analysis_results)  # Should be 9 when all succeed
        }
    except Exception as e:
        import traceback
        print(f"Error in analyze_student_eligibility: {str(e)}")
        print(traceback.format_exc())
        return {**state, "error": f"Error analyzing eligibility: {str(e)}"}

async def make_decision(state: DecisionState) -> DecisionState:
    """Make a decision based on all votes from different agent types and RAG sources"""
    try:
        # Check if we have analysis votes
        if not state.get("analysis_votes"):
            return {**state, "error": "No analysis votes available for decision making."}
        
        # Process all votes to make the final decision
        decision_votes = []
        
        # Group votes by agent type and RAG source for detailed breakdown
        vote_breakdown = {
            "by_agent": {
                "academic": {"approve": 0, "reject": 0},
                "holistic": {"approve": 0, "reject": 0},
                "equity": {"approve": 0, "reject": 0}
            },
            "by_source": {}
        }
        
        # Process each vote
        for vote in state["analysis_votes"]:
            # Extract decision
            decision = vote.get("decision", False)
            agent_type = vote.get("agent_type", "unknown")
            rag_source = vote.get("rag_source", "unknown")
            
            # Count in agent breakdown
            vote_type = "approve" if decision else "reject"
            if agent_type in vote_breakdown["by_agent"]:
                vote_breakdown["by_agent"][agent_type][vote_type] += 1
            
            # Count in source breakdown
            if rag_source not in vote_breakdown["by_source"]:
                vote_breakdown["by_source"][rag_source] = {"approve": 0, "reject": 0}
            vote_breakdown["by_source"][rag_source][vote_type] += 1
            
            # Add to decision votes
            decision_votes.append({
                "decision": decision,
                "reasoning": vote.get("reasoning", "No reasoning provided"),
                "confidence": vote.get("confidence", 0.6),
                "agent_type": agent_type,
                "rag_source": rag_source
            })
        
        # Count total votes
        total_votes = len(decision_votes)
        approve_votes = sum(1 for vote in decision_votes if vote["decision"])
        reject_votes = total_votes - approve_votes
        
        # Calculate confidence based on vote margin
        vote_margin = abs(approve_votes - reject_votes) / total_votes if total_votes > 0 else 0
        confidence = 0.5 + (vote_margin / 2)  # Scale to 0.5-1.0 range
        
        # Make final decision (majority wins)
        final_decision = approve_votes > reject_votes
        
        # Create a recommendation with detailed voting results
        if final_decision:
            recommendation = f"APPROVED: Student meets criteria with {confidence:.2f} confidence. Approved by {approve_votes} of {total_votes} votes across {len(vote_breakdown['by_agent'])} agent types and {len(vote_breakdown['by_source'])} RAG sources."
        else:
            recommendation = f"REJECTED: Student does not meet criteria with {confidence:.2f} confidence. Rejected by {reject_votes} of {total_votes} votes across {len(vote_breakdown['by_agent'])} agent types and {len(vote_breakdown['by_source'])} RAG sources."
        
        # Update state with decision, confidence, and recommendation
        return {
            **state, 
            "decision": final_decision,
            "decision_votes": {
                "votes": decision_votes,
                "vote_count": {"approve": approve_votes, "reject": reject_votes},
                "vote_breakdown": vote_breakdown,
                "confidence": confidence
            },
            "confidence": confidence,
            "recommendation": recommendation
        }
    except Exception as e:
        import traceback
        print(f"Error in make_decision: {str(e)}")
        print(traceback.format_exc())
        return {**state, "error": f"Error making decision: {str(e)}"}

async def explain_decision(state: DecisionState) -> DecisionState:
    """Provide a detailed explanation for the decision"""
    try:
        # Check if we have a decision
        if "decision" not in state:
            return {**state, "error": "No decision available to explain."}
        
        # Get the explainer model
        _, _, explainer_model, _ = setup_models()
        
        # Create a context for the explanation
        student_context = json.dumps(state["student_data"], indent=2)
        scholarship_context = json.dumps(state["scholarship_criteria"], indent=2)
        decision_votes = json.dumps(state["decision_votes"], indent=2) if state.get("decision_votes") else "{}"
        
        # Extract sources information for evidence
        sources_context = ""
        for i, source in enumerate(state.get("sources", [])[:3]):  # Limit to top 3 sources
            source_text = source.get("text", "")
            source_meta = source.get("metadata", {})
            sources_context += f"Source {i+1}: {source_text}\n"
            sources_context += f"Metadata: {json.dumps(source_meta)}\n\n"
        
        # Create a prompt for the model
        prompt = f"""
        Provide a detailed explanation for the {'APPROVAL' if state['decision'] else 'REJECTION'} of this student's scholarship application.
        
        STUDENT DATA:
        {student_context}
        
        SCHOLARSHIP CRITERIA:
        {scholarship_context}
        
        ANALYSIS SUMMARY:
        {state['analysis'][:500]}
        
        DECISION DETAILS:
        * Final Decision: {"APPROVED" if state['decision'] else "REJECTED"}
        * Confidence: {state.get('confidence', 0.0):.2f} out of 1.0
        * Recommendation: {state['recommendation']}
        * Vote Details: {decision_votes}
        
        SUPPORTING EVIDENCE:
        {sources_context}
        
        Please provide a compassionate yet factual explanation that the student can understand. Include:
        1. The key reasons for the decision
        2. The level of confidence in the decision and how many sources agreed
        3. Areas where the student excelled (if any)
        4. Areas where the student did not meet criteria (if any)
        5. Constructive advice for future applications (if rejection)
        """
        
        # Get the explanation from the model
        system_prompt = "You are a compassionate education counselor explaining scholarship decisions to students."
        explanation = await explainer_model.generate(prompt, system_prompt=system_prompt)
        
        # Update state with explanation
        return {**state, "explanation": explanation}
    except Exception as e:
        return {**state, "error": f"Error explaining decision: {str(e)}"}

# Create the decision-making graph
def create_decision_graph():
    """Create a LangGraph for scholarship decision making"""
    # Initialize the graph
    workflow = StateGraph(DecisionState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_information)
    workflow.add_node("analyze", analyze_student_eligibility)
    workflow.add_node("decide", make_decision)
    workflow.add_node("explain", explain_decision)
    
    # Add edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "decide")
    workflow.add_edge("decide", "explain")
    workflow.add_edge("explain", END)
    
    # Compile the graph
    return workflow.compile()

# Function to run the graph
async def process_decision(student_data: Dict[str, Any], scholarship_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Process a scholarship decision through the decision-making graph"""
    try:
        # Create initial state
        initial_state = DecisionState(
            student_data=student_data,
            scholarship_criteria=scholarship_criteria,
            query=f"Scholarship evaluation for {student_data.get('name', 'student')}",
            rag_results=None,
            aggregated_result=None,
            sources=None,
            analysis=None,
            analysis_votes=None,
            recommendation=None,
            decision=None,
            decision_votes=None,
            confidence=None,
            explanation=None,
            error=None
        )
        
        # Create graph
        graph = create_decision_graph()
        
        # Run the graph
        result = await graph.ainvoke(initial_state)
        
        # Check for errors in the result
        if result.get("error"):
            # If there's an error in the state, include it in the response
            return {
                "student": student_data.get("name", "Unknown Student"),
                "decision": "Rejected",  # Default to reject on error
                "explanation": f"Error in decision process: {result['error']}",
                "sources": [],
                "error": result["error"],
                "source": "decision_making"
            }
        
        # Extract decision votes safely
        decision_votes = result.get("decision_votes") or {}
        vote_count = decision_votes.get("vote_count", {}) if isinstance(decision_votes, dict) else {}
        vote_breakdown = decision_votes.get("vote_breakdown", {}) if isinstance(decision_votes, dict) else {}
        
        # Return the final response with safely accessed fields and detailed vote information
        return {
            "student": student_data.get("name", "Unknown Student"),
            "decision": "Approved" if result.get("decision", False) else "Rejected",
            "explanation": result.get("explanation") or "No explanation provided.",
            "confidence": result.get("confidence") or 0.0,
            "vote_count": vote_count,
            "vote_breakdown": vote_breakdown,
            "total_votes": sum(vote_count.values()) if vote_count else 0,
            "sources": result.get("sources") or [],
            "error": result.get("error"),
            "source": "decision_making"
        }
    except Exception as e:
        # Log the error
        import traceback
        print(f"Error in process_decision: {str(e)}")
        print(traceback.format_exc())
        
        # Return an error response
        return {
            "student": student_data.get("name", "Unknown Student"),
            "decision": "Rejected",  # Default to reject on error
            "explanation": f"Error processing decision: {str(e)}",
            "error": str(e),
            "source": "decision_making"
        } 
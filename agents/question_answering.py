from typing import Dict, List, Any, Optional, Annotated
from typing_extensions import TypedDict
import json
import asyncio
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel

from llms.gemini import GeminiLLM
from rag.rag_api import RAGSimulator
from agents.aggregator import VotingAggregator

# Define the state for the Q/A agent
class QuestionState(TypedDict):
    query: str
    rag_results: Optional[List[Dict[str, Any]]]
    aggregated_result: Optional[Dict[str, Any]]
    response: Optional[str]
    sources: Optional[List[Dict[str, Any]]]
    confidence: Optional[float]
    error: Optional[str]

# Set up the models
def setup_models():
    # Use different Gemini models for different tasks
    router_model = GeminiLLM(model_name="gemini-1.5-flash")
    synthesizer_model = GeminiLLM(model_name="gemini-1.5-pro")
    
    # Set up RAG simulator
    rag_simulator = RAGSimulator()
    
    return router_model, synthesizer_model, rag_simulator

# Define the agents/nodes for the graph
async def retrieve_information(state: QuestionState) -> QuestionState:
    """Retrieve information from RAG APIs"""
    try:
        # Get RAG simulator
        rag_simulator = RAGSimulator()
        
        # Call all RAG APIs with the query
        rag_results = await rag_simulator.call_all_rag_apis(state["query"])
        
        # Check if all API calls were successful
        all_successful = all(result.get("success", False) for result in rag_results)
        if not all_successful:
            errors = [result.get("error", "Unknown error") for result in rag_results if result.get("error")]
            if not any(result.get("success", False) for result in rag_results):
                return {**state, "error": f"Error retrieving information: {', '.join(errors)}"}
        
        # Use the voting aggregator to combine results
        aggregator = VotingAggregator()
        aggregated_result = aggregator.aggregate_rag_results(rag_results)
        
        # Update state with results, sources, and confidence
        return {
            **state, 
            "rag_results": rag_results,
            "aggregated_result": aggregated_result,
            "sources": aggregated_result.get("sources", []),
            "confidence": aggregated_result.get("confidence", 0.0)
        }
    except Exception as e:
        return {**state, "error": f"Error retrieving information: {str(e)}"}

async def synthesize_response(state: QuestionState) -> QuestionState:
    """Synthesize a response from RAG results using Gemini model"""
    try:
        # Check if we have aggregated results
        if not state.get("aggregated_result") or not state.get("aggregated_result", {}).get("answer"):
            return {**state, "response": "No information available for this query."}
        
        # Get the synthesizer model
        _, synthesizer_model, _ = setup_models()
        
        # Get the aggregated answer
        aggregated_answer = state["aggregated_result"].get("answer", "")
        
        # Extract sources information
        sources_context = ""
        for i, source in enumerate(state.get("sources", [])[:5]):  # Limit to top 5 sources for context
            source_text = source.get("text", "")
            source_meta = source.get("metadata", {})
            source_score = source.get("score", 0)
            sources_context += f"Source {i+1} (Relevance: {source_score:.2f}): {source_text}\n"
            sources_context += f"Metadata: {json.dumps(source_meta)}\n\n"
        
        # Create a prompt for the model
        prompt = f"""
        Based on the following information, please provide a comprehensive
        and helpful answer to the user's question:
        
        USER QUERY: {state["query"]}
        
        AGGREGATED ANSWER FROM KNOWLEDGE BASES:
        {aggregated_answer}
        
        RELEVANT SOURCES:
        {sources_context}
        
        CONFIDENCE: {state.get("confidence", 0.0):.2f} out of 1.0
        
        Please synthesize this information into a clear, concise, and helpful response.
        Make sure to integrate information from all knowledge bases if relevant to the query.
        When citing information, reference the sources appropriately.
        If the confidence is low, please indicate that the answer might be uncertain.
        """
        
        # Get the response from the model
        system_prompt = "You are a helpful assistant that provides accurate information based on the knowledge bases provided."
        response = await synthesizer_model.generate(prompt, system_prompt=system_prompt)
        
        # Update state with response
        return {**state, "response": response}
    except Exception as e:
        return {**state, "error": f"Error synthesizing response: {str(e)}"}

# Create the Q/A graph
def create_qa_graph():
    """Create a LangGraph for question answering"""
    # Initialize the graph
    workflow = StateGraph(QuestionState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_information)
    workflow.add_node("synthesize", synthesize_response)
    
    # Add edges - make sure to connect from START to the first node
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "synthesize")
    workflow.add_edge("synthesize", END)
    
    # Compile the graph
    return workflow.compile()


# Function to run the graph
async def process_question(query: str) -> Dict[str, Any]:
    """Process a question through the Q/A graph"""
    # Create initial state
    initial_state = QuestionState(
        query=query,
        rag_results=None,
        aggregated_result=None,
        response=None,
        sources=None,
        confidence=None,
        error=None
    )
    
    # Create graph
    graph = create_qa_graph()
    
    # Run the graph
    result = await graph.ainvoke(initial_state)
    
    # Return the final response
    return {
        "query": query,
        "response": result["response"] if not result.get("error") else f"Error: {result['error']}",
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", 0.0),
        "error": result.get("error"),
        "source": "question_answering"
    } 
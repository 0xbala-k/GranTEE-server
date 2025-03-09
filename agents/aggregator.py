from typing import Dict, List, Any, Optional
import json
from collections import Counter

class VotingAggregator:
    """
    Aggregator that implements a voting system across multiple RAG sources.
    Uses weighted voting based on source relevance and confidence.
    """
    
    @staticmethod
    def aggregate_rag_results(rag_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple RAG sources using a voting mechanism.
        
        Args:
            rag_results: List of RAG API results
            
        Returns:
            Aggregated result with combined answer, sources, and confidence scores
        """
        if not rag_results:
            return {
                "answer": "No information available.",
                "sources": [],
                "error": "No RAG results to aggregate",
                "success": False,
                "confidence": 0.0
            }
        
        # Extract answers and sources from all results
        all_answers = []
        all_sources = []
        errors = []
        
        for result in rag_results:
            if result.get("success", False) and result.get("answer"):
                all_answers.append(result.get("answer", ""))
                
            if result.get("sources"):
                all_sources.extend(result.get("sources", []))
                
            if result.get("error"):
                errors.append(result.get("error"))
        
        # Sort sources by score (highest first)
        all_sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # If we have errors but no answers, return error
        if errors and not all_answers:
            return {
                "answer": "Error retrieving information.",
                "sources": all_sources[:5],  # Include top 5 sources if available
                "error": "; ".join(errors),
                "success": False,
                "confidence": 0.0
            }
        
        # Prepare aggregated result
        aggregated_result = {
            "answer": "\n\n".join(all_answers),
            "sources": all_sources[:10],  # Limit to top 10 sources
            "error": None,
            "success": True,
            "confidence": 0.0
        }
        
        # Calculate confidence based on number of successful RAGs and source scores
        if all_sources:
            # Average of top 5 source scores
            top_scores = [source.get("score", 0) for source in all_sources[:5]]
            avg_score = sum(top_scores) / len(top_scores) if top_scores else 0
            
            # Weight by number of successful RAGs
            successful_rags = sum(1 for r in rag_results if r.get("success", False))
            rag_weight = successful_rags / len(rag_results)
            
            aggregated_result["confidence"] = avg_score * rag_weight
        
        return aggregated_result
    
    @staticmethod
    def vote_on_decision(decision_votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Vote on a decision using multiple analysis results.
        
        Args:
            decision_votes: List of decision results, each with a boolean decision
            
        Returns:
            Final decision based on weighted voting
        """
        if not decision_votes:
            return {
                "decision": False,
                "confidence": 0.0,
                "vote_count": {"approve": 0, "reject": 0},
                "reasoning": "No decision data available"
            }
        
        # Count votes
        approve_votes = sum(1 for vote in decision_votes if vote.get("decision", False))
        reject_votes = len(decision_votes) - approve_votes
        
        # Calculate confidence based on vote margin
        vote_margin = abs(approve_votes - reject_votes) / len(decision_votes)
        confidence = 0.5 + (vote_margin / 2)  # Scale to 0.5-1.0 range
        
        # Make final decision
        final_decision = approve_votes > reject_votes
        
        # Extract reasoning from majority votes
        majority_type = "approve" if final_decision else "reject"
        majority_votes = [vote for vote in decision_votes if vote.get("decision", False) == final_decision]
        
        # Combine reasoning from up to 3 majority votes
        reasoning_samples = [vote.get("reasoning", "") for vote in majority_votes[:3] if vote.get("reasoning")]
        combined_reasoning = "\n\n".join(reasoning_samples) if reasoning_samples else "Decision based on majority vote"
        
        return {
            "decision": final_decision,
            "confidence": confidence,
            "vote_count": {"approve": approve_votes, "reject": reject_votes},
            "reasoning": combined_reasoning
        }
        
    @staticmethod
    def combine_analyses(analyses: List[str]) -> str:
        """
        Combine multiple analysis texts into a single comprehensive analysis.
        
        Args:
            analyses: List of analysis texts from different agents/RAGs
            
        Returns:
            Combined analysis text
        """
        if not analyses:
            return "No analysis available."
        
        # Simple combination for now - in a real system, you might use an LLM to synthesize
        return "\n\n--- ANALYSIS SOURCES ---\n\n".join(analyses) 
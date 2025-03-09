from typing import Dict, List, Any, Optional
import json
import asyncio
import aiohttp

class RAGSimulator:
    """
    Client for three different RAG APIs.
    """
    
    def __init__(self):
        # RAG API endpoints
        self.rag_endpoints = {
            "deep_search": {
                "url": "https://cl-rag.onrender.com/api/routes/chat/deep-search/query",
                "top_k": 15,
                "name": "Deep Search RAG"
            },
            "community_search": {
                "url": "https://cl-rag-community.onrender.com/api/routes/chat/community-search/query",
                "top_k": 10,
                "name": "Community Search RAG"
            },
            "fast_search": {
                "url": "https://cl-rag-fast.onrender.com/api/routes/chat/fast-search/query",
                "top_k": 5,
                "name": "Fast Search RAG"
            }
        }
        
    async def _call_rag_api(self, endpoint_key: str, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call a single RAG API endpoint
        
        Args:
            endpoint_key: Key for the endpoint configuration
            query: The query to send
            context: Optional context information
            
        Returns:
            API response or error information
        """
        endpoint_config = self.rag_endpoints.get(endpoint_key)
        if not endpoint_config:
            return {
                "answer": f"Error: Unknown endpoint {endpoint_key}",
                "sources": [],
                "error": f"Unknown endpoint {endpoint_key}",
                "success": False
            }
        
        url = endpoint_config["url"]
        top_k = endpoint_config["top_k"]
        
        # Prepare the request payload
        payload = {
            "query": query,
            "top_k": top_k,
            "include_sources": True,
            "include_metadata": True,
            "use_fallbacks": True
        }
        
        # If we have context, we might want to add it to the query
        if context:
            # Optional: Enhance the query with context
            context_str = json.dumps(context)
            enhanced_query = f"{query} (Context: {context_str})"
            payload["query"] = enhanced_query[:1000]  # Limit length to be safe
        
        try:
            # Make an async HTTP request to the RAG API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30  # 30 second timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "answer": f"Error from {endpoint_config['name']}: Status {response.status}",
                            "sources": [],
                            "error": f"API error: {error_text}",
                            "success": False
                        }
                    
                    # Parse the response
                    result = await response.json()
                    
                    # Format the response to match our expected structure
                    formatted_response = {
                        "answer": result.get("answer", "No answer provided"),
                        "sources": result.get("sources", []),
                        "error": None,
                        "success": True,
                        "source": endpoint_config["name"]
                    }
                    
                    return formatted_response
                    
        except Exception as e:
            return {
                "answer": f"Error calling {endpoint_config['name']}: {str(e)}",
                "sources": [],
                "error": str(e),
                "success": False,
                "source": endpoint_config["name"]
            }
    
    async def rag_api1(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call the Deep Search RAG API"""
        return await self._call_rag_api("deep_search", query, context)
    
    async def rag_api2(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call the Community Search RAG API"""
        return await self._call_rag_api("community_search", query, context)
    
    async def rag_api3(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call the Fast Search RAG API"""
        return await self._call_rag_api("fast_search", query, context)

    async def call_all_rag_apis(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Call all three RAG APIs concurrently and return combined results"""
        tasks = [
            self.rag_api1(query, context),
            self.rag_api2(query, context),
            self.rag_api3(query, context)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in the results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                api_name = ["Deep Search", "Community Search", "Fast Search"][i]
                processed_results.append({
                    "answer": f"Error calling {api_name} RAG: {str(result)}",
                    "sources": [],
                    "error": str(result),
                    "success": False,
                    "source": api_name
                })
            else:
                processed_results.append(result)
        
        return processed_results 
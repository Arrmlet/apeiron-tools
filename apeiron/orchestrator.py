"""
Tool Orchestrator for Apeiron Tools

Coordinates execution of distributed MCP tools across multiple servers.
Implements circuit breaker patterns, parallel execution, and intelligent routing.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
from contextlib import asynccontextmanager

from .protocol import ToolRequest, ToolResult
from .registry import MCPServerRegistry


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for MCP server failures"""
    server_id: str
    failure_threshold: int = 5
    timeout: float = 60.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed through circuit breaker"""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if current_time - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
            
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class QueryAnalyzer:
    """Analyzes user queries to determine required tools and complexity"""
    
    def __init__(self):
        self.tool_keywords = {
            "filesystem": ["file", "directory", "folder", "read", "write", "create", "delete", "list"],
            "web": ["url", "website", "fetch", "download", "http", "api", "scrape"],
            "memory": ["remember", "store", "save", "recall", "search", "knowledge"],
            "git": ["github", "repository", "repo", "commit", "pull", "push", "branch"],
            "database": ["database", "sql", "query", "table", "insert", "select", "update"],
            "compute": ["calculate", "compute", "analyze", "process", "transform"]
        }
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine required tools and complexity"""
        query_lower = query.lower()
        
        # Determine relevant categories
        relevant_categories = []
        for category, keywords in self.tool_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_categories.append(category)
                
        # Estimate complexity based on query characteristics
        complexity_indicators = [
            len(query.split()) > 20,  # Long query
            len(relevant_categories) > 2,  # Multi-domain
            any(word in query_lower for word in ["and", "then", "also", "combine", "merge"]),  # Sequential operations
            any(word in query_lower for word in ["all", "every", "each", "multiple"]),  # Bulk operations
        ]
        
        complexity_score = sum(complexity_indicators)
        if complexity_score >= 3:
            complexity = "complex"
        elif complexity_score >= 1:
            complexity = "medium"
        else:
            complexity = "simple"
            
        return {
            "relevant_categories": relevant_categories,
            "complexity": complexity,
            "estimated_tools": min(len(relevant_categories) * 3, 50),
            "requires_sequential": "then" in query_lower or "after" in query_lower,
            "requires_aggregation": any(word in query_lower for word in ["combine", "merge", "aggregate", "summarize"])
        }


class ToolOrchestrator:
    """
    Orchestrates distributed tool execution across multiple MCP servers.
    
    Handles parallel execution, circuit breaking, retry logic, and result aggregation
    for unlimited tool orchestration beyond context limits.
    """
    
    def __init__(self, registry: MCPServerRegistry):
        self.registry = registry
        self.session: Optional[aiohttp.ClientSession] = None
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.query_analyzer = QueryAnalyzer()
        self.max_concurrent_tools = 20
        self.default_timeout = 30.0
        self.retry_attempts = 3
        self.retry_backoff = 1.0
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize the orchestrator"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.default_timeout)
        )
        
        # Initialize circuit breakers for all servers
        for server_id in self.registry.servers.keys():
            self.circuit_breakers[server_id] = CircuitBreaker(server_id)
            
        self.logger.info("Tool Orchestrator initialized")
        
    async def shutdown(self):
        """Shutdown the orchestrator"""
        if self.session:
            await self.session.close()
        self.logger.info("Tool Orchestrator shutdown complete")
        
    async def execute_query(self, user_query: str, max_tools: int = 50, timeout: float = 30.0) -> Dict[str, Any]:
        """Execute a user query using distributed tool orchestration"""
        start_time = time.time()
        
        try:
            # Analyze the query
            analysis = self.query_analyzer.analyze_query(user_query)
            self.logger.info(f"Query analysis: {analysis}")
            
            # Determine required tools
            tool_requests = await self._determine_tools(user_query, analysis, max_tools)
            
            # Execute tools
            results = await self._execute_tools(tool_requests, timeout)
            
            # Aggregate results
            final_response = await self._aggregate_results(user_query, results)
            
            execution_time = time.time() - start_time
            
            return {
                "user_query": user_query,
                "analysis": analysis,
                "tool_results": results,
                "final_response": final_response,
                "execution_time": execution_time,
                "tools_used": len(results),
                "success_rate": len([r for r in results if r.success]) / len(results) if results else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return {
                "user_query": user_query,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "tools_used": 0,
                "success_rate": 0.0
            }
            
    async def _determine_tools(self, query: str, analysis: Dict[str, Any], max_tools: int) -> List[ToolRequest]:
        """Determine which tools to execute based on query analysis"""
        tool_requests = []
        
        # Get relevant tools for each category
        for category in analysis["relevant_categories"]:
            servers = await self.registry.get_servers_by_category(category)
            healthy_servers = await self.registry.get_healthy_servers()
            available_servers = [s for s in servers if s in healthy_servers]
            
            for server_id in available_servers[:2]:  # Limit servers per category
                server_config = self.registry.servers[server_id]
                for tool_name in server_config.tools:
                    if len(tool_requests) >= max_tools:
                        break
                        
                    tool_requests.append(ToolRequest(
                        tool_name=tool_name,
                        server_category=category,
                        arguments={"query": query},
                        priority=1 if category in analysis["relevant_categories"][:2] else 2
                    ))
                    
        # Sort by priority
        tool_requests.sort(key=lambda x: x.priority)
        
        return tool_requests[:max_tools]
        
    async def _execute_tools(self, tool_requests: List[ToolRequest], timeout: float) -> List[ToolResult]:
        """Execute tools in parallel with circuit breaking and retry logic"""
        semaphore = asyncio.Semaphore(self.max_concurrent_tools)
        
        async def execute_single_tool(request: ToolRequest) -> ToolResult:
            async with semaphore:
                return await self._execute_tool_with_retry(request, timeout)
                
        # Execute all tools concurrently
        tasks = [execute_single_tool(request) for request in tool_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        tool_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool_results.append(ToolResult(
                    tool_name=tool_requests[i].tool_name,
                    success=False,
                    error=str(result),
                    execution_time=0.0,
                    server_id="unknown"
                ))
            else:
                tool_results.append(result)
                
        return tool_results
        
    async def _execute_tool_with_retry(self, request: ToolRequest, timeout: float) -> ToolResult:
        """Execute a single tool with retry logic and circuit breaking"""
        best_server = await self.registry.get_best_server_for_tool(request.tool_name)
        
        if not best_server:
            return ToolResult(
                tool_name=request.tool_name,
                success=False,
                error="No available servers for tool",
                execution_time=0.0,
                server_id="none"
            )
            
        circuit_breaker = self.circuit_breakers.get(best_server)
        if circuit_breaker and not circuit_breaker.should_allow_request():
            return ToolResult(
                tool_name=request.tool_name,
                success=False,
                error="Circuit breaker open",
                execution_time=0.0,
                server_id=best_server
            )
            
        # Retry loop
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                result = await self._call_tool(best_server, request, timeout)
                
                # Record success in circuit breaker
                if circuit_breaker:
                    circuit_breaker.record_success()
                    
                return result
                
            except Exception as e:
                last_error = e
                
                # Record failure in circuit breaker
                if circuit_breaker:
                    circuit_breaker.record_failure()
                    
                # Exponential backoff
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_backoff * (2 ** attempt))
                    
        return ToolResult(
            tool_name=request.tool_name,
            success=False,
            error=str(last_error),
            execution_time=0.0,
            server_id=best_server
        )
        
    async def _call_tool(self, server_id: str, request: ToolRequest, timeout: float) -> ToolResult:
        """Make actual HTTP call to MCP server tool"""
        if not self.session:
            raise RuntimeError("Orchestrator not initialized")
            
        server_config = self.registry.servers[server_id]
        start_time = time.time()
        
        url = f"http://{server_config.host}:{server_config.port}/tools/{request.tool_name}"
        payload = {
            "arguments": request.arguments
        }
        
        try:
            async with self.session.post(url, json=payload, timeout=timeout) as response:
                execution_time = time.time() - start_time
                
                if response.status == 200:
                    result_data = await response.json()
                    return ToolResult(
                        tool_name=request.tool_name,
                        success=True,
                        result=result_data,
                        execution_time=execution_time,
                        server_id=server_id
                    )
                else:
                    error_text = await response.text()
                    return ToolResult(
                        tool_name=request.tool_name,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}",
                        execution_time=execution_time,
                        server_id=server_id
                    )
                    
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=request.tool_name,
                success=False,
                error="Request timeout",
                execution_time=time.time() - start_time,
                server_id=server_id
            )
            
    async def _aggregate_results(self, query: str, results: List[ToolResult]) -> str:
        """Aggregate tool results into a coherent response"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return "No tools executed successfully. Please try a simpler query."
            
        # Group results by category
        categories = {}
        for result in successful_results:
            server_id = result.server_id
            if server_id in self.registry.servers:
                category = self.registry.servers[server_id].category
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)
                
        # Build response
        response_parts = [f"Query: {query}\n"]
        
        for category, category_results in categories.items():
            response_parts.append(f"\n{category.title()} Results:")
            for result in category_results:
                response_parts.append(f"- {result.tool_name}: {str(result.result)[:200]}")
                
        response_parts.append(f"\nExecuted {len(successful_results)} tools successfully.")
        
        return "\n".join(response_parts)
        
    async def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics"""
        circuit_stats = {}
        for server_id, breaker in self.circuit_breakers.items():
            circuit_stats[server_id] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time
            }
            
        return {
            "max_concurrent_tools": self.max_concurrent_tools,
            "default_timeout": self.default_timeout,
            "retry_attempts": self.retry_attempts,
            "circuit_breakers": circuit_stats,
            "active_servers": len(await self.registry.get_healthy_servers())
        }
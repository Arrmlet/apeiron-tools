"""
MCPSynapse Protocol Definition for Apeiron Tools

Custom Bittensor synapse for unlimited MCP tool orchestration.
Enables distributed tool execution across multiple MCP servers.
"""

import time
from typing import List, Dict, Any, Optional
import bittensor as bt
from pydantic import BaseModel, Field


class ToolRequest(BaseModel):
    """Individual tool execution request"""
    tool_name: str = Field(..., description="Name of the tool to execute")
    server_category: str = Field(..., description="MCP server category (filesystem, web, etc.)")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    priority: int = Field(default=1, description="Execution priority (1-10)")


class ToolResult(BaseModel):
    """Result from tool execution"""
    tool_name: str = Field(..., description="Name of executed tool")
    success: bool = Field(..., description="Whether execution succeeded")
    result: Any = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: float = Field(..., description="Time taken to execute (seconds)")
    server_id: str = Field(..., description="ID of MCP server that executed the tool")


class MCPSynapse(bt.Synapse):
    """
    Custom Bittensor synapse for MCP tool orchestration.
    
    This synapse enables unlimited tool orchestration by coordinating
    execution across multiple distributed MCP servers.
    """
    
    # Input fields
    user_query: str = Field(..., description="User's natural language query")
    required_tools: List[ToolRequest] = Field(default_factory=list, description="Specific tools to execute")
    max_tools: int = Field(default=50, description="Maximum number of tools to use")
    timeout: float = Field(default=30.0, description="Maximum execution time in seconds")
    query_complexity: str = Field(default="medium", description="Query complexity level (simple, medium, complex)")
    
    # Output fields  
    tool_results: List[ToolResult] = Field(default_factory=list, description="Results from tool executions")
    final_response: str = Field(default="", description="Aggregated response to user query")
    execution_summary: Dict[str, Any] = Field(default_factory=dict, description="Execution statistics")
    total_execution_time: float = Field(default=0.0, description="Total time for all tool executions")
    tools_used: int = Field(default=0, description="Number of tools actually used")
    success_rate: float = Field(default=0.0, description="Percentage of tools that executed successfully")
    
    # Metadata
    miner_hotkey: str = Field(default="", description="Hotkey of responding miner")
    timestamp: float = Field(default_factory=time.time, description="Request timestamp")
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "forbid"
        
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        successful_tools = [r for r in self.tool_results if r.success]
        failed_tools = [r for r in self.tool_results if not r.success]
        
        return {
            "total_tools": len(self.tool_results),
            "successful_tools": len(successful_tools),
            "failed_tools": len(failed_tools),
            "success_rate": len(successful_tools) / len(self.tool_results) if self.tool_results else 0.0,
            "avg_execution_time": sum(r.execution_time for r in self.tool_results) / len(self.tool_results) if self.tool_results else 0.0,
            "total_execution_time": self.total_execution_time,
            "servers_used": len(set(r.server_id for r in self.tool_results)),
            "query_complexity": self.query_complexity,
            "timestamp": self.timestamp
        }
        
    def add_tool_result(self, result: ToolResult):
        """Add a tool execution result"""
        self.tool_results.append(result)
        self.tools_used = len(self.tool_results)
        self.total_execution_time += result.execution_time
        
        # Update success rate
        successful = sum(1 for r in self.tool_results if r.success)
        self.success_rate = successful / len(self.tool_results)
        
    def is_complete(self) -> bool:
        """Check if synapse execution is complete"""
        return (
            len(self.tool_results) >= len(self.required_tools) or
            self.tools_used >= self.max_tools or
            self.total_execution_time >= self.timeout
        )
        
    def get_failed_tools(self) -> List[ToolResult]:
        """Get list of failed tool executions"""
        return [r for r in self.tool_results if not r.success]
        
    def get_successful_tools(self) -> List[ToolResult]:
        """Get list of successful tool executions"""
        return [r for r in self.tool_results if r.success]
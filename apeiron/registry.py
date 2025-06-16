"""
MCP Server Registry for Apeiron Tools

Manages available MCP servers, their capabilities, health status,
and lifecycle management for distributed tool orchestration.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from pydantic import BaseModel, Field


class ServerStatus(Enum):
    """MCP Server status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    server_id: str
    name: str
    category: str
    host: str
    port: int
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)
    health_check_path: str = "/health"
    max_concurrent_requests: int = 10
    timeout: float = 30.0
    retry_attempts: int = 3


@dataclass
class ServerHealth:
    """Health status of an MCP server"""
    status: ServerStatus
    last_check: float
    response_time: float
    error_count: int
    total_requests: int
    success_rate: float
    uptime: float
    version: Optional[str] = None


class MCPServerRegistry:
    """
    Registry for managing MCP servers and their capabilities.
    
    Handles server discovery, health monitoring, lifecycle management,
    and tool routing for distributed orchestration.
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.health_status: Dict[str, ServerHealth] = {}
        self.server_tools: Dict[str, Set[str]] = {}
        self.tool_to_servers: Dict[str, List[str]] = {}
        self.category_servers: Dict[str, List[str]] = {}
        self.running_processes: Dict[str, asyncio.subprocess.Process] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.health_check_interval = 30.0
        self.health_check_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize the registry and start health monitoring"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10.0)
        )
        await self._load_default_servers()
        await self.start_health_monitoring()
        self.logger.info("MCP Server Registry initialized")
        
    async def shutdown(self):
        """Shutdown the registry and cleanup resources"""
        if self.health_check_task:
            self.health_check_task.cancel()
            
        # Stop all running servers
        for server_id in list(self.running_processes.keys()):
            await self.stop_server(server_id)
            
        if self.session:
            await self.session.close()
            
        self.logger.info("MCP Server Registry shutdown complete")
        
    async def _load_default_servers(self):
        """Load default MCP server configurations"""
        default_servers = [
            MCPServerConfig(
                server_id="filesystem",
                name="Filesystem Operations",
                category="filesystem",
                host="localhost",
                port=8001,
                command="npx",
                args=["@modelcontextprotocol/server-filesystem", "/tmp"],
                tools=["read_file", "write_file", "list_directory", "create_directory", "delete_file"]
            ),
            MCPServerConfig(
                server_id="fetch",
                name="Web Content Fetcher", 
                category="web",
                host="localhost",
                port=8002,
                command="npx",
                args=["@modelcontextprotocol/server-fetch"],
                tools=["fetch", "get_webpage", "download_file"]
            ),
            MCPServerConfig(
                server_id="memory",
                name="Memory Storage",
                category="memory",
                host="localhost", 
                port=8003,
                command="npx",
                args=["@modelcontextprotocol/server-memory"],
                tools=["store", "retrieve", "search", "delete", "list_keys"]
            ),
            MCPServerConfig(
                server_id="github",
                name="GitHub Integration",
                category="git",
                host="localhost",
                port=8004,
                command="npx",
                args=["@modelcontextprotocol/server-github"],
                tools=["get_repository", "list_files", "get_file_content", "create_issue", "search_repositories"]
            ),
            MCPServerConfig(
                server_id="sqlite", 
                name="SQLite Database",
                category="database",
                host="localhost",
                port=8005,
                command="npx",
                args=["@modelcontextprotocol/server-sqlite", "temp.db"],
                tools=["execute_query", "create_table", "insert_data", "select_data", "describe_table"]
            )
        ]
        
        for server_config in default_servers:
            await self.register_server(server_config)
            
    async def register_server(self, config: MCPServerConfig):
        """Register a new MCP server"""
        self.servers[config.server_id] = config
        
        # Initialize health status
        self.health_status[config.server_id] = ServerHealth(
            status=ServerStatus.STARTING,
            last_check=time.time(),
            response_time=0.0,
            error_count=0,
            total_requests=0,
            success_rate=1.0,
            uptime=0.0
        )
        
        # Register tools
        self.server_tools[config.server_id] = set(config.tools)
        for tool in config.tools:
            if tool not in self.tool_to_servers:
                self.tool_to_servers[tool] = []
            self.tool_to_servers[tool].append(config.server_id)
            
        # Register category
        if config.category not in self.category_servers:
            self.category_servers[config.category] = []
        self.category_servers[config.category].append(config.server_id)
        
        self.logger.info(f"Registered MCP server: {config.server_id}")
        
    async def start_server(self, server_id: str) -> bool:
        """Start an MCP server"""
        if server_id not in self.servers:
            self.logger.error(f"Server {server_id} not found in registry")
            return False
            
        config = self.servers[server_id]
        
        if server_id in self.running_processes:
            self.logger.warning(f"Server {server_id} is already running")
            return True
            
        try:
            # Start the server process
            if config.command:
                process = await asyncio.create_subprocess_exec(
                    config.command,
                    *config.args,
                    env={**config.env, "PORT": str(config.port)},
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                self.running_processes[server_id] = process
                
            # Update status
            self.health_status[server_id].status = ServerStatus.STARTING
            
            # Wait for server to be ready
            await asyncio.sleep(2.0)
            
            # Perform health check
            is_healthy = await self._health_check(server_id)
            if is_healthy:
                self.health_status[server_id].status = ServerStatus.HEALTHY
                self.logger.info(f"Started MCP server: {server_id}")
                return True
            else:
                await self.stop_server(server_id)
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start server {server_id}: {e}")
            self.health_status[server_id].status = ServerStatus.FAILED
            return False
            
    async def stop_server(self, server_id: str):
        """Stop an MCP server"""
        if server_id not in self.running_processes:
            return
            
        try:
            process = self.running_processes[server_id]
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5.0)
            
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            
        except Exception as e:
            self.logger.error(f"Error stopping server {server_id}: {e}")
            
        finally:
            del self.running_processes[server_id]
            self.health_status[server_id].status = ServerStatus.STOPPING
            self.logger.info(f"Stopped MCP server: {server_id}")
            
    async def get_servers_for_tool(self, tool_name: str) -> List[str]:
        """Get list of server IDs that provide a specific tool"""
        return self.tool_to_servers.get(tool_name, [])
        
    async def get_servers_by_category(self, category: str) -> List[str]:
        """Get list of server IDs in a specific category"""
        return self.category_servers.get(category, [])
        
    async def get_healthy_servers(self) -> List[str]:
        """Get list of healthy server IDs"""
        return [
            server_id for server_id, health in self.health_status.items()
            if health.status == ServerStatus.HEALTHY
        ]
        
    async def get_best_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get the best server for executing a specific tool"""
        candidate_servers = await self.get_servers_for_tool(tool_name)
        healthy_servers = await self.get_healthy_servers()
        
        # Filter to healthy servers that have the tool
        available_servers = [s for s in candidate_servers if s in healthy_servers]
        
        if not available_servers:
            return None
            
        # Select server with best performance metrics
        best_server = min(
            available_servers,
            key=lambda s: (
                self.health_status[s].response_time,
                -self.health_status[s].success_rate
            )
        )
        
        return best_server
        
    async def _health_check(self, server_id: str) -> bool:
        """Perform health check on a server"""
        if not self.session:
            return False
            
        config = self.servers[server_id]
        health_status = self.health_status[server_id]
        
        try:
            start_time = time.time()
            url = f"http://{config.host}:{config.port}{config.health_check_path}"
            
            async with self.session.get(url) as response:
                response_time = time.time() - start_time
                is_healthy = response.status == 200
                
                # Update health metrics
                health_status.last_check = time.time()
                health_status.response_time = response_time
                health_status.total_requests += 1
                
                if is_healthy:
                    health_status.success_rate = (
                        health_status.success_rate * (health_status.total_requests - 1) + 1.0
                    ) / health_status.total_requests
                else:
                    health_status.error_count += 1
                    health_status.success_rate = (
                        health_status.success_rate * (health_status.total_requests - 1)
                    ) / health_status.total_requests
                    
                return is_healthy
                
        except Exception as e:
            health_status.error_count += 1
            health_status.total_requests += 1
            health_status.success_rate = (
                health_status.success_rate * (health_status.total_requests - 1)
            ) / health_status.total_requests
            self.logger.error(f"Health check failed for {server_id}: {e}")
            return False
            
    async def start_health_monitoring(self):
        """Start periodic health monitoring"""
        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
        
    async def _health_monitor_loop(self):
        """Health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                for server_id in list(self.servers.keys()):
                    is_healthy = await self._health_check(server_id)
                    current_status = self.health_status[server_id].status
                    
                    if is_healthy and current_status == ServerStatus.UNHEALTHY:
                        self.health_status[server_id].status = ServerStatus.HEALTHY
                        self.logger.info(f"Server {server_id} recovered")
                    elif not is_healthy and current_status == ServerStatus.HEALTHY:
                        self.health_status[server_id].status = ServerStatus.UNHEALTHY
                        self.logger.warning(f"Server {server_id} became unhealthy")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics"""
        healthy_count = len(await self.get_healthy_servers())
        
        return {
            "total_servers": len(self.servers),
            "healthy_servers": healthy_count,
            "unhealthy_servers": len(self.servers) - healthy_count,
            "total_tools": len(self.tool_to_servers),
            "categories": list(self.category_servers.keys()),
            "server_details": {
                server_id: {
                    "status": health.status.value,
                    "response_time": health.response_time,
                    "success_rate": health.success_rate,
                    "uptime": time.time() - health.last_check,
                    "tools": list(self.server_tools.get(server_id, set()))
                }
                for server_id, health in self.health_status.items()
            }
        }
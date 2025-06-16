"""
Apeiron Tools Miner

Bittensor miner that orchestrates unlimited MCP tools across distributed servers.
Implements the vision: "Access Every Knowledge The World Has"
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bittensor as bt
import torch
from common.protocol import MCPSynapse, ToolRequest, ToolResult
from apeiron.registry import MCPServerRegistry
from apeiron.orchestrator import ToolOrchestrator


class ApeironMiner:
    """
    Apeiron Tools Miner Implementation
    
    Hosts multiple MCP servers and orchestrates unlimited tool execution
    for complex queries beyond traditional context window limitations.
    """
    
    def __init__(self, config: bt.config):
        self.config = config
        self.wallet = bt.wallet(config=config)
        self.subtensor = bt.subtensor(config=config)
        self.metagraph = bt.metagraph(netuid=config.netuid, network=self.subtensor.network, sync=False)
        
        # Initialize components
        self.registry = MCPServerRegistry()
        self.orchestrator = ToolOrchestrator(self.registry)
        
        # Setup logging
        self.setup_logging()
        
        # Setup axon
        self.axon = bt.axon(wallet=self.wallet, config=config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        
        self.logger.info(f"Apeiron Miner initialized on subnet {config.netuid}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('miner.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    async def forward(self, synapse: MCPSynapse) -> MCPSynapse:
        """
        Forward pass: Process MCP tool orchestration request
        
        This is where the magic happens - unlimited tool orchestration!
        """
        self.logger.info(f"Processing query: {synapse.user_query[:100]}...")
        start_time = time.time()
        
        try:
            # Set miner hotkey
            synapse.miner_hotkey = self.wallet.hotkey.ss58_address
            
            # Execute distributed tool orchestration
            execution_result = await self.orchestrator.execute_query(
                user_query=synapse.user_query,
                max_tools=synapse.max_tools,
                timeout=synapse.timeout
            )
            
            # Populate synapse with results
            if "tool_results" in execution_result:
                for result_dict in execution_result["tool_results"]:
                    if isinstance(result_dict, ToolResult):
                        synapse.add_tool_result(result_dict)
                    else:
                        # Convert dict to ToolResult if needed
                        tool_result = ToolResult(
                            tool_name=result_dict.get("tool_name", "unknown"),
                            success=result_dict.get("success", False),
                            result=result_dict.get("result"),
                            error=result_dict.get("error"),
                            execution_time=result_dict.get("execution_time", 0.0),
                            server_id=result_dict.get("server_id", "unknown")
                        )
                        synapse.add_tool_result(tool_result)
            
            synapse.final_response = execution_result.get("final_response", "")
            synapse.execution_summary = execution_result.get("analysis", {})
            
            execution_time = time.time() - start_time
            self.logger.info(f"Query processed in {execution_time:.2f}s with {synapse.tools_used} tools")
            
            return synapse
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {e}")
            self.logger.error(traceback.format_exc())
            
            # Return error response
            synapse.final_response = f"Processing error: {str(e)}"
            synapse.total_execution_time = time.time() - start_time
            return synapse
            
    async def blacklist(self, synapse: MCPSynapse) -> bool:
        """Blacklist function to filter requests"""
        # Basic blacklist logic
        if len(synapse.user_query) > 10000:  # Too long
            self.logger.warning("Blacklisted: Query too long")
            return True
            
        if synapse.max_tools > 200:  # Unreasonable tool count
            self.logger.warning("Blacklisted: Too many tools requested")
            return True
            
        # Check for malicious patterns
        malicious_patterns = ["rm -rf", "delete *", "drop table", "exec(", "eval("]
        query_lower = synapse.user_query.lower()
        if any(pattern in query_lower for pattern in malicious_patterns):
            self.logger.warning("Blacklisted: Potential malicious query")
            return True
            
        return False
        
    async def priority(self, synapse: MCPSynapse) -> float:
        """Priority function for request ordering"""
        # Higher priority for complex queries that showcase our capabilities
        priority = 0.0
        
        query_lower = synapse.user_query.lower()
        
        # Boost priority for multi-domain queries
        domains = ["file", "web", "database", "git", "memory", "api"]
        domain_count = sum(1 for domain in domains if domain in query_lower)
        priority += domain_count * 0.1
        
        # Boost priority for complex operations
        complex_keywords = ["analyze", "combine", "merge", "aggregate", "comprehensive"]
        complexity_boost = sum(0.05 for keyword in complex_keywords if keyword in query_lower)
        priority += complexity_boost
        
        # Boost priority based on expected tool usage
        if synapse.max_tools > 20:
            priority += 0.2
            
        return min(priority, 1.0)
        
    async def start_server(self):
        """Start the miner server"""
        try:
            # Initialize components
            await self.registry.initialize()
            await self.orchestrator.initialize()
            
            # Start all MCP servers
            self.logger.info("Starting MCP servers...")
            for server_id in self.registry.servers.keys():
                success = await self.registry.start_server(server_id)
                if success:
                    self.logger.info(f"Started MCP server: {server_id}")
                else:
                    self.logger.error(f"Failed to start MCP server: {server_id}")
                    
            # Sync metagraph
            self.logger.info("Syncing metagraph...")
            self.metagraph.sync(subtensor=self.subtensor)
            
            # Start axon
            self.logger.info("Starting axon...")
            self.axon.start()
            
            # Register on network
            if not self.subtensor.is_hotkey_registered(
                netuid=self.config.netuid,
                hotkey_ss58=self.wallet.hotkey.ss58_address
            ):
                self.logger.info("Registering on network...")
                success = self.subtensor.register(
                    wallet=self.wallet,
                    netuid=self.config.netuid,
                )
                if not success:
                    self.logger.error("Failed to register on network")
                    return
                    
            self.logger.info("=€ Apeiron Miner started successfully!")
            self.logger.info(f"Hotkey: {self.wallet.hotkey.ss58_address}")
            self.logger.info(f"Network: {self.subtensor.network}")
            self.logger.info(f"Netuid: {self.config.netuid}")
            
            # Print server status
            await self._print_server_status()
            
            # Keep the server running
            while True:
                try:
                    # Periodic health check and stats
                    await self._periodic_maintenance()
                    await asyncio.sleep(60)  # Check every minute
                    
                except KeyboardInterrupt:
                    self.logger.info("Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(10)
                    
        except Exception as e:
            self.logger.error(f"Error starting miner: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            await self.shutdown()
            
    async def _print_server_status(self):
        """Print status of all MCP servers"""
        self.logger.info("\n" + "="*60)
        self.logger.info("MCP SERVER STATUS")
        self.logger.info("="*60)
        
        registry_stats = await self.registry.get_registry_stats()
        self.logger.info(f"Total Servers: {registry_stats['total_servers']}")
        self.logger.info(f"Healthy Servers: {registry_stats['healthy_servers']}")
        self.logger.info(f"Available Tools: {registry_stats['total_tools']}")
        
        for server_id, details in registry_stats["server_details"].items():
            status_emoji = "" if details["status"] == "healthy" else "L"
            self.logger.info(f"{status_emoji} {server_id}: {details['status']} ({len(details['tools'])} tools)")
            
        self.logger.info("="*60)
        
    async def _periodic_maintenance(self):
        """Periodic maintenance tasks"""
        try:
            # Check server health
            healthy_servers = await self.registry.get_healthy_servers()
            total_servers = len(self.registry.servers)
            
            if len(healthy_servers) < total_servers:
                self.logger.warning(f"Only {len(healthy_servers)}/{total_servers} servers healthy")
                
            # Restart failed servers
            for server_id, health in self.registry.health_status.items():
                if health.status.value == "failed":
                    self.logger.info(f"Attempting to restart failed server: {server_id}")
                    await self.registry.start_server(server_id)
                    
            # Sync metagraph periodically
            self.metagraph.sync(subtensor=self.subtensor)
            
        except Exception as e:
            self.logger.error(f"Error in periodic maintenance: {e}")
            
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down Apeiron Miner...")
        
        try:
            # Stop orchestrator
            await self.orchestrator.shutdown()
            
            # Stop registry and servers
            await self.registry.shutdown()
            
            # Stop axon
            if hasattr(self, 'axon'):
                self.axon.stop()
                
            self.logger.info("Apeiron Miner shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def get_config():
    """Get miner configuration"""
    parser = bt.config()
    parser.add_argument("--netuid", type=int, default=122, help="Bittensor subnet netuid")
    parser.add_argument("--wallet.name", type=str, default="default", help="Wallet name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", help="Wallet hotkey")
    parser.add_argument("--logging.level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--axon.port", type=int, default=8091, help="Axon port")
    
    config = bt.config(parser=parser)
    return config


async def main():
    """Main miner function"""
    try:
        config = get_config()
        miner = ApeironMiner(config)
        await miner.start_server()
        
    except KeyboardInterrupt:
        print("\nGracefully shutting down miner...")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("> Starting Apeiron Tools Miner...")
    print("Vision: Access Every Knowledge The World Has")
    print("="*50)
    
    asyncio.run(main())
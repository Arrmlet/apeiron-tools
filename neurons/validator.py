"""
Apeiron Tools Validator

Bittensor validator that tests miners with complex queries and scores their
unlimited tool orchestration capabilities using multi-criteria evaluation.
"""

import asyncio
import logging
import time
import traceback
import random
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bittensor as bt
import torch
from common.protocol import MCPSynapse, ToolRequest
from apeiron.scoring import ResponseScorer, ValidationResult


class QueryGenerator:
    """Generates test queries of varying complexity for validator testing"""
    
    def __init__(self):
        self.simple_queries = [
            "List files in the current directory",
            "Fetch the content from https://example.com",
            "Store the value 'hello world' in memory with key 'greeting'",
            "Search for Python repositories on GitHub",
            "Create a temporary database table for users",
        ]
        
        self.medium_queries = [
            "Analyze the file structure of this project and create a summary report",
            "Fetch weather data from multiple cities and store it in a database",
            "Clone a GitHub repository, analyze its code structure, and generate documentation", 
            "Download web content, extract key information, and store it with searchable tags",
            "Process log files to identify errors and create an automated report",
        ]
        
        self.complex_queries = [
            "Build a comprehensive analysis of this codebase: list all files, analyze dependencies, check for security issues, generate documentation, and create a deployment plan",
            "Research market trends by fetching data from multiple sources, analyzing patterns, storing insights in a knowledge base, and generating a strategic report",
            "Orchestrate a complete CI/CD pipeline: analyze code quality, run tests, build documentation, deploy to staging, and notify stakeholders",
            "Create an intelligent knowledge management system: index documents, extract entities, build relationships, enable semantic search, and provide insights",
            "Perform comprehensive competitor analysis: gather data from multiple sources, analyze strengths/weaknesses, track trends, and generate strategic recommendations",
        ]
        
    def generate_query(self, complexity: str = "random") -> Tuple[str, str]:
        """Generate a test query of specified complexity"""
        if complexity == "random":
            complexity = random.choice(["simple", "medium", "complex"])
            
        if complexity == "simple":
            query = random.choice(self.simple_queries)
        elif complexity == "medium":
            query = random.choice(self.medium_queries)
        else:  # complex
            query = random.choice(self.complex_queries)
            
        return query, complexity
        
    def generate_batch(self, count: int = 10) -> List[Tuple[str, str]]:
        """Generate a batch of queries with mixed complexity"""
        queries = []
        
        # Distribution: 30% simple, 50% medium, 20% complex
        for i in range(count):
            if i < count * 0.3:
                complexity = "simple"
            elif i < count * 0.8:
                complexity = "medium"
            else:
                complexity = "complex"
                
            query, actual_complexity = self.generate_query(complexity)
            queries.append((query, actual_complexity))
            
        random.shuffle(queries)
        return queries


class ApeironValidator:
    """
    Apeiron Tools Validator Implementation
    
    Tests miners with diverse queries and evaluates their unlimited tool
    orchestration capabilities using intelligent multi-criteria scoring.
    """
    
    def __init__(self, config: bt.config):
        self.config = config
        self.wallet = bt.wallet(config=config)
        self.subtensor = bt.subtensor(config=config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = bt.metagraph(netuid=config.netuid, network=self.subtensor.network, sync=False)
        
        # Initialize components
        self.scorer = ResponseScorer()
        self.query_generator = QueryGenerator()
        
        # Scoring and validation state
        self.miner_scores: Dict[str, float] = {}
        self.validation_history: Dict[str, List[ValidationResult]] = {}
        self.last_updated_block = 0
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info(f"Apeiron Validator initialized on subnet {config.netuid}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('validator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    async def validate_miner(self, uid: int, hotkey: str) -> Optional[ValidationResult]:
        """Validate a single miner with a test query"""
        try:
            # Generate test query
            query, complexity = self.query_generator.generate_query()
            self.logger.info(f"Testing miner {uid} with {complexity} query: {query[:60]}...")
            
            # Create synapse
            synapse = MCPSynapse(
                user_query=query,
                max_tools=50 if complexity == "complex" else 25 if complexity == "medium" else 10,
                timeout=60.0 if complexity == "complex" else 30.0,
                query_complexity=complexity
            )
            
            # Query miner
            start_time = time.time()
            axon = self.metagraph.axons[uid]
            
            try:
                responses = await self.dendrite.forward(
                    axons=[axon],
                    synapse=synapse,
                    timeout=synapse.timeout + 10.0,  # Give extra time for network
                    deserialize=False
                )
                
                if not responses or len(responses) == 0:
                    self.logger.warning(f"No response from miner {uid}")
                    return None
                    
                response = responses[0]
                query_time = time.time() - start_time
                
                # Score the response
                validation_result = await self.scorer.score_response(response, complexity)
                validation_result.execution_stats["query_time"] = query_time
                validation_result.execution_stats["miner_uid"] = uid
                validation_result.execution_stats["miner_hotkey"] = hotkey
                
                self.logger.info(f"Miner {uid} scored {validation_result.total_score:.1f}/100")
                
                return validation_result
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout querying miner {uid}")
                return None
            except Exception as e:
                self.logger.error(f"Error querying miner {uid}: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error validating miner {uid}: {e}")
            return None
            
    async def validate_all_miners(self) -> Dict[int, ValidationResult]:
        """Validate all miners in parallel"""
        self.logger.info("Starting validation round for all miners")
        
        # Get active miners
        active_uids = []
        for uid in range(len(self.metagraph.hotkeys)):
            if self.metagraph.axons[uid].is_serving:
                active_uids.append(uid)
                
        if not active_uids:
            self.logger.warning("No active miners found")
            return {}
            
        self.logger.info(f"Validating {len(active_uids)} active miners")
        
        # Validate miners in parallel (limit concurrency)
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent validations
        
        async def validate_with_semaphore(uid: int) -> Tuple[int, Optional[ValidationResult]]:
            async with semaphore:
                hotkey = self.metagraph.hotkeys[uid]
                result = await self.validate_miner(uid, hotkey)
                return uid, result
                
        tasks = [validate_with_semaphore(uid) for uid in active_uids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        validation_results = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Validation task failed: {result}")
                continue
                
            uid, validation_result = result
            if validation_result:
                validation_results[uid] = validation_result
                
                # Update history
                hotkey = self.metagraph.hotkeys[uid]
                if hotkey not in self.validation_history:
                    self.validation_history[hotkey] = []
                self.validation_history[hotkey].append(validation_result)
                
                # Keep only last 50 results per miner
                if len(self.validation_history[hotkey]) > 50:
                    self.validation_history[hotkey] = self.validation_history[hotkey][-50:]
                    
        self.logger.info(f"Validation round complete: {len(validation_results)} miners scored")
        return validation_results
        
    def calculate_miner_scores(self, validation_results: Dict[int, ValidationResult]):
        """Calculate and update miner scores based on validation results"""
        self.logger.info("Calculating miner scores...")
        
        for uid, result in validation_results.items():
            hotkey = self.metagraph.hotkeys[uid]
            
            # Get recent validation history
            recent_results = self.validation_history.get(hotkey, [])[-10:]  # Last 10 validations
            
            if recent_results:
                # Calculate weighted average (more recent = higher weight)
                total_weighted_score = 0.0
                total_weight = 0.0
                
                for i, past_result in enumerate(recent_results):
                    weight = (i + 1) / len(recent_results)  # More recent = higher weight
                    total_weighted_score += past_result.total_score * weight
                    total_weight += weight
                    
                average_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
                self.miner_scores[hotkey] = average_score
                
                self.logger.info(f"Miner {uid} ({hotkey[:8]}...): {average_score:.1f}/100 (based on {len(recent_results)} validations)")
            else:
                self.miner_scores[hotkey] = 0.0
                
    def set_weights(self):
        """Set weights on the network based on miner scores"""
        self.logger.info("Setting weights on network...")
        
        try:
            # Prepare weights array
            weights = torch.zeros(len(self.metagraph.hotkeys))
            
            for uid, hotkey in enumerate(self.metagraph.hotkeys):
                if hotkey in self.miner_scores:
                    # Normalize score to 0-1 range and apply sigmoid for distribution
                    normalized_score = self.miner_scores[hotkey] / 100.0
                    sigmoid_score = 1 / (1 + torch.exp(-5 * (normalized_score - 0.5)))
                    weights[uid] = sigmoid_score
                else:
                    weights[uid] = 0.0
                    
            # Normalize weights to sum to 1
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # If no miners have scores, use equal weights
                weights = torch.ones(len(self.metagraph.hotkeys)) / len(self.metagraph.hotkeys)
                
            # Set weights on network
            result, message = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=torch.arange(len(weights)),
                weights=weights,
                wait_for_finalization=True,
                wait_for_inclusion=True,
            )
            
            if result:
                self.logger.info(" Successfully set weights on network")
                self.logger.info(f"Average weight: {weights.mean():.4f}, Max weight: {weights.max():.4f}")
            else:
                self.logger.error(f"L Failed to set weights: {message}")
                
        except Exception as e:
            self.logger.error(f"Error setting weights: {e}")
            
    async def validation_loop(self):
        """Main validation loop"""
        self.logger.info("Starting validation loop...")
        
        validation_interval = 300  # 5 minutes between validation rounds
        weight_update_interval = 900  # 15 minutes between weight updates
        last_weight_update = 0
        
        while True:
            try:
                # Sync metagraph
                self.logger.info("Syncing metagraph...")
                self.metagraph.sync(subtensor=self.subtensor)
                
                # Run validation round
                validation_results = await self.validate_all_miners()
                
                if validation_results:
                    # Calculate scores
                    self.calculate_miner_scores(validation_results)
                    
                    # Print leaderboard
                    self._print_leaderboard()
                    
                    # Update weights if enough time has passed
                    current_time = time.time()
                    if current_time - last_weight_update >= weight_update_interval:
                        self.set_weights()
                        last_weight_update = current_time
                        
                # Wait before next validation
                await asyncio.sleep(validation_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Error in validation loop: {e}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(60)  # Wait 1 minute before retrying
                
    def _print_leaderboard(self):
        """Print current miner leaderboard"""
        if not self.miner_scores:
            return
            
        self.logger.info("\n" + "="*80)
        self.logger.info("APEIRON TOOLS LEADERBOARD")
        self.logger.info("="*80)
        
        # Sort miners by score
        sorted_miners = sorted(self.miner_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (hotkey, score) in enumerate(sorted_miners[:10]):  # Top 10
            # Find UID for this hotkey
            uid = None
            for u, hk in enumerate(self.metagraph.hotkeys):
                if hk == hotkey:
                    uid = u
                    break
                    
            if uid is not None:
                # Get recent performance stats
                recent_results = self.validation_history.get(hotkey, [])[-5:]
                if recent_results:
                    avg_tools = sum(r.execution_stats.get("tools_evaluated", 0) for r in recent_results) / len(recent_results)
                    avg_time = sum(r.execution_stats.get("total_execution_time", 0) for r in recent_results) / len(recent_results)
                    stats = f"Tools: {avg_tools:.1f}, Time: {avg_time:.1f}s"
                else:
                    stats = "No recent data"
                    
                rank_emoji = ">G" if i == 0 else ">H" if i == 1 else ">I" if i == 2 else f"{i+1:2d}."
                self.logger.info(f"{rank_emoji} UID {uid:3d} | {score:5.1f}/100 | {hotkey[:16]}... | {stats}")
                
        self.logger.info("="*80)
        
    async def start_validator(self):
        """Start the validator"""
        try:
            self.logger.info("=€ Starting Apeiron Tools Validator...")
            self.logger.info(f"Hotkey: {self.wallet.hotkey.ss58_address}")
            self.logger.info(f"Network: {self.subtensor.network}")
            self.logger.info(f"Netuid: {self.config.netuid}")
            
            # Check if registered
            if not self.subtensor.is_hotkey_registered(
                netuid=self.config.netuid,
                hotkey_ss58=self.wallet.hotkey.ss58_address
            ):
                self.logger.error("L Validator not registered on network")
                self.logger.info("Please register first: btcli subnet register --netuid 122")
                return
                
            self.logger.info(" Validator registered and ready")
            
            # Start validation loop
            await self.validation_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting validator: {e}")
            self.logger.error(traceback.format_exc())


def get_config():
    """Get validator configuration"""
    parser = bt.config()
    parser.add_argument("--netuid", type=int, default=122, help="Bittensor subnet netuid")
    parser.add_argument("--wallet.name", type=str, default="default", help="Wallet name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", help="Wallet hotkey")
    parser.add_argument("--logging.level", type=str, default="INFO", help="Logging level")
    
    config = bt.config(parser=parser)
    return config


async def main():
    """Main validator function"""
    try:
        config = get_config()
        validator = ApeironValidator(config)
        await validator.start_validator()
        
    except KeyboardInterrupt:
        print("\nGracefully shutting down validator...")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("= Starting Apeiron Tools Validator...")
    print("Mission: Evaluate unlimited tool orchestration capabilities")
    print("="*60)
    
    asyncio.run(main())
"""
Response Scoring System for Apeiron Tools

Implements multi-criteria scoring for validator assessment of miner responses.
Evaluates tool coverage, result quality, execution speed, error handling, and efficiency.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from .protocol import ToolResult, MCPSynapse


class ScoreCategory(Enum):
    """Scoring categories with weights"""
    TOOL_COVERAGE = ("tool_coverage", 0.30)
    RESULT_QUALITY = ("result_quality", 0.25)
    EXECUTION_SPEED = ("execution_speed", 0.20)
    ERROR_HANDLING = ("error_handling", 0.15)
    RESOURCE_EFFICIENCY = ("resource_efficiency", 0.10)
    
    def __init__(self, category_name: str, weight: float):
        self.category_name = category_name
        self.weight = weight


@dataclass
class ScoreBreakdown:
    """Detailed score breakdown"""
    category: ScoreCategory
    score: float
    max_score: float
    details: Dict[str, Any]
    
    @property
    def weighted_score(self) -> float:
        """Get weighted score for this category"""
        return (self.score / self.max_score) * self.category.weight * 100


@dataclass
class ValidationResult:
    """Complete validation result with scores and feedback"""
    total_score: float
    category_scores: List[ScoreBreakdown]
    feedback: str
    recommendations: List[str]
    execution_stats: Dict[str, Any]
    timestamp: float


class ResponseScorer:
    """
    Multi-criteria scoring system for evaluating miner responses.
    
    Implements intelligent scoring across multiple dimensions to assess
    the quality of distributed tool orchestration responses.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality assessment keywords
        self.quality_indicators = {
            "high": ["accurate", "complete", "comprehensive", "detailed", "correct", "precise"],
            "medium": ["partial", "basic", "limited", "simple", "general"],
            "low": ["error", "failed", "incomplete", "wrong", "missing", "broken"]
        }
        
        # Expected tool categories for different query types
        self.query_tool_mapping = {
            "filesystem": ["read_file", "write_file", "list_directory", "create_directory"],
            "web": ["fetch", "get_webpage", "download_file"],
            "database": ["execute_query", "select_data", "insert_data"],
            "git": ["get_repository", "list_files", "get_file_content"],
            "memory": ["store", "retrieve", "search"],
            "analysis": ["analyze", "compute", "calculate", "process"]
        }
        
    async def score_response(self, synapse: MCPSynapse, expected_complexity: str = "medium") -> ValidationResult:
        """Score a complete miner response"""
        start_time = time.time()
        
        try:
            # Score each category
            category_scores = []
            
            # 1. Tool Coverage (30%)
            coverage_score = await self._score_tool_coverage(synapse, expected_complexity)
            category_scores.append(coverage_score)
            
            # 2. Result Quality (25%) 
            quality_score = await self._score_result_quality(synapse)
            category_scores.append(quality_score)
            
            # 3. Execution Speed (20%)
            speed_score = await self._score_execution_speed(synapse, expected_complexity)
            category_scores.append(speed_score)
            
            # 4. Error Handling (15%)
            error_score = await self._score_error_handling(synapse)
            category_scores.append(error_score)
            
            # 5. Resource Efficiency (10%)
            efficiency_score = await self._score_resource_efficiency(synapse)
            category_scores.append(efficiency_score)
            
            # Calculate total weighted score
            total_score = sum(score.weighted_score for score in category_scores)
            
            # Generate feedback and recommendations
            feedback = self._generate_feedback(category_scores, synapse)
            recommendations = self._generate_recommendations(category_scores, synapse)
            
            # Execution statistics
            execution_stats = {
                "scoring_time": time.time() - start_time,
                "tools_evaluated": len(synapse.tool_results),
                "successful_tools": len(synapse.get_successful_tools()),
                "failed_tools": len(synapse.get_failed_tools()),
                "total_execution_time": synapse.total_execution_time
            }
            
            return ValidationResult(
                total_score=total_score,
                category_scores=category_scores,
                feedback=feedback,
                recommendations=recommendations,
                execution_stats=execution_stats,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error scoring response: {e}")
            return ValidationResult(
                total_score=0.0,
                category_scores=[],
                feedback=f"Scoring error: {str(e)}",
                recommendations=["Fix critical errors before resubmission"],
                execution_stats={"error": str(e)},
                timestamp=time.time()
            )
            
    async def _score_tool_coverage(self, synapse: MCPSynapse, expected_complexity: str) -> ScoreBreakdown:
        """Score tool coverage - how well the response covers relevant tools"""
        max_score = 100.0
        
        # Analyze query to determine expected tool categories
        query_lower = synapse.user_query.lower()
        expected_categories = []
        
        for category, keywords in [
            ("filesystem", ["file", "directory", "folder", "read", "write"]),
            ("web", ["url", "website", "fetch", "download", "http"]),
            ("database", ["database", "sql", "query", "table"]),
            ("git", ["github", "repository", "repo", "commit"]),
            ("memory", ["remember", "store", "save", "recall"]),
        ]:
            if any(keyword in query_lower for keyword in keywords):
                expected_categories.append(category)
                
        # Expected tool count based on complexity
        complexity_multiplier = {"simple": 1, "medium": 2, "complex": 3}
        expected_tool_count = len(expected_categories) * complexity_multiplier.get(expected_complexity, 2)
        
        # Evaluate actual coverage
        actual_tools = len(synapse.tool_results)
        successful_tools = len(synapse.get_successful_tools())
        
        # Score components
        tool_count_score = min(actual_tools / max(expected_tool_count, 1), 1.0) * 40
        success_rate_score = (successful_tools / max(actual_tools, 1)) * 40
        category_coverage_score = len(set(self._get_tool_categories(synapse))) / max(len(expected_categories), 1) * 20
        
        total_score = tool_count_score + success_rate_score + min(category_coverage_score, 20)
        
        return ScoreBreakdown(
            category=ScoreCategory.TOOL_COVERAGE,
            score=total_score,
            max_score=max_score,
            details={
                "expected_categories": expected_categories,
                "actual_tools": actual_tools,
                "successful_tools": successful_tools,
                "expected_tool_count": expected_tool_count,
                "category_coverage": len(set(self._get_tool_categories(synapse))),
                "tool_count_score": tool_count_score,
                "success_rate_score": success_rate_score,
                "category_coverage_score": min(category_coverage_score, 20)
            }
        )
        
    async def _score_result_quality(self, synapse: MCPSynapse) -> ScoreBreakdown:
        """Score result quality based on content analysis"""
        max_score = 100.0
        
        if not synapse.final_response:
            return ScoreBreakdown(
                category=ScoreCategory.RESULT_QUALITY,
                score=0.0,
                max_score=max_score,
                details={"error": "No final response provided"}
            )
            
        response_text = synapse.final_response.lower()
        
        # Quality indicators analysis
        high_quality_count = sum(1 for indicator in self.quality_indicators["high"] if indicator in response_text)
        medium_quality_count = sum(1 for indicator in self.quality_indicators["medium"] if indicator in response_text)
        low_quality_count = sum(1 for indicator in self.quality_indicators["low"] if indicator in response_text)
        
        # Content analysis
        response_length = len(synapse.final_response)
        has_structured_output = any(marker in synapse.final_response for marker in ["- ", "1.", "* ", "\n\n"])
        addresses_query = len(set(synapse.user_query.lower().split()) & set(response_text.split())) > 2
        
        # Calculate quality score
        quality_indicators_score = (high_quality_count * 3 - low_quality_count) * 10
        length_score = min(response_length / 500, 1.0) * 20  # Optimal around 500 chars
        structure_score = 20 if has_structured_output else 10
        relevance_score = 30 if addresses_query else 10
        
        total_score = max(0, min(100, quality_indicators_score + length_score + structure_score + relevance_score))
        
        return ScoreBreakdown(
            category=ScoreCategory.RESULT_QUALITY,
            score=total_score,
            max_score=max_score,
            details={
                "response_length": response_length,
                "high_quality_indicators": high_quality_count,
                "low_quality_indicators": low_quality_count,
                "has_structured_output": has_structured_output,
                "addresses_query": addresses_query,
                "quality_indicators_score": quality_indicators_score,
                "length_score": length_score,
                "structure_score": structure_score,
                "relevance_score": relevance_score
            }
        )
        
    async def _score_execution_speed(self, synapse: MCPSynapse, expected_complexity: str) -> ScoreBreakdown:
        """Score execution speed relative to expected performance"""
        max_score = 100.0
        
        # Expected time thresholds based on complexity
        time_thresholds = {
            "simple": {"excellent": 2.0, "good": 5.0, "acceptable": 10.0},
            "medium": {"excellent": 5.0, "good": 15.0, "acceptable": 30.0},
            "complex": {"excellent": 15.0, "good": 30.0, "acceptable": 60.0}
        }
        
        thresholds = time_thresholds.get(expected_complexity, time_thresholds["medium"])
        execution_time = synapse.total_execution_time
        
        # Calculate speed score
        if execution_time <= thresholds["excellent"]:
            speed_score = 100.0
        elif execution_time <= thresholds["good"]:
            speed_score = 80.0
        elif execution_time <= thresholds["acceptable"]:
            speed_score = 60.0
        else:
            speed_score = max(20.0, 60.0 - (execution_time - thresholds["acceptable"]) * 2)
            
        # Parallel execution bonus
        if len(synapse.tool_results) > 1:
            avg_tool_time = execution_time / len(synapse.tool_results)
            if avg_tool_time < execution_time / 2:  # Indicates parallel execution
                speed_score = min(100.0, speed_score + 10.0)
                
        return ScoreBreakdown(
            category=ScoreCategory.EXECUTION_SPEED,
            score=speed_score,
            max_score=max_score,
            details={
                "execution_time": execution_time,
                "expected_complexity": expected_complexity,
                "thresholds": thresholds,
                "tools_executed": len(synapse.tool_results),
                "avg_tool_time": execution_time / max(len(synapse.tool_results), 1)
            }
        )
        
    async def _score_error_handling(self, synapse: MCPSynapse) -> ScoreBreakdown:
        """Score error handling and recovery capabilities"""
        max_score = 100.0
        
        total_tools = len(synapse.tool_results)
        if total_tools == 0:
            return ScoreBreakdown(
                category=ScoreCategory.ERROR_HANDLING,
                score=0.0,
                max_score=max_score,
                details={"error": "No tools executed"}
            )
            
        failed_tools = synapse.get_failed_tools()
        successful_tools = synapse.get_successful_tools()
        
        # Base score from success rate
        success_rate = len(successful_tools) / total_tools
        base_score = success_rate * 60
        
        # Error message quality
        error_quality_score = 0
        for failed_tool in failed_tools:
            if failed_tool.error:
                if any(word in failed_tool.error.lower() for word in ["timeout", "connection", "network"]):
                    error_quality_score += 5  # Infrastructure errors are acceptable
                elif "circuit breaker" in failed_tool.error.lower():
                    error_quality_score += 8  # Good resilience pattern
                elif len(failed_tool.error) > 10:
                    error_quality_score += 3  # Detailed error message
                    
        # Recovery attempts (inferred from execution patterns)
        recovery_score = 0
        if failed_tools and successful_tools:
            recovery_score = 10  # Some tools succeeded despite failures
            
        # Graceful degradation
        degradation_score = 0
        if failed_tools and synapse.final_response:
            if "partial" in synapse.final_response.lower() or "some tools" in synapse.final_response.lower():
                degradation_score = 15  # Acknowledged limitations
                
        total_score = min(100.0, base_score + error_quality_score + recovery_score + degradation_score)
        
        return ScoreBreakdown(
            category=ScoreCategory.ERROR_HANDLING,
            score=total_score,
            max_score=max_score,
            details={
                "total_tools": total_tools,
                "failed_tools": len(failed_tools),
                "success_rate": success_rate,
                "base_score": base_score,
                "error_quality_score": error_quality_score,
                "recovery_score": recovery_score,
                "degradation_score": degradation_score
            }
        )
        
    async def _score_resource_efficiency(self, synapse: MCPSynapse) -> ScoreBreakdown:
        """Score resource efficiency and optimization"""
        max_score = 100.0
        
        if not synapse.tool_results:
            return ScoreBreakdown(
                category=ScoreCategory.RESOURCE_EFFICIENCY,
                score=0.0,
                max_score=max_score,
                details={"error": "No tools executed"}
            )
            
        # Tool selection efficiency
        unique_servers = len(set(result.server_id for result in synapse.tool_results))
        total_tools = len(synapse.tool_results)
        
        # Prefer diverse server usage but not excessive
        server_diversity_score = min(unique_servers / max(total_tools / 3, 1), 1.0) * 30
        
        # Execution time per tool efficiency
        avg_execution_time = synapse.total_execution_time / total_tools
        if avg_execution_time < 1.0:
            time_efficiency_score = 30
        elif avg_execution_time < 3.0:
            time_efficiency_score = 20
        else:
            time_efficiency_score = max(5, 20 - (avg_execution_time - 3.0) * 2)
            
        # Tool relevance (avoid unnecessary tools)
        query_words = set(synapse.user_query.lower().split())
        relevant_tools = 0
        
        for result in synapse.tool_results:
            if result.success and result.result:
                result_str = str(result.result).lower()
                if len(query_words & set(result_str.split())) > 0:
                    relevant_tools += 1
                    
        relevance_score = (relevant_tools / total_tools) * 40 if total_tools > 0 else 0
        
        total_score = server_diversity_score + time_efficiency_score + relevance_score
        
        return ScoreBreakdown(
            category=ScoreCategory.RESOURCE_EFFICIENCY,
            score=total_score,
            max_score=max_score,
            details={
                "unique_servers": unique_servers,
                "total_tools": total_tools,
                "avg_execution_time": avg_execution_time,
                "relevant_tools": relevant_tools,
                "server_diversity_score": server_diversity_score,
                "time_efficiency_score": time_efficiency_score,
                "relevance_score": relevance_score
            }
        )
        
    def _get_tool_categories(self, synapse: MCPSynapse) -> List[str]:
        """Extract tool categories from synapse results"""
        categories = []
        for result in synapse.tool_results:
            tool_name = result.tool_name.lower()
            if any(fs_tool in tool_name for fs_tool in ["file", "directory", "read", "write"]):
                categories.append("filesystem")
            elif any(web_tool in tool_name for web_tool in ["fetch", "download", "http", "url"]):
                categories.append("web")
            elif any(db_tool in tool_name for db_tool in ["query", "select", "insert", "database"]):
                categories.append("database")
            elif any(git_tool in tool_name for git_tool in ["repository", "commit", "branch"]):
                categories.append("git")
            elif any(mem_tool in tool_name for mem_tool in ["store", "retrieve", "memory"]):
                categories.append("memory")
        return categories
        
    def _generate_feedback(self, category_scores: List[ScoreBreakdown], synapse: MCPSynapse) -> str:
        """Generate human-readable feedback"""
        feedback_parts = []
        
        # Overall performance
        total_score = sum(score.weighted_score for score in category_scores)
        if total_score >= 80:
            feedback_parts.append("Excellent performance with strong tool orchestration.")
        elif total_score >= 60:
            feedback_parts.append("Good performance with room for optimization.")
        else:
            feedback_parts.append("Performance needs significant improvement.")
            
        # Category-specific feedback
        for score in category_scores:
            if score.weighted_score < 15:  # Below 50% of max weighted score
                feedback_parts.append(f"Low {score.category.category_name.replace('_', ' ')} score needs attention.")
                
        return " ".join(feedback_parts)
        
    def _generate_recommendations(self, category_scores: List[ScoreBreakdown], synapse: MCPSynapse) -> List[str]:
        """Generate specific improvement recommendations"""
        recommendations = []
        
        for score in category_scores:
            if score.weighted_score < 15:  # Below 50% of max weighted score
                if score.category == ScoreCategory.TOOL_COVERAGE:
                    recommendations.append("Increase tool diversity and ensure comprehensive coverage of query domains")
                elif score.category == ScoreCategory.RESULT_QUALITY:
                    recommendations.append("Improve response quality with more detailed and structured outputs")
                elif score.category == ScoreCategory.EXECUTION_SPEED:
                    recommendations.append("Optimize execution speed through better parallelization and server selection")
                elif score.category == ScoreCategory.ERROR_HANDLING:
                    recommendations.append("Implement better error recovery and graceful degradation strategies")
                elif score.category == ScoreCategory.RESOURCE_EFFICIENCY:
                    recommendations.append("Optimize resource usage by selecting more relevant tools and servers")
                    
        if not recommendations:
            recommendations.append("Continue current performance level and monitor for consistency")
            
        return recommendations
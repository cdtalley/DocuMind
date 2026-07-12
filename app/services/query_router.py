"""
Intelligent Query Router and Classification System

This module implements sophisticated query routing for optimal RAG performance:

- Multi-dimensional query classification and analysis
- Adaptive routing to different retrieval strategies and libraries
- Context-aware parameter optimization
- Performance-based routing decisions with feedback loops
- Specialized handlers for different query types

Demonstrates advanced understanding of:
- Query intent recognition and classification
- Adaptive system architecture
- Performance-driven decision making
- Context-aware optimization
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

from app.models.response_models import AnswerResponse
from app.services.agentic_rag import QueryAnalysis, QueryComplexity, QueryType
from app.utils.ollama_client import OllamaClient

logger = logging.getLogger("documind.routing")


class LibraryType(Enum):
    """Available knowledge libraries"""
    PAPERS = "papers"
    PUBLIC = "public"
    MIXED = "mixed"


class RoutingStrategy(Enum):
    """Query routing strategies"""
    PERFORMANCE_BASED = "performance"  # Route based on historical performance
    CONTENT_BASED = "content"  # Route based on query content analysis
    ADAPTIVE = "adaptive"  # Combine multiple factors
    LOAD_BALANCED = "load_balanced"  # Balance across resources


@dataclass
class RoutingDecision:
    """Query routing decision with rationale"""
    library: LibraryType
    retrieval_strategy: str
    query_mode: str
    top_k: int
    additional_params: dict[str, Any]
    confidence: float
    reasoning: str
    expected_performance: dict[str, float]


@dataclass
class QueryContext:
    """Extended query context for routing decisions"""
    original_query: str
    processed_query: str
    user_session_id: str | None = None
    previous_queries: list[str] = None
    user_preferences: dict[str, Any] = None
    time_constraints: float | None = None  # Max acceptable latency
    quality_requirements: float | None = None  # Min acceptable quality
    
    def __post_init__(self):
        if self.previous_queries is None:
            self.previous_queries = []
        if self.user_preferences is None:
            self.user_preferences = {}


@dataclass
class PerformanceProfile:
    """Performance profile for a routing configuration"""
    library: str
    strategy: str
    query_mode: str
    
    # Performance metrics
    avg_latency: float = 0.0
    avg_confidence: float = 0.0
    success_rate: float = 0.0
    user_satisfaction: float = 0.0
    
    # Usage statistics
    query_count: int = 0
    last_used: float = 0.0
    
    def update_performance(self, latency: float, confidence: float, success: bool, satisfaction: float = None):
        """Update performance metrics with new data point"""
        self.query_count += 1
        self.last_used = time.time()
        
        # Exponential moving average for responsiveness to recent performance
        alpha = 0.1  # Learning rate
        
        self.avg_latency = (1 - alpha) * self.avg_latency + alpha * latency
        self.avg_confidence = (1 - alpha) * self.avg_confidence + alpha * confidence
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        
        if satisfaction is not None:
            self.user_satisfaction = (1 - alpha) * self.user_satisfaction + alpha * satisfaction


class QueryClassifier:
    """
    Advanced query classification system with multiple dimensions
    """
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        
        # Classification patterns
        self.domain_patterns = {
            "academic": [r"\bpaper\b", r"\bresearch\b", r"\bstudy\b", r"\bmethod\b", r"\balgorithm\b"],
            "technical": [r"\bimplement\b", r"\bcode\b", r"\bAPI\b", r"\btechnical\b", r"\barchitecture\b"],
            "general": [r"\bwhat is\b", r"\bhow to\b", r"\bexplain\b", r"\bdefinition\b"],
            "comparative": [r"\bcompare\b", r"\bvs\b", r"\bversus\b", r"\bdifference\b", r"\bbetter\b"],
        }
        
        self.complexity_indicators = {
            "simple": [r"^what is", r"^define", r"^who", r"^when", r"^where"],
            "moderate": [r"how does", r"why", r"explain", r"describe"],
            "complex": [r"analyze", r"evaluate", r"compare.*and", r"relationship between"],
        }
        
        logger.info("Initialized Query Classifier with pattern matching and LLM analysis")
    
    def classify_query_comprehensive(self, query: str, context: QueryContext = None) -> dict[str, Any]:
        """
        Comprehensive multi-dimensional query classification
        """
        logger.info(f"Classifying query: {query[:50]}...")
        
        # Basic pattern-based classification
        domain = self._classify_domain(query)
        complexity = self._classify_complexity(query)
        intent = self._classify_intent(query)
        
        # LLM-based detailed analysis
        detailed_analysis = self._llm_detailed_analysis(query)
        
        # Context-aware adjustments
        if context:
            domain, complexity, intent = self._adjust_for_context(
                domain, complexity, intent, context
            )
        
        classification = {
            "domain": domain,
            "complexity": complexity,
            "intent": intent,
            "detailed_analysis": detailed_analysis,
            "requires_multi_step": self._requires_multi_step(query, complexity, intent),
            "expected_library": self._predict_best_library(domain, intent),
            "urgency": self._assess_urgency(query, context),
            "technical_level": self._assess_technical_level(query, detailed_analysis),
            "confidence": detailed_analysis.get("confidence", 0.7)
        }
        
        return classification
    
    def _classify_domain(self, query: str) -> str:
        """Classify query domain using pattern matching"""
        query_lower = query.lower()
        
        domain_scores = {}
        for domain, patterns in self.domain_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            if score > 0:
                domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else "general"
    
    def _classify_complexity(self, query: str) -> str:
        """Classify query complexity"""
        query_lower = query.lower()
        
        for complexity, patterns in self.complexity_indicators.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                return complexity
        
        # Length-based fallback
        word_count = len(query.split())
        if word_count <= 5:
            return "simple"
        elif word_count <= 15:
            return "moderate"
        else:
            return "complex"
    
    def _classify_intent(self, query: str) -> str:
        """Classify user intent"""
        query_lower = query.lower()
        
        intent_patterns = {
            "factual": [r"^what", r"^who", r"^when", r"^where", r"^which"],
            "procedural": [r"^how", r"step", r"process", r"method", r"way to"],
            "analytical": [r"analyze", r"evaluate", r"assess", r"critique", r"review"],
            "comparative": [r"compare", r"vs", r"versus", r"difference", r"similarity"],
            "exploratory": [r"explore", r"investigate", r"research", r"study"],
            "creative": [r"generate", r"create", r"design", r"brainstorm"],
        }
        
        for intent, patterns in intent_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                return intent
        
        return "informational"
    
    def _llm_detailed_analysis(self, query: str) -> dict[str, Any]:
        """Use LLM for detailed query analysis"""
        analysis_prompt = f"""
        Analyze this query and provide structured insights for optimal retrieval strategy.

        Query: {query}

        Provide analysis in this JSON format:
        {{
          "primary_concepts": ["concept1", "concept2"],
          "secondary_concepts": ["concept3", "concept4"],
          "information_need": "factual|conceptual|procedural|analytical",
          "scope": "narrow|broad|multi_faceted",
          "technical_depth": "basic|intermediate|advanced",
          "time_sensitivity": "current|historical|timeless",
          "answer_type": "short_answer|explanation|analysis|comparison",
          "confidence": 0.0-1.0
        }}
        """
        
        try:
            response = self.ollama_client.chat(
                [{"role": "user", "content": analysis_prompt}], 
                temperature=0.2
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
        
        # Fallback analysis
        return {
            "primary_concepts": [],
            "information_need": "factual",
            "scope": "narrow",
            "technical_depth": "basic",
            "confidence": 0.5
        }
    
    def _adjust_for_context(
        self, 
        domain: str, 
        complexity: str, 
        intent: str, 
        context: QueryContext
    ) -> tuple[str, str, str]:
        """Adjust classification based on context"""
        # Consider previous queries
        if context.previous_queries:
            recent_query = context.previous_queries[-1] if context.previous_queries else ""
            if "research" in recent_query.lower() or "paper" in recent_query.lower():
                domain = "academic"
        
        # Consider user preferences
        if context.user_preferences.get("preferred_detail_level") == "high":
            if complexity == "simple":
                complexity = "moderate"
        
        # Consider time constraints
        if context.time_constraints and context.time_constraints < 3.0:
            # Under time pressure, prefer simpler approaches
            if complexity == "complex":
                complexity = "moderate"
        
        return domain, complexity, intent
    
    def _requires_multi_step(self, query: str, complexity: str, intent: str) -> bool:
        """Determine if query requires multi-step processing"""
        multi_step_indicators = [
            "step by step", "process of", "how does.*work", "relationship between",
            "compare.*and.*in terms of", "analyze.*impact"
        ]
        
        if any(re.search(pattern, query.lower()) for pattern in multi_step_indicators):
            return True
        
        return complexity == "complex" and intent in ["analytical", "comparative", "exploratory"]
    
    def _predict_best_library(self, domain: str, intent: str) -> LibraryType:
        """Predict the best library for the query"""
        if domain == "academic" or intent in ["analytical", "comparative"]:
            return LibraryType.PAPERS
        elif domain == "general" or intent in ["factual", "informational"]:
            return LibraryType.PUBLIC
        else:
            return LibraryType.MIXED
    
    def _assess_urgency(self, query: str, context: QueryContext) -> str:
        """Assess query urgency"""
        urgency_indicators = ["urgent", "quickly", "asap", "immediately", "fast"]
        
        if any(indicator in query.lower() for indicator in urgency_indicators):
            return "high"
        
        if context and context.time_constraints and context.time_constraints < 5.0:
            return "high"
        
        return "normal"
    
    def _assess_technical_level(self, query: str, detailed_analysis: dict) -> str:
        """Assess technical level required"""
        technical_terms = [
            "algorithm", "implementation", "architecture", "framework",
            "methodology", "optimization", "performance", "scalability"
        ]
        
        tech_count = sum(1 for term in technical_terms if term in query.lower())
        
        if tech_count >= 2 or detailed_analysis.get("technical_depth") == "advanced":
            return "advanced"
        elif tech_count >= 1 or detailed_analysis.get("technical_depth") == "intermediate":
            return "intermediate"
        else:
            return "basic"


class PerformanceOracle:
    """
    Tracks and predicts performance for different routing configurations
    """
    
    def __init__(self):
        self.performance_profiles: dict[str, PerformanceProfile] = {}
        self.global_stats = {
            "total_queries": 0,
            "avg_latency": 0.0,
            "avg_confidence": 0.0
        }
        
        logger.info("Initialized Performance Oracle")
    
    def get_profile_key(self, library: str, strategy: str, query_mode: str) -> str:
        """Generate unique key for performance profile"""
        return f"{library}:{strategy}:{query_mode}"
    
    def predict_performance(
        self, 
        library: LibraryType, 
        strategy: str, 
        query_mode: str,
        query_classification: dict[str, Any]
    ) -> dict[str, float]:
        """Predict performance for a routing configuration"""
        profile_key = self.get_profile_key(library.value, strategy, query_mode)
        
        if profile_key in self.performance_profiles:
            profile = self.performance_profiles[profile_key]
            
            # Adjust predictions based on query characteristics
            complexity_multiplier = self._get_complexity_multiplier(
                query_classification.get("complexity", "moderate")
            )
            
            predicted = {
                "latency": profile.avg_latency * complexity_multiplier,
                "confidence": profile.avg_confidence,
                "success_probability": profile.success_rate,
                "user_satisfaction": profile.user_satisfaction,
                "data_confidence": min(1.0, profile.query_count / 50.0)  # More data = higher confidence
            }
        else:
            # No historical data, use informed estimates
            predicted = self._estimate_baseline_performance(library, strategy, query_mode)
        
        return predicted
    
    def update_performance(
        self,
        library: LibraryType,
        strategy: str,
        query_mode: str,
        actual_latency: float,
        actual_confidence: float,
        success: bool,
        user_satisfaction: float = None
    ) -> None:
        """Update performance profile with actual results"""
        profile_key = self.get_profile_key(library.value, strategy, query_mode)
        
        if profile_key not in self.performance_profiles:
            self.performance_profiles[profile_key] = PerformanceProfile(
                library=library.value,
                strategy=strategy,
                query_mode=query_mode
            )
        
        profile = self.performance_profiles[profile_key]
        profile.update_performance(actual_latency, actual_confidence, success, user_satisfaction)
        
        # Update global stats
        self.global_stats["total_queries"] += 1
        alpha = 0.1
        self.global_stats["avg_latency"] = (
            (1 - alpha) * self.global_stats["avg_latency"] + alpha * actual_latency
        )
        self.global_stats["avg_confidence"] = (
            (1 - alpha) * self.global_stats["avg_confidence"] + alpha * actual_confidence
        )
    
    def _get_complexity_multiplier(self, complexity: str) -> float:
        """Get latency multiplier based on query complexity"""
        multipliers = {
            "simple": 0.8,
            "moderate": 1.0,
            "complex": 1.5
        }
        return multipliers.get(complexity, 1.0)
    
    def _estimate_baseline_performance(
        self, 
        library: LibraryType, 
        strategy: str, 
        query_mode: str
    ) -> dict[str, float]:
        """Estimate baseline performance for new configurations"""
        # Conservative estimates for new configurations
        base_estimates = {
            LibraryType.PAPERS: {"latency": 3.0, "confidence": 0.7, "success": 0.8},
            LibraryType.PUBLIC: {"latency": 2.5, "confidence": 0.75, "success": 0.85},
            LibraryType.MIXED: {"latency": 4.0, "confidence": 0.65, "success": 0.75}
        }
        
        strategy_adjustments = {
            "baseline": 1.0,
            "hyde": 1.3,
            "flare": 1.2,
            "multi_query": 1.5
        }
        
        base = base_estimates.get(library, base_estimates[LibraryType.PUBLIC])
        strategy_mult = strategy_adjustments.get(strategy, 1.0)
        
        return {
            "latency": base["latency"] * strategy_mult,
            "confidence": base["confidence"],
            "success_probability": base["success"],
            "user_satisfaction": 0.7,
            "data_confidence": 0.1  # Low confidence without historical data
        }
    
    def get_best_configurations(
        self, 
        query_classification: dict[str, Any], 
        max_latency: float = None,
        min_confidence: float = None
    ) -> list[tuple[str, dict[str, float]]]:
        """Get best performing configurations matching constraints"""
        candidates = []
        
        for profile_key, profile in self.performance_profiles.items():
            if profile.query_count < 5:  # Need minimum data for reliability
                continue
            
            library_val, strategy, query_mode = profile_key.split(":")
            library = LibraryType(library_val)
            
            predicted = self.predict_performance(library, strategy, query_mode, query_classification)
            
            # Apply constraints
            if max_latency and predicted["latency"] > max_latency:
                continue
            if min_confidence and predicted["confidence"] < min_confidence:
                continue
            
            # Calculate composite score
            score = (
                0.3 * (1.0 - predicted["latency"] / 10.0) +  # Latency (normalize to 10s max)
                0.3 * predicted["confidence"] +
                0.2 * predicted["success_probability"] +
                0.2 * predicted["user_satisfaction"]
            )
            
            candidates.append((profile_key, predicted, score))
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(key, pred) for key, pred, score in candidates[:5]]


class IntelligentQueryRouter:
    """
    Main intelligent query router that combines classification and performance optimization
    """
    
    def __init__(self, ollama_client: OllamaClient, settings):
        self.classifier = QueryClassifier(ollama_client)
        self.performance_oracle = PerformanceOracle(settings)
        self.settings = settings
        
        # Routing configuration
        self.routing_strategy = getattr(settings, "ROUTING_STRATEGY", RoutingStrategy.ADAPTIVE)
        self.enable_performance_learning = getattr(settings, "ENABLE_ROUTING_LEARNING", True)
        
        # Default routing rules
        self.default_routes = {
            "academic": {LibraryType.PAPERS: ["compare", "methodology", "general"]},
            "general": {LibraryType.PUBLIC: ["general", "datasets"]},
            "technical": {LibraryType.MIXED: ["methodology", "general"]},
        }
        
        logger.info(f"Initialized Intelligent Query Router with {self.routing_strategy.value} strategy")
    
    def route_query(
        self, 
        query: str, 
        context: QueryContext = None,
        constraints: dict[str, Any] = None
    ) -> RoutingDecision:
        """
        Intelligently route query to optimal configuration
        
        Args:
            query: User query to route
            context: Additional context for routing decisions
            constraints: Performance constraints (max_latency, min_confidence, etc.)
        
        Returns:
            RoutingDecision with optimal configuration
        """
        logger.info(f"Routing query: {query[:50]}...")
        
        # Classify query comprehensively
        classification = self.classifier.classify_query_comprehensive(query, context)
        
        # Apply routing strategy
        if self.routing_strategy == RoutingStrategy.PERFORMANCE_BASED:
            decision = self._performance_based_routing(classification, constraints or {})
        elif self.routing_strategy == RoutingStrategy.CONTENT_BASED:
            decision = self._content_based_routing(classification)
        elif self.routing_strategy == RoutingStrategy.ADAPTIVE:
            decision = self._adaptive_routing(classification, constraints or {}, context)
        else:
            decision = self._load_balanced_routing(classification)
        
        logger.info(f"Routing decision: {decision.library.value}:{decision.retrieval_strategy}:{decision.query_mode}")
        return decision
    
    def _performance_based_routing(
        self, 
        classification: dict[str, Any], 
        constraints: dict[str, Any]
    ) -> RoutingDecision:
        """Route based on historical performance data"""
        best_configs = self.performance_oracle.get_best_configurations(
            classification,
            max_latency=constraints.get("max_latency"),
            min_confidence=constraints.get("min_confidence")
        )
        
        if best_configs:
            best_key, predicted = best_configs[0]
            library_val, strategy, query_mode = best_key.split(":")
            library = LibraryType(library_val)
            
            return RoutingDecision(
                library=library,
                retrieval_strategy=strategy,
                query_mode=query_mode,
                top_k=self._determine_top_k(classification),
                additional_params={},
                confidence=predicted["data_confidence"],
                reasoning=f"Performance-based: best historical performance ({predicted['data_confidence']:.2f} confidence)",
                expected_performance=predicted
            )
        else:
            # Fallback to content-based routing
            return self._content_based_routing(classification)
    
    def _content_based_routing(self, classification: dict[str, Any]) -> RoutingDecision:
        """Route based on query content analysis"""
        domain = classification["domain"]
        intent = classification["intent"]
        complexity = classification["complexity"]
        
        # Determine library
        library = classification["expected_library"]
        
        # Determine strategy based on complexity and intent
        if complexity == "complex" or intent in ["analytical", "comparative"]:
            if classification.get("requires_multi_step", False):
                strategy = "multi_query"
            else:
                strategy = "flare"
        elif intent in ["factual", "informational"]:
            strategy = "baseline"
        else:
            strategy = "hyde"
        
        # Determine query mode
        if intent == "comparative":
            query_mode = "compare"
        elif intent in ["analytical", "procedural"]:
            query_mode = "methodology"
        elif domain == "academic" and "datasets" in classification.get("primary_concepts", []):
            query_mode = "datasets"
        else:
            query_mode = "general"
        
        return RoutingDecision(
            library=library,
            retrieval_strategy=strategy,
            query_mode=query_mode,
            top_k=self._determine_top_k(classification),
            additional_params={},
            confidence=classification.get("confidence", 0.7),
            reasoning=f"Content-based: {domain}/{intent}/{complexity} → {library.value}:{strategy}:{query_mode}",
            expected_performance=self.performance_oracle.predict_performance(
                library, strategy, query_mode, classification
            )
        )
    
    def _adaptive_routing(
        self, 
        classification: dict[str, Any], 
        constraints: dict[str, Any],
        context: QueryContext
    ) -> RoutingDecision:
        """Adaptive routing combining multiple factors"""
        # Get content-based recommendation
        content_decision = self._content_based_routing(classification)
        
        # Get performance-based alternatives
        best_configs = self.performance_oracle.get_best_configurations(
            classification,
            max_latency=constraints.get("max_latency"),
            min_confidence=constraints.get("min_confidence")
        )
        
        # Decide based on confidence and constraints
        if best_configs and best_configs[0][1]["data_confidence"] > 0.7:
            # High confidence in performance data, use performance-based
            return self._performance_based_routing(classification, constraints)
        
        # Consider urgency and constraints
        urgency = classification.get("urgency", "normal")
        if urgency == "high" or constraints.get("max_latency", float("inf")) < 3.0:
            # Under time pressure, prefer faster strategies
            content_decision.retrieval_strategy = "baseline"
            content_decision.reasoning += " (optimized for speed)"
        
        # Consider user preferences from context
        if context and context.user_preferences:
            quality_pref = context.user_preferences.get("quality_preference", "balanced")
            if quality_pref == "high":
                if content_decision.retrieval_strategy == "baseline":
                    content_decision.retrieval_strategy = "flare"
                content_decision.top_k = min(12, content_decision.top_k + 2)
                content_decision.reasoning += " (optimized for quality)"
        
        content_decision.reasoning = f"Adaptive: {content_decision.reasoning}"
        return content_decision
    
    def _load_balanced_routing(self, classification: dict[str, Any]) -> RoutingDecision:
        """Route to balance load across resources"""
        # Simple load balancing - in production would consider actual system load
        libraries = [LibraryType.PAPERS, LibraryType.PUBLIC]
        strategies = ["baseline", "flare", "hyde"]
        
        # Rotate through combinations
        import random
        library = random.choice(libraries)
        strategy = random.choice(strategies)
        
        # Still respect content appropriateness
        if classification["domain"] == "academic":
            library = LibraryType.PAPERS
        elif classification["domain"] == "general":
            library = LibraryType.PUBLIC
        
        return RoutingDecision(
            library=library,
            retrieval_strategy=strategy,
            query_mode="general",
            top_k=6,
            additional_params={},
            confidence=0.5,
            reasoning="Load-balanced routing",
            expected_performance={"latency": 3.0, "confidence": 0.7}
        )
    
    def _determine_top_k(self, classification: dict[str, Any]) -> int:
        """Determine optimal top_k based on query characteristics"""
        base_k = 6
        
        complexity = classification.get("complexity", "moderate")
        intent = classification.get("intent", "informational")
        scope = classification.get("detailed_analysis", {}).get("scope", "narrow")
        
        # Adjust based on characteristics
        if complexity == "complex":
            base_k += 2
        if intent in ["comparative", "analytical"]:
            base_k += 2
        if scope == "broad":
            base_k += 1
        
        return min(12, max(3, base_k))
    
    def record_performance(
        self,
        decision: RoutingDecision,
        actual_response: AnswerResponse,
        user_feedback: dict[str, Any] = None
    ) -> None:
        """Record actual performance for learning"""
        if not self.enable_performance_learning:
            return
        
        # Extract performance metrics
        latency = getattr(actual_response, "total_latency", 0.0)
        confidence = actual_response.confidence
        success = actual_response.has_answer and len(actual_response.sources) > 0
        satisfaction = user_feedback.get("satisfaction", None) if user_feedback else None
        
        # Update performance oracle
        self.performance_oracle.update_performance(
            decision.library,
            decision.retrieval_strategy,
            decision.query_mode,
            latency,
            confidence,
            success,
            satisfaction
        )
        
        logger.info(f"Recorded performance: {latency:.2f}s, {confidence:.2f} confidence")
    
    def get_routing_analytics(self) -> dict[str, Any]:
        """Get routing performance analytics"""
        return {
            "routing_strategy": self.routing_strategy.value,
            "performance_learning_enabled": self.enable_performance_learning,
            "total_configurations": len(self.performance_oracle.performance_profiles),
            "global_stats": self.performance_oracle.global_stats,
            "top_configurations": [
                {
                    "config": key,
                    "performance": {
                        "avg_latency": profile.avg_latency,
                        "avg_confidence": profile.avg_confidence,
                        "success_rate": profile.success_rate,
                        "query_count": profile.query_count
                    }
                }
                for key, profile in sorted(
                    self.performance_oracle.performance_profiles.items(),
                    key=lambda x: x[1].query_count,
                    reverse=True
                )[:10]
            ]
        }
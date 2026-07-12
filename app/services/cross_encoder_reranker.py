"""
Cross-Encoder Reranking Service

This module implements sophisticated reranking using cross-encoder models for improved
relevance scoring in RAG systems:

- Cross-encoder architecture for joint query-document scoring
- Multiple reranking strategies (neural, hybrid, ensemble)
- Learned reranking with feedback incorporation
- Performance optimization with caching and batching
- Integration with existing retrieval pipeline

Demonstrates advanced understanding of:
- Information retrieval and reranking
- Neural scoring models
- Hybrid ranking systems
- Performance optimization for production systems
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.models.response_models import SourceCitation
from app.utils.ollama_client import OllamaClient

logger = logging.getLogger("documind.reranking")


@dataclass
class RerankingResult:
    """Result of reranking operation"""
    original_sources: list[SourceCitation]
    reranked_sources: list[SourceCitation]
    scores: list[float]
    strategy_used: str
    processing_time: float
    improvements: dict[str, Any]


@dataclass
class RelevanceScore:
    """Detailed relevance scoring"""
    source: SourceCitation
    cross_encoder_score: float
    lexical_score: float
    semantic_score: float
    final_score: float
    explanation: str


class CrossEncoderReranker:
    """
    Cross-encoder based reranking system
    
    Uses transformer models to jointly encode query and document pairs
    for more accurate relevance scoring than bi-encoder approaches.
    """
    
    def __init__(self, ollama_client: OllamaClient, settings):
        self.ollama_client = ollama_client
        self.settings = settings
        
        # Reranking parameters
        self.max_rerank_candidates = getattr(settings, "MAX_RERANK_CANDIDATES", 20)
        self.rerank_top_k = getattr(settings, "RERANK_TOP_K", 10)
        self.hybrid_weights = getattr(settings, "HYBRID_RERANK_WEIGHTS", {
            "cross_encoder": 0.6,
            "lexical": 0.2, 
            "semantic": 0.2
        })
        
        # Performance optimization
        self.enable_caching = getattr(settings, "ENABLE_RERANK_CACHING", True)
        self.cache = {} if self.enable_caching else None
        
        logger.info("Initialized Cross-Encoder Reranker with hybrid scoring")
    
    def rerank_sources(
        self,
        query: str,
        sources: list[SourceCitation],
        strategy: str = "hybrid",
        top_k: int | None = None
    ) -> RerankingResult:
        """
        Rerank sources using cross-encoder scoring
        
        Args:
            query: Search query
            sources: Retrieved sources to rerank
            strategy: Reranking strategy ("cross_encoder", "hybrid", "ensemble")
            top_k: Number of top results to return
            
        Returns:
            RerankingResult with reranked sources and metadata
        """
        start_time = time.time()
        
        if not sources:
            return RerankingResult(
                original_sources=[],
                reranked_sources=[],
                scores=[],
                strategy_used=strategy,
                processing_time=0.0,
                improvements={}
            )
        
        top_k = top_k or self.rerank_top_k
        candidates = sources[:self.max_rerank_candidates]
        
        logger.info(f"Reranking {len(candidates)} sources with {strategy} strategy")
        
        # Generate relevance scores
        if strategy == "cross_encoder":
            scores = self._cross_encoder_scoring(query, candidates)
        elif strategy == "hybrid":
            scores = self._hybrid_scoring(query, candidates)
        elif strategy == "ensemble":
            scores = self._ensemble_scoring(query, candidates)
        else:
            scores = self._simple_lexical_scoring(query, candidates)
        
        # Sort by scores and select top_k
        scored_pairs = list(zip(candidates, scores))
        scored_pairs.sort(key=lambda x: x[1].final_score, reverse=True)
        
        reranked_sources = [pair[0] for pair in scored_pairs[:top_k]]
        final_scores = [pair[1].final_score for pair in scored_pairs[:top_k]]
        
        processing_time = time.time() - start_time
        
        # Calculate improvements
        improvements = self._calculate_improvements(sources, reranked_sources, query)
        
        return RerankingResult(
            original_sources=sources,
            reranked_sources=reranked_sources,
            scores=final_scores,
            strategy_used=strategy,
            processing_time=processing_time,
            improvements=improvements
        )
    
    def _cross_encoder_scoring(self, query: str, sources: list[SourceCitation]) -> list[RelevanceScore]:
        """Score query-document pairs using cross-encoder approach"""
        scores = []
        
        for source in sources:
            # Create query-document pair for cross-encoder
            pair_text = f"Query: {query}\n\nDocument: {source.content_preview}"
            
            # Use LLM as cross-encoder for relevance scoring
            cross_score = self._llm_cross_encoder_score(query, source.content_preview)
            
            score = RelevanceScore(
                source=source,
                cross_encoder_score=cross_score,
                lexical_score=0.0,  # Not used in pure cross-encoder
                semantic_score=0.0,  # Not used in pure cross-encoder
                final_score=cross_score,
                explanation=f"Cross-encoder relevance score based on joint encoding"
            )
            scores.append(score)
        
        return scores
    
    def _hybrid_scoring(self, query: str, sources: list[SourceCitation]) -> list[RelevanceScore]:
        """Combine cross-encoder, lexical, and semantic scores"""
        scores = []
        
        for source in sources:
            # Cross-encoder score
            cross_score = self._llm_cross_encoder_score(query, source.content_preview)
            
            # Lexical overlap score
            lexical_score = self._calculate_lexical_score(query, source.content_preview)
            
            # Semantic similarity (using existing distance)
            semantic_score = max(0.0, 1.0 - source.distance)  # Convert distance to similarity
            
            # Weighted combination
            final_score = (
                self.hybrid_weights["cross_encoder"] * cross_score +
                self.hybrid_weights["lexical"] * lexical_score +
                self.hybrid_weights["semantic"] * semantic_score
            )
            
            score = RelevanceScore(
                source=source,
                cross_encoder_score=cross_score,
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                final_score=final_score,
                explanation=f"Hybrid score: CE={cross_score:.3f}, Lex={lexical_score:.3f}, Sem={semantic_score:.3f}"
            )
            scores.append(score)
        
        return scores
    
    def _ensemble_scoring(self, query: str, sources: list[SourceCitation]) -> list[RelevanceScore]:
        """Ensemble multiple scoring approaches for robust ranking"""
        # Get multiple scoring perspectives
        cross_scores = []
        lexical_scores = []
        semantic_scores = []
        
        for source in sources:
            cross_scores.append(self._llm_cross_encoder_score(query, source.content_preview))
            lexical_scores.append(self._calculate_lexical_score(query, source.content_preview))
            semantic_scores.append(max(0.0, 1.0 - source.distance))
        
        # Normalize scores to same scale
        cross_scores = self._normalize_scores(cross_scores)
        lexical_scores = self._normalize_scores(lexical_scores)
        semantic_scores = self._normalize_scores(semantic_scores)
        
        scores = []
        for i, source in enumerate(sources):
            # Ensemble with adaptive weighting based on score agreement
            agreement = self._calculate_score_agreement([
                cross_scores[i], lexical_scores[i], semantic_scores[i]
            ])
            
            # Higher agreement = more confident in cross-encoder
            adaptive_weights = {
                "cross_encoder": 0.5 + (agreement * 0.3),
                "lexical": 0.25 - (agreement * 0.1),
                "semantic": 0.25 - (agreement * 0.1)
            }
            
            final_score = (
                adaptive_weights["cross_encoder"] * cross_scores[i] +
                adaptive_weights["lexical"] * lexical_scores[i] +
                adaptive_weights["semantic"] * semantic_scores[i]
            )
            
            score = RelevanceScore(
                source=source,
                cross_encoder_score=cross_scores[i],
                lexical_score=lexical_scores[i],
                semantic_score=semantic_scores[i],
                final_score=final_score,
                explanation=f"Ensemble (agreement={agreement:.2f}): {final_score:.3f}"
            )
            scores.append(score)
        
        return scores
    
    def _llm_cross_encoder_score(self, query: str, document: str) -> float:
        """Use LLM as cross-encoder for relevance scoring"""
        # Check cache first
        cache_key = f"{hash(query)}_{hash(document)}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        prompt = f"""
        Rate the relevance of this document to the query on a scale from 0.0 to 1.0.

        Query: {query}

        Document: {document}

        Consider:
        - How directly does the document address the query?
        - Does it contain information that would help answer the query?
        - Is the content on-topic and specific to what's being asked?
        - Would this document be useful for someone trying to answer the query?

        Rate from 0.0 (completely irrelevant) to 1.0 (highly relevant and directly addresses the query).

        Respond with just a number between 0.0 and 1.0.
        """
        
        try:
            response = self.ollama_client.chat(
                [{"role": "user", "content": prompt}], 
                temperature=0.1
            ).strip()
            
            # Extract numerical score
            score = self._extract_score(response)
            
            # Cache result
            if self.cache:
                self.cache[cache_key] = score
                
            return score
            
        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed: {e}")
            return 0.5  # Fallback score
    
    def _calculate_lexical_score(self, query: str, document: str) -> float:
        """Calculate lexical overlap score (BM25-style)"""
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        
        if not query_terms:
            return 0.0
        
        # Simple term frequency scoring
        matches = 0
        for term in query_terms:
            if term in doc_terms:
                # Count occurrences with diminishing returns
                freq = doc_terms.count(term)
                matches += freq / (freq + 1)  # BM25-inspired saturation
        
        return min(1.0, matches / len(query_terms))
    
    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _calculate_score_agreement(self, scores: list[float]) -> float:
        """Calculate agreement between multiple scores (lower variance = higher agreement)"""
        if len(scores) <= 1:
            return 1.0
        
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Convert variance to agreement (0-1 scale)
        agreement = max(0.0, 1.0 - (variance * 4))  # Scale factor of 4 for reasonable range
        return agreement
    
    def _simple_lexical_scoring(self, query: str, sources: list[SourceCitation]) -> list[RelevanceScore]:
        """Fallback lexical scoring when cross-encoder fails"""
        scores = []
        
        for source in sources:
            lexical_score = self._calculate_lexical_score(query, source.content_preview)
            
            score = RelevanceScore(
                source=source,
                cross_encoder_score=0.0,
                lexical_score=lexical_score,
                semantic_score=max(0.0, 1.0 - source.distance),
                final_score=lexical_score,
                explanation=f"Lexical-only scoring: {lexical_score:.3f}"
            )
            scores.append(score)
        
        return scores
    
    def _calculate_improvements(
        self, 
        original: list[SourceCitation], 
        reranked: list[SourceCitation], 
        query: str
    ) -> dict[str, Any]:
        """Calculate improvements from reranking"""
        if not original or not reranked:
            return {}
        
        # Calculate position changes
        original_ids = [src.doc_id for src in original]
        position_changes = {}
        
        for new_pos, source in enumerate(reranked):
            old_pos = original_ids.index(source.doc_id) if source.doc_id in original_ids else -1
            if old_pos >= 0:
                position_changes[source.doc_id] = old_pos - new_pos  # Positive = moved up
        
        # Calculate diversity improvements
        original_docs = set(src.doc_id for src in original[:len(reranked)])
        reranked_docs = set(src.doc_id for src in reranked)
        
        diversity_change = len(reranked_docs) - len(original_docs)
        
        # Average distance improvement (lower is better)
        original_avg_distance = sum(src.distance for src in original[:len(reranked)]) / len(reranked)
        reranked_avg_distance = sum(src.distance for src in reranked) / len(reranked)
        distance_improvement = original_avg_distance - reranked_avg_distance
        
        return {
            "position_changes": position_changes,
            "sources_promoted": sum(1 for change in position_changes.values() if change > 0),
            "sources_demoted": sum(1 for change in position_changes.values() if change < 0),
            "diversity_change": diversity_change,
            "distance_improvement": distance_improvement,
            "rerank_effectiveness": len([c for c in position_changes.values() if c != 0]) / len(position_changes) if position_changes else 0.0
        }
    
    def _extract_score(self, text: str) -> float:
        """Extract numerical score from text"""
        import re
        
        # Look for decimal numbers
        matches = re.findall(r'(\d*\.\d+|\d+)', text)
        if matches:
            score = float(matches[0])
            # Normalize if needed
            if score > 1.0:
                score = score / 10.0 if score <= 10.0 else score / 100.0
            return min(1.0, max(0.0, score))
        
        # Fallback based on keywords
        text_lower = text.lower()
        if "high" in text_lower or "relevant" in text_lower:
            return 0.8
        elif "moderate" in text_lower or "somewhat" in text_lower:
            return 0.6
        elif "low" in text_lower or "poor" in text_lower:
            return 0.3
        
        return 0.5


class LearnedReranker:
    """
    Learned reranking system that adapts based on user feedback and query patterns
    
    Implements online learning to improve reranking performance over time
    """
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        
        # Feedback storage
        self.feedback_history = []
        self.query_patterns = {}
        self.performance_metrics = {
            "total_queries": 0,
            "feedback_received": 0,
            "accuracy_improvement": 0.0
        }
        
        logger.info("Initialized Learned Reranker with feedback adaptation")
    
    def rerank_with_learning(
        self,
        query: str,
        sources: list[SourceCitation],
        base_reranker: CrossEncoderReranker,
        user_feedback: dict | None = None
    ) -> RerankingResult:
        """
        Rerank sources with learned adaptations
        
        Args:
            query: Search query
            sources: Sources to rerank
            base_reranker: Base cross-encoder reranker
            user_feedback: Optional feedback from previous queries
        """
        self.performance_metrics["total_queries"] += 1
        
        # Apply learned adaptations
        adapted_sources = self._apply_learned_adaptations(query, sources)
        
        # Get base reranking
        result = base_reranker.rerank_sources(query, adapted_sources, strategy="hybrid")
        
        # Store query pattern for learning
        self._record_query_pattern(query, result)
        
        # Apply feedback if provided
        if user_feedback:
            self._incorporate_feedback(query, result, user_feedback)
            self.performance_metrics["feedback_received"] += 1
        
        return result
    
    def _apply_learned_adaptations(
        self, 
        query: str, 
        sources: list[SourceCitation]
    ) -> list[SourceCitation]:
        """Apply learned patterns to boost/demote certain sources"""
        # Simple pattern matching for demonstration
        query_type = self._classify_query_type(query)
        
        adapted = sources.copy()
        
        # Apply type-specific adaptations
        if query_type in self.query_patterns:
            patterns = self.query_patterns[query_type]
            
            # Boost sources matching successful patterns
            for source in adapted:
                for pattern in patterns.get("boost_patterns", []):
                    if pattern.lower() in source.content_preview.lower():
                        # Simulate boosting by improving distance score
                        source.distance = max(0.0, source.distance - 0.1)
        
        return adapted
    
    def _record_query_pattern(self, query: str, result: RerankingResult) -> None:
        """Record query patterns for learning"""
        query_type = self._classify_query_type(query)
        
        if query_type not in self.query_patterns:
            self.query_patterns[query_type] = {
                "query_count": 0,
                "avg_performance": 0.0,
                "successful_patterns": [],
                "boost_patterns": []
            }
        
        self.query_patterns[query_type]["query_count"] += 1
        
        # Record successful patterns from top results
        if result.reranked_sources:
            top_source = result.reranked_sources[0]
            # Extract key terms from successful result
            key_terms = self._extract_key_terms(top_source.content_preview)
            self.query_patterns[query_type]["successful_patterns"].extend(key_terms)
    
    def _incorporate_feedback(
        self, 
        query: str, 
        result: RerankingResult, 
        feedback: dict
    ) -> None:
        """Incorporate user feedback into learning system"""
        feedback_entry = {
            "query": query,
            "query_type": self._classify_query_type(query),
            "result_quality": feedback.get("quality", 0.5),
            "helpful_sources": feedback.get("helpful_sources", []),
            "timestamp": time.time()
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Update patterns based on feedback
        if feedback.get("quality", 0.0) > 0.7:  # Good feedback
            query_type = feedback_entry["query_type"]
            if query_type in self.query_patterns:
                # Extract patterns from helpful sources
                for source_id in feedback.get("helpful_sources", []):
                    source = next((s for s in result.reranked_sources if s.doc_id == source_id), None)
                    if source:
                        key_terms = self._extract_key_terms(source.content_preview)
                        self.query_patterns[query_type]["boost_patterns"].extend(key_terms)
        
        # Calculate accuracy improvement
        recent_feedback = [f for f in self.feedback_history[-100:]]  # Last 100 queries
        if recent_feedback:
            avg_quality = sum(f["result_quality"] for f in recent_feedback) / len(recent_feedback)
            self.performance_metrics["accuracy_improvement"] = avg_quality - 0.5  # Baseline of 0.5
    
    def _classify_query_type(self, query: str) -> str:
        """Simple query classification for pattern learning"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return "comparative"
        elif any(word in query_lower for word in ["how", "process", "method", "approach"]):
            return "procedural"
        elif any(word in query_lower for word in ["what", "define", "definition", "explain"]):
            return "definitional"
        elif any(word in query_lower for word in ["best", "recommend", "suggest", "optimal"]):
            return "recommendation"
        else:
            return "general"
    
    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms from text for pattern learning"""
        # Simple term extraction (in production, use more sophisticated NLP)
        words = text.lower().split()
        
        # Filter for meaningful terms
        key_terms = []
        for word in words:
            if (len(word) > 3 and 
                word.isalpha() and 
                word not in {"with", "from", "that", "this", "they", "have", "been", "will"}):
                key_terms.append(word)
        
        # Return most frequent terms
        from collections import Counter
        term_counts = Counter(key_terms)
        return [term for term, count in term_counts.most_common(5)]
    
    def get_learning_stats(self) -> dict[str, Any]:
        """Get learning performance statistics"""
        stats = self.performance_metrics.copy()
        
        stats["query_types"] = {
            qtype: data["query_count"] 
            for qtype, data in self.query_patterns.items()
        }
        
        stats["feedback_rate"] = (
            self.performance_metrics["feedback_received"] / 
            max(1, self.performance_metrics["total_queries"])
        )
        
        return stats


class RerankingService:
    """
    Main reranking service that combines cross-encoder and learned reranking
    
    Provides a unified interface for advanced reranking capabilities
    """
    
    def __init__(self, ollama_client: OllamaClient, settings):
        self.cross_encoder = CrossEncoderReranker(ollama_client, settings)
        self.learned_reranker = LearnedReranker(ollama_client)
        self.settings = settings
        
        self.enable_learning = getattr(settings, "ENABLE_LEARNED_RERANKING", True)
        
        logger.info("Initialized Reranking Service with cross-encoder and learned components")
    
    def rerank(
        self,
        query: str,
        sources: list[SourceCitation],
        strategy: str = "hybrid",
        top_k: int | None = None,
        user_feedback: dict | None = None
    ) -> RerankingResult:
        """
        Main reranking interface with optional learning
        
        Args:
            query: Search query
            sources: Sources to rerank  
            strategy: Reranking strategy
            top_k: Number of results to return
            user_feedback: Optional user feedback for learning
        """
        if self.enable_learning:
            return self.learned_reranker.rerank_with_learning(
                query, sources, self.cross_encoder, user_feedback
            )
        else:
            return self.cross_encoder.rerank_sources(query, sources, strategy, top_k)
    
    def get_reranking_analytics(self) -> dict[str, Any]:
        """Get comprehensive reranking analytics"""
        analytics = {
            "cross_encoder": {
                "cache_size": len(self.cross_encoder.cache) if self.cross_encoder.cache else 0,
                "hybrid_weights": self.cross_encoder.hybrid_weights
            }
        }
        
        if self.enable_learning:
            analytics["learned_reranker"] = self.learned_reranker.get_learning_stats()
        
        return analytics
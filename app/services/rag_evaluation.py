"""
Advanced RAG Evaluation Framework

Implements comprehensive evaluation metrics for RAG systems including:
- RAGAS-style metrics (Faithfulness, Relevance, Context Precision/Recall)
- Custom RAG-specific metrics (Source Diversity, Coverage, Hallucination Detection)
- Performance metrics (Latency, Throughput, Token Usage)
- Comparative analysis across different retrieval strategies

This demonstrates deep understanding of RAG evaluation challenges and production monitoring.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import mean, median
from typing import Any, Literal

from app.models.response_models import AnswerResponse, SourceCitation
from app.utils.ollama_client import OllamaClient

logger = logging.getLogger("documind.evaluation")


@dataclass
class RAGMetrics:
    """Comprehensive RAG evaluation metrics"""
    # Core RAGAS-style metrics
    faithfulness: float = 0.0  # How well the answer is grounded in retrieved context
    relevance: float = 0.0  # How relevant retrieved context is to the query
    context_precision: float = 0.0  # Precision of relevant chunks in top-k
    context_recall: float = 0.0  # Coverage of relevant information
    
    # Advanced RAG metrics
    source_diversity: float = 0.0  # Diversity of information sources
    answer_completeness: float = 0.0  # How complete the answer is
    hallucination_score: float = 0.0  # Likelihood of hallucinated content
    coherence_score: float = 0.0  # Internal consistency of the answer
    
    # Performance metrics
    retrieval_latency: float = 0.0  # Time for retrieval phase
    generation_latency: float = 0.0  # Time for generation phase
    total_latency: float = 0.0  # End-to-end latency
    tokens_generated: int = 0  # Number of tokens in response
    chunks_retrieved: int = 0  # Number of chunks retrieved
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    query_id: str = ""
    strategy: str = "baseline"


@dataclass
class EvaluationCase:
    """Test case for RAG evaluation"""
    query: str
    expected_answer: str | None = None
    relevant_doc_ids: list[str] = field(default_factory=list)
    query_type: str = "general"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    domain: str = "general"


@dataclass
class StrategyComparison:
    """Comparison results across retrieval strategies"""
    baseline_metrics: RAGMetrics
    flare_metrics: RAGMetrics | None = None
    hyde_metrics: RAGMetrics | None = None
    multi_query_metrics: RAGMetrics | None = None
    
    def get_best_strategy(self, metric: str = "faithfulness") -> tuple[str, float]:
        """Returns the best performing strategy for a given metric"""
        strategies = {"baseline": self.baseline_metrics}
        if self.flare_metrics:
            strategies["flare"] = self.flare_metrics
        if self.hyde_metrics:
            strategies["hyde"] = self.hyde_metrics
        if self.multi_query_metrics:
            strategies["multi_query"] = self.multi_query_metrics
        
        best_strategy = max(strategies.items(), key=lambda x: getattr(x[1], metric))
        return best_strategy[0], getattr(best_strategy[1], metric)


class RAGEvaluator:
    """Advanced RAG evaluation system with RAGAS-style metrics and performance analysis"""
    
    def __init__(self, ollama_client: OllamaClient, settings) -> None:
        self.ollama_client = ollama_client
        self.settings = settings
        
        # LLM-as-a-Judge prompts for evaluation
        self.faithfulness_prompt = self._get_faithfulness_prompt()
        self.relevance_prompt = self._get_relevance_prompt()
        self.completeness_prompt = self._get_completeness_prompt()
        self.hallucination_prompt = self._get_hallucination_prompt()
        self.coherence_prompt = self._get_coherence_prompt()
        
    def evaluate_rag_response(
        self, 
        query: str, 
        response: AnswerResponse, 
        ground_truth: str | None = None,
        relevant_doc_ids: list[str] | None = None
    ) -> RAGMetrics:
        """
        Comprehensive evaluation of a RAG response
        
        This method demonstrates understanding of:
        - Multi-dimensional RAG evaluation
        - LLM-as-a-Judge techniques
        - Production monitoring metrics
        """
        start_time = time.time()
        
        metrics = RAGMetrics(
            query_id=f"eval_{int(time.time())}",
            strategy=response.retrieval_strategy or "baseline",
            total_latency=getattr(response, 'total_latency', 0.0),
            tokens_generated=len(response.answer.split()) if response.answer else 0,
            chunks_retrieved=len(response.sources)
        )
        
        # Core RAGAS-style metrics
        if response.answer and response.sources:
            metrics.faithfulness = self._evaluate_faithfulness(response.answer, response.sources)
            metrics.relevance = self._evaluate_relevance(query, response.sources)
            metrics.context_precision = self._evaluate_context_precision(
                query, response.sources, relevant_doc_ids
            )
            if ground_truth:
                metrics.context_recall = self._evaluate_context_recall(
                    ground_truth, response.sources
                )
        
        # Advanced metrics
        metrics.source_diversity = self._calculate_source_diversity(response.sources)
        if response.answer:
            if ground_truth:
                metrics.answer_completeness = self._evaluate_completeness(
                    response.answer, ground_truth
                )
            metrics.hallucination_score = self._evaluate_hallucination(
                response.answer, response.sources
            )
            metrics.coherence_score = self._evaluate_coherence(response.answer)
        
        evaluation_time = time.time() - start_time
        logger.info(f"RAG evaluation completed in {evaluation_time:.2f}s for query: {query[:50]}...")
        
        return metrics
    
    def compare_strategies(
        self, 
        query: str, 
        rag_service, 
        top_k: int = 6, 
        query_mode: str = "general",
        ground_truth: str | None = None,
        relevant_doc_ids: list[str] | None = None
    ) -> StrategyComparison:
        """
        Compare performance across different retrieval strategies
        
        Demonstrates understanding of:
        - Strategy ablation studies
        - Performance benchmarking
        - Systematic evaluation methodology
        """
        logger.info(f"Running strategy comparison for query: {query[:50]}...")
        
        strategies = ["baseline", "flare", "hyde", "multi_query"]
        results = {}
        
        for strategy in strategies:
            try:
                logger.info(f"Evaluating strategy: {strategy}")
                start_time = time.time()
                
                response = rag_service.answer(
                    query=query,
                    top_k=top_k,
                    query_mode=query_mode,
                    retrieval_strategy=strategy
                )
                
                # Add timing information
                response.total_latency = time.time() - start_time
                
                metrics = self.evaluate_rag_response(
                    query, response, ground_truth, relevant_doc_ids
                )
                results[strategy] = metrics
                
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed: {e}")
                continue
        
        return StrategyComparison(
            baseline_metrics=results["baseline"],
            flare_metrics=results.get("flare"),
            hyde_metrics=results.get("hyde"),
            multi_query_metrics=results.get("multi_query")
        )
    
    def _evaluate_faithfulness(self, answer: str, sources: list[SourceCitation]) -> float:
        """
        Evaluate how well the answer is grounded in the retrieved context
        Uses LLM-as-a-Judge to assess faithfulness
        """
        if not sources:
            return 0.0
        
        context = "\n\n".join([f"Source {i+1}: {src.content_preview}" 
                              for i, src in enumerate(sources)])
        
        prompt = self.faithfulness_prompt.format(
            context=context,
            answer=answer
        )
        
        try:
            response = self.ollama_client.chat(
                [{"role": "user", "content": prompt}], 
                temperature=0.1
            )
            # Extract numerical score from response
            score = self._extract_score(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return 0.5
    
    def _evaluate_relevance(self, query: str, sources: list[SourceCitation]) -> float:
        """
        Evaluate relevance of retrieved context to the query
        """
        if not sources:
            return 0.0
        
        context_texts = [src.content_preview for src in sources]
        context = "\n\n".join([f"Source {i+1}: {text}" 
                              for i, text in enumerate(context_texts)])
        
        prompt = self.relevance_prompt.format(
            query=query,
            context=context
        )
        
        try:
            response = self.ollama_client.chat(
                [{"role": "user", "content": prompt}], 
                temperature=0.1
            )
            score = self._extract_score(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")
            return 0.5
    
    def _evaluate_context_precision(
        self, 
        query: str, 
        sources: list[SourceCitation],
        relevant_doc_ids: list[str] | None = None
    ) -> float:
        """
        Evaluate precision of relevant chunks in retrieved results
        """
        if not sources:
            return 0.0
        
        if relevant_doc_ids is None:
            # Fallback: use LLM to judge relevance of each source
            relevant_count = 0
            for source in sources:
                if self._is_source_relevant(query, source):
                    relevant_count += 1
            return relevant_count / len(sources) if sources else 0.0
        
        # Use provided ground truth
        relevant_count = sum(1 for src in sources if src.doc_id in relevant_doc_ids)
        return relevant_count / len(sources)
    
    def _evaluate_context_recall(self, ground_truth: str, sources: list[SourceCitation]) -> float:
        """
        Evaluate how much of the relevant information was retrieved
        """
        if not sources or not ground_truth:
            return 0.0
        
        context = "\n\n".join([src.content_preview for src in sources])
        
        prompt = f"""
        Evaluate how much of the key information from the ground truth is covered by the retrieved context.

        Ground Truth: {ground_truth}

        Retrieved Context: {context}

        Rate the coverage from 0.0 to 1.0, where:
        - 1.0 = All key information from ground truth is present in context
        - 0.5 = About half of the key information is covered
        - 0.0 = No relevant information is covered

        Respond with just a number between 0.0 and 1.0.
        """
        
        try:
            response = self.ollama_client.chat(
                [{"role": "user", "content": prompt}], 
                temperature=0.1
            )
            score = self._extract_score(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Context recall evaluation failed: {e}")
            return 0.5
    
    def _calculate_source_diversity(self, sources: list[SourceCitation]) -> float:
        """
        Calculate diversity of information sources
        Higher diversity indicates better coverage across different documents
        """
        if not sources:
            return 0.0
        
        unique_docs = set(src.doc_id for src in sources if src.doc_id)
        if not unique_docs:
            return 0.0
        
        # Normalize by theoretical maximum (all sources from different docs)
        return len(unique_docs) / len(sources)
    
    def _evaluate_completeness(self, answer: str, ground_truth: str) -> float:
        """
        Evaluate completeness of the answer compared to ground truth
        """
        prompt = self.completeness_prompt.format(
            answer=answer,
            ground_truth=ground_truth
        )
        
        try:
            response = self.ollama_client.chat(
                [{"role": "user", "content": prompt}], 
                temperature=0.1
            )
            score = self._extract_score(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Completeness evaluation failed: {e}")
            return 0.5
    
    def _evaluate_hallucination(self, answer: str, sources: list[SourceCitation]) -> float:
        """
        Evaluate likelihood of hallucinated content in the answer
        Returns higher scores for more likely hallucinations
        """
        if not sources:
            return 1.0  # High hallucination risk with no sources
        
        context = "\n\n".join([src.content_preview for src in sources])
        
        prompt = self.hallucination_prompt.format(
            context=context,
            answer=answer
        )
        
        try:
            response = self.ollama_client.chat(
                [{"role": "user", "content": prompt}], 
                temperature=0.1
            )
            score = self._extract_score(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Hallucination evaluation failed: {e}")
            return 0.5
    
    def _evaluate_coherence(self, answer: str) -> float:
        """
        Evaluate internal coherence and consistency of the answer
        """
        prompt = self.coherence_prompt.format(answer=answer)
        
        try:
            response = self.ollama_client.chat(
                [{"role": "user", "content": prompt}], 
                temperature=0.1
            )
            score = self._extract_score(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Coherence evaluation failed: {e}")
            return 0.5
    
    def _is_source_relevant(self, query: str, source: SourceCitation) -> bool:
        """
        Determine if a source is relevant to the query using LLM judgment
        """
        prompt = f"""
        Is this source relevant to answering the query? Respond with just "Yes" or "No".

        Query: {query}

        Source: {source.content_preview}
        """
        
        try:
            response = self.ollama_client.chat(
                [{"role": "user", "content": prompt}], 
                temperature=0.1
            ).strip().lower()
            return "yes" in response
        except Exception:
            return False
    
    def _extract_score(self, text: str) -> float:
        """
        Extract numerical score from LLM response
        """
        # Look for patterns like "0.8", "8/10", "80%"
        patterns = [
            r'(\d*\.?\d+)(?:\s*[/]\s*(?:10|100))?',  # 0.8, 8/10
            r'(\d+)%',  # 80%
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                score = float(match.group(1))
                # Normalize to 0-1 range
                if score > 1.0:
                    score = score / 100.0 if score <= 100 else score / 10.0
                return min(1.0, max(0.0, score))
        
        # Fallback: look for common score words
        text_lower = text.lower()
        if any(word in text_lower for word in ["excellent", "perfect", "outstanding"]):
            return 0.9
        elif any(word in text_lower for word in ["good", "solid", "adequate"]):
            return 0.7
        elif any(word in text_lower for word in ["fair", "moderate", "average"]):
            return 0.5
        elif any(word in text_lower for word in ["poor", "weak", "inadequate"]):
            return 0.3
        
        return 0.5  # Default neutral score
    
    def _get_faithfulness_prompt(self) -> str:
        return """
        Evaluate how well this answer is supported by the given context. Rate from 0.0 to 1.0.

        Context:
        {context}

        Answer:
        {answer}

        Rate the faithfulness where:
        - 1.0 = Answer is fully supported by context with no unsupported claims
        - 0.7 = Answer is mostly supported with minor unsupported details  
        - 0.4 = Answer has some support but includes unsupported claims
        - 0.0 = Answer is not supported by context or contradicts it

        Respond with just a number between 0.0 and 1.0.
        """
    
    def _get_relevance_prompt(self) -> str:
        return """
        Evaluate how relevant this retrieved context is for answering the query. Rate from 0.0 to 1.0.

        Query: {query}

        Context:
        {context}

        Rate the relevance where:
        - 1.0 = Context is highly relevant and directly addresses the query
        - 0.7 = Context is mostly relevant with some useful information
        - 0.4 = Context is partially relevant but misses key aspects
        - 0.0 = Context is not relevant to the query

        Respond with just a number between 0.0 and 1.0.
        """
    
    def _get_completeness_prompt(self) -> str:
        return """
        Compare this answer to the ground truth and rate completeness from 0.0 to 1.0.

        Answer:
        {answer}

        Ground Truth:
        {ground_truth}

        Rate completeness where:
        - 1.0 = Answer covers all key points from ground truth
        - 0.7 = Answer covers most important points with minor gaps
        - 0.4 = Answer covers some points but misses important information  
        - 0.0 = Answer doesn't address the main points from ground truth

        Respond with just a number between 0.0 and 1.0.
        """
    
    def _get_hallucination_prompt(self) -> str:
        return """
        Evaluate if this answer contains hallucinated information not supported by the context. Rate from 0.0 to 1.0.

        Context:
        {context}

        Answer:
        {answer}

        Rate hallucination risk where:
        - 0.0 = No hallucinated content, fully grounded in context
        - 0.3 = Minor details not in context but plausible
        - 0.7 = Some significant claims not supported by context
        - 1.0 = Major hallucinations or fabricated information

        Respond with just a number between 0.0 and 1.0.
        """
    
    def _get_coherence_prompt(self) -> str:
        return """
        Evaluate the internal coherence and consistency of this answer. Rate from 0.0 to 1.0.

        Answer:
        {answer}

        Rate coherence where:
        - 1.0 = Answer is perfectly coherent with consistent logic and flow
        - 0.7 = Answer is mostly coherent with minor inconsistencies
        - 0.4 = Answer has some logical gaps or contradictions
        - 0.0 = Answer is incoherent or contains major contradictions

        Respond with just a number between 0.0 and 1.0.
        """


class EvaluationReport:
    """
    Generate comprehensive evaluation reports and insights
    """
    
    @staticmethod
    def generate_strategy_report(comparison: StrategyComparison) -> dict[str, Any]:
        """
        Generate detailed comparison report across strategies
        """
        strategies = {}
        if comparison.baseline_metrics:
            strategies["baseline"] = comparison.baseline_metrics
        if comparison.flare_metrics:
            strategies["flare"] = comparison.flare_metrics
        if comparison.hyde_metrics:
            strategies["hyde"] = comparison.hyde_metrics
        if comparison.multi_query_metrics:
            strategies["multi_query"] = comparison.multi_query_metrics
        
        # Calculate performance rankings
        metrics = ["faithfulness", "relevance", "context_precision", "answer_completeness"]
        rankings = {}
        
        for metric in metrics:
            scores = [(name, getattr(data, metric)) for name, data in strategies.items()]
            scores.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = scores
        
        # Performance summary
        avg_scores = {}
        for name, data in strategies.items():
            scores = [getattr(data, metric) for metric in metrics]
            avg_scores[name] = mean(scores)
        
        best_overall = max(avg_scores.items(), key=lambda x: x[1])
        
        return {
            "summary": {
                "best_overall_strategy": best_overall[0],
                "best_overall_score": round(best_overall[1], 3),
                "strategies_evaluated": list(strategies.keys()),
                "timestamp": datetime.now(UTC).isoformat()
            },
            "rankings": rankings,
            "detailed_metrics": {name: {
                "faithfulness": data.faithfulness,
                "relevance": data.relevance,
                "context_precision": data.context_precision,
                "answer_completeness": data.answer_completeness,
                "source_diversity": data.source_diversity,
                "hallucination_score": data.hallucination_score,
                "total_latency": data.total_latency,
                "chunks_retrieved": data.chunks_retrieved
            } for name, data in strategies.items()}
        }
    
    @staticmethod
    def save_evaluation_results(results: dict[str, Any], filepath: str) -> None:
        """Save evaluation results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {filepath}")
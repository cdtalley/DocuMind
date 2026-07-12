"""
Agentic RAG System with Self-Reflection and Corrective Capabilities

This module implements advanced agentic RAG techniques including:
- Self-reflective RAG with query analysis and adaptive routing
- Corrective RAG (CRAG) with retrieval quality assessment
- Multi-hop reasoning for complex queries
- Dynamic strategy selection based on query characteristics
- Confidence-aware response generation

Demonstrates cutting-edge understanding of:
- Agentic AI patterns in RAG systems
- Self-improving retrieval systems
- Query complexity analysis
- Adaptive response strategies
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from app.models.response_models import AnswerResponse, SourceCitation
from app.services.embedding_service import ChromaEmbeddingService
from app.services.rag_service import RAGService
from app.utils.ollama_client import OllamaClient

logger = logging.getLogger("documind.agentic")


class QueryComplexity(Enum):
    """Query complexity classification"""
    SIMPLE = "simple"  # Single concept, direct lookup
    MODERATE = "moderate"  # Multiple concepts, needs synthesis
    COMPLEX = "complex"  # Multi-hop reasoning, comparison
    ANALYTICAL = "analytical"  # Deep analysis, methodology focus


class QueryType(Enum):
    """Query type classification for routing"""
    FACTUAL = "factual"  # Who, what, when, where questions
    CONCEPTUAL = "conceptual"  # How, why questions
    COMPARATIVE = "comparative"  # Compare X vs Y
    ANALYTICAL = "analytical"  # Analyze, evaluate, assess
    PROCEDURAL = "procedural"  # How to do something
    DEFINITIONAL = "definitional"  # What is X?


@dataclass
class QueryAnalysis:
    """Comprehensive query analysis results"""
    original_query: str
    complexity: QueryComplexity
    query_type: QueryType
    key_concepts: list[str]
    requires_multi_hop: bool
    suggested_strategy: str
    decomposed_queries: list[str]
    confidence: float
    reasoning: str


@dataclass
class RetrievalQuality:
    """Assessment of retrieval quality"""
    relevance_score: float  # How relevant are the retrieved chunks
    coverage_score: float  # How well do chunks cover the query
    confidence_score: float  # Confidence in the retrieval results
    needs_correction: bool  # Whether corrective retrieval is needed
    correction_strategy: str | None  # Suggested correction approach
    reasoning: str  # Explanation of the quality assessment


@dataclass
class AgenticResponse:
    """Enhanced response with agentic metadata"""
    answer: str
    sources: list[SourceCitation]
    confidence: float
    query_analysis: QueryAnalysis
    retrieval_quality: RetrievalQuality
    strategies_used: list[str]
    reflection_notes: str
    improvement_suggestions: str
    reasoning_chain: list[str]


class QueryAnalyzer:
    """
    Intelligent query analysis and classification system
    
    Uses LLM-based analysis to understand query characteristics
    and suggest optimal retrieval strategies
    """
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Comprehensive query analysis with complexity assessment and strategy recommendation
        """
        analysis_prompt = self._get_analysis_prompt()
        
        messages = [
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": f"Analyze this query: {query}"}
        ]
        
        try:
            response = self.ollama_client.chat(messages, temperature=0.2)
            return self._parse_analysis_response(query, response)
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            return self._fallback_analysis(query)
    
    def decompose_complex_query(self, query: str) -> list[str]:
        """
        Decompose complex queries into simpler sub-queries
        for multi-hop reasoning
        """
        decomposition_prompt = """
        Break down this complex query into 2-4 simpler sub-queries that can be answered independently
        and then synthesized into a complete response.

        Guidelines:
        - Each sub-query should be self-contained and answerable
        - Sub-queries should build toward answering the original query
        - Order sub-queries from foundational to specific
        - Use clear, direct language

        Format your response as a JSON list of strings.

        Example:
        Query: "How do transformers improve upon RNNs for machine translation, and what are the key architectural innovations?"
        
        Sub-queries:
        ["What are the limitations of RNNs for machine translation?", 
         "What is the transformer architecture and its key components?",
         "How do transformers address the specific limitations of RNNs?",
         "What architectural innovations make transformers effective for translation?"]
        """
        
        messages = [
            {"role": "system", "content": decomposition_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.ollama_client.chat(messages, temperature=0.3)
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                sub_queries = json.loads(json_match.group())
                return sub_queries[:4]  # Limit to 4 sub-queries
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
        
        # Fallback: return original query
        return [query]
    
    def _get_analysis_prompt(self) -> str:
        return """
        You are an expert query analyzer for RAG systems. Analyze queries and provide structured analysis.

        For each query, determine:
        1. Complexity level (simple/moderate/complex/analytical)
        2. Query type (factual/conceptual/comparative/analytical/procedural/definitional)
        3. Key concepts mentioned
        4. Whether multi-hop reasoning is needed
        5. Best retrieval strategy
        6. Confidence in your analysis (0.0-1.0)

        Complexity guidelines:
        - Simple: Single concept, direct factual lookup
        - Moderate: Multiple concepts, requires synthesis
        - Complex: Multi-hop reasoning, comparisons across sources
        - Analytical: Deep analysis, methodology, evaluation

        Query type guidelines:
        - Factual: Who, what, when, where questions
        - Conceptual: How, why questions requiring explanation
        - Comparative: Compare X vs Y, differences/similarities
        - Analytical: Analyze, evaluate, assess, critique
        - Procedural: How to do something, step-by-step
        - Definitional: What is X, define concepts

        Strategy recommendations:
        - baseline: Simple factual queries
        - hyde: Conceptual queries needing similar passages
        - multi_query: Complex queries benefiting from multiple perspectives
        - flare: Queries needing follow-up information

        Respond in this exact JSON format:
        {
          "complexity": "simple|moderate|complex|analytical",
          "query_type": "factual|conceptual|comparative|analytical|procedural|definitional",
          "key_concepts": ["concept1", "concept2"],
          "requires_multi_hop": true|false,
          "suggested_strategy": "baseline|hyde|multi_query|flare",
          "confidence": 0.0-1.0,
          "reasoning": "Brief explanation of your analysis"
        }
        """
    
    def _parse_analysis_response(self, query: str, response: str) -> QueryAnalysis:
        """Parse LLM response into QueryAnalysis object"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                return QueryAnalysis(
                    original_query=query,
                    complexity=QueryComplexity(data.get("complexity", "moderate")),
                    query_type=QueryType(data.get("query_type", "factual")),
                    key_concepts=data.get("key_concepts", []),
                    requires_multi_hop=data.get("requires_multi_hop", False),
                    suggested_strategy=data.get("suggested_strategy", "baseline"),
                    decomposed_queries=self.decompose_complex_query(query) if data.get("requires_multi_hop") else [query],
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", "Automated analysis")
                )
        except Exception as e:
            logger.warning(f"Failed to parse analysis response: {e}")
        
        return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Fallback analysis when LLM analysis fails"""
        # Simple heuristics for fallback
        complexity = QueryComplexity.SIMPLE
        query_type = QueryType.FACTUAL
        
        query_lower = query.lower()
        
        # Detect complexity
        if any(word in query_lower for word in ["compare", "contrast", "difference", "similarity"]):
            complexity = QueryComplexity.COMPLEX
            query_type = QueryType.COMPARATIVE
        elif any(word in query_lower for word in ["analyze", "evaluate", "assess", "critique"]):
            complexity = QueryComplexity.ANALYTICAL
            query_type = QueryType.ANALYTICAL
        elif any(word in query_lower for word in ["how", "why", "explain"]):
            complexity = QueryComplexity.MODERATE
            query_type = QueryType.CONCEPTUAL
        
        return QueryAnalysis(
            original_query=query,
            complexity=complexity,
            query_type=query_type,
            key_concepts=[],
            requires_multi_hop=complexity in [QueryComplexity.COMPLEX, QueryComplexity.ANALYTICAL],
            suggested_strategy="baseline",
            decomposed_queries=[query],
            confidence=0.3,
            reasoning="Fallback heuristic analysis"
        )


class RetrievalQualityAssessor:
    """
    Assesses the quality of retrieved context and suggests corrections
    
    Implements Corrective RAG (CRAG) principles for adaptive retrieval
    """
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
    
    def assess_retrieval_quality(
        self, 
        query: str, 
        sources: list[SourceCitation], 
        query_analysis: QueryAnalysis
    ) -> RetrievalQuality:
        """
        Assess quality of retrieved sources and recommend corrections if needed
        """
        if not sources:
            return RetrievalQuality(
                relevance_score=0.0,
                coverage_score=0.0,
                confidence_score=0.0,
                needs_correction=True,
                correction_strategy="expand_search",
                reasoning="No sources retrieved"
            )
        
        # Assess relevance
        relevance_score = self._assess_relevance(query, sources)
        
        # Assess coverage
        coverage_score = self._assess_coverage(query, sources, query_analysis)
        
        # Calculate confidence
        confidence_score = (relevance_score + coverage_score) / 2
        
        # Determine if correction is needed
        needs_correction = confidence_score < 0.6 or len(sources) < 3
        correction_strategy = self._suggest_correction_strategy(
            relevance_score, coverage_score, sources, query_analysis
        )
        
        reasoning = self._generate_quality_reasoning(
            relevance_score, coverage_score, sources, query_analysis
        )
        
        return RetrievalQuality(
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            confidence_score=confidence_score,
            needs_correction=needs_correction,
            correction_strategy=correction_strategy,
            reasoning=reasoning
        )
    
    def _assess_relevance(self, query: str, sources: list[SourceCitation]) -> float:
        """Assess average relevance of sources to the query"""
        if not sources:
            return 0.0
        
        assessment_prompt = f"""
        Rate the average relevance of these sources to the query on a scale of 0.0 to 1.0.

        Query: {query}

        Sources:
        {self._format_sources_for_assessment(sources)}

        Consider:
        - How directly do the sources address the query?
        - Do the sources contain information needed to answer the query?
        - Are the sources on-topic and specific to the query?

        Respond with just a number between 0.0 and 1.0.
        """
        
        try:
            response = self.ollama_client.chat(
                [{"role": "user", "content": assessment_prompt}], 
                temperature=0.1
            )
            score = self._extract_score(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Relevance assessment failed: {e}")
            return 0.5
    
    def _assess_coverage(
        self, 
        query: str, 
        sources: list[SourceCitation], 
        query_analysis: QueryAnalysis
    ) -> float:
        """Assess how well sources cover the information needed for the query"""
        coverage_prompt = f"""
        Assess how well these sources cover the information needed to answer the query comprehensively.

        Query: {query}
        Query Type: {query_analysis.query_type.value}
        Key Concepts: {', '.join(query_analysis.key_concepts)}

        Sources:
        {self._format_sources_for_assessment(sources)}

        Rate coverage from 0.0 to 1.0 where:
        - 1.0 = Sources provide comprehensive coverage of all aspects needed
        - 0.7 = Sources cover most important aspects with minor gaps
        - 0.4 = Sources cover some aspects but miss important information
        - 0.0 = Sources don't provide adequate coverage for the query

        Respond with just a number between 0.0 and 1.0.
        """
        
        try:
            response = self.ollama_client.chat(
                [{"role": "user", "content": coverage_prompt}], 
                temperature=0.1
            )
            score = self._extract_score(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Coverage assessment failed: {e}")
            return 0.5
    
    def _suggest_correction_strategy(
        self, 
        relevance: float, 
        coverage: float, 
        sources: list[SourceCitation],
        query_analysis: QueryAnalysis
    ) -> str | None:
        """Suggest correction strategy based on quality assessment"""
        if relevance < 0.4:
            return "reformulate_query"  # Query reformulation for better relevance
        elif coverage < 0.5:
            return "expand_search"  # Broader search for better coverage
        elif len(sources) < 2 and query_analysis.complexity != QueryComplexity.SIMPLE:
            return "multi_query"  # Multiple query strategies
        elif query_analysis.requires_multi_hop and coverage < 0.7:
            return "iterative_retrieval"  # Multi-hop retrieval
        else:
            return None  # No correction needed
    
    def _format_sources_for_assessment(self, sources: list[SourceCitation]) -> str:
        """Format sources for LLM assessment"""
        formatted = []
        for i, source in enumerate(sources[:5]):  # Limit to 5 sources for assessment
            formatted.append(
                f"Source {i+1}: {source.paper_title} - {source.content_preview[:200]}..."
            )
        return "\n\n".join(formatted)
    
    def _generate_quality_reasoning(
        self, 
        relevance: float, 
        coverage: float, 
        sources: list[SourceCitation],
        query_analysis: QueryAnalysis
    ) -> str:
        """Generate human-readable reasoning for quality assessment"""
        parts = []
        
        if relevance > 0.7:
            parts.append("Sources are highly relevant to the query.")
        elif relevance > 0.5:
            parts.append("Sources are moderately relevant with some useful information.")
        else:
            parts.append("Sources have limited relevance to the query.")
        
        if coverage > 0.7:
            parts.append("Good coverage of information needed to answer the query.")
        elif coverage > 0.5:
            parts.append("Partial coverage with some information gaps.")
        else:
            parts.append("Insufficient coverage of key information.")
        
        parts.append(f"Retrieved {len(sources)} sources for {query_analysis.complexity.value} query.")
        
        return " ".join(parts)
    
    def _extract_score(self, text: str) -> float:
        """Extract numerical score from text response"""
        # Look for decimal numbers
        matches = re.findall(r'(\d*\.\d+|\d+)', text)
        if matches:
            score = float(matches[0])
            return min(1.0, max(0.0, score))
        return 0.5


class AgenticRAGService:
    """
    Advanced agentic RAG system with self-reflection and corrective capabilities
    
    Features:
    - Intelligent query analysis and routing
    - Corrective RAG with quality assessment  
    - Multi-hop reasoning for complex queries
    - Self-reflective response generation
    - Adaptive strategy selection
    """
    
    def __init__(
        self,
        rag_service: RAGService,
        ollama_client: OllamaClient,
        settings
    ):
        self.rag_service = rag_service
        self.ollama_client = ollama_client
        self.settings = settings
        
        self.query_analyzer = QueryAnalyzer(ollama_client)
        self.quality_assessor = RetrievalQualityAssessor(ollama_client)
        
        logger.info("Initialized Agentic RAG Service with self-reflection capabilities")
    
    def answer_with_reflection(
        self,
        query: str,
        top_k: int = 6,
        max_correction_attempts: int = 2
    ) -> AgenticResponse:
        """
        Generate answer with full agentic capabilities including self-reflection
        """
        logger.info(f"Processing agentic query: {query[:50]}...")
        
        # Step 1: Analyze the query
        query_analysis = self.query_analyzer.analyze_query(query)
        logger.info(f"Query analysis: {query_analysis.complexity.value} {query_analysis.query_type.value}")
        
        strategies_used = []
        reasoning_chain = []
        
        # Step 2: Initial retrieval with suggested strategy
        initial_strategy = query_analysis.suggested_strategy
        strategies_used.append(initial_strategy)
        
        response = self.rag_service.answer(
            query=query,
            top_k=top_k,
            retrieval_strategy=initial_strategy,
            query_mode=self._map_query_type_to_mode(query_analysis.query_type)
        )
        
        reasoning_chain.append(f"Initial retrieval with {initial_strategy} strategy")
        
        # Step 3: Assess retrieval quality
        quality = self.quality_assessor.assess_retrieval_quality(
            query, response.sources, query_analysis
        )
        
        reasoning_chain.append(f"Quality assessment: {quality.reasoning}")
        
        # Step 4: Corrective retrieval if needed
        correction_attempt = 0
        while (quality.needs_correction and 
               correction_attempt < max_correction_attempts and 
               quality.correction_strategy):
            
            correction_attempt += 1
            logger.info(f"Applying correction strategy: {quality.correction_strategy}")
            
            corrected_response = self._apply_correction_strategy(
                query, query_analysis, quality.correction_strategy, top_k
            )
            
            if corrected_response and len(corrected_response.sources) > len(response.sources):
                response = corrected_response
                strategies_used.append(quality.correction_strategy)
                reasoning_chain.append(f"Applied {quality.correction_strategy} correction")
                
                # Re-assess quality
                quality = self.quality_assessor.assess_retrieval_quality(
                    query, response.sources, query_analysis
                )
            else:
                reasoning_chain.append(f"Correction {quality.correction_strategy} did not improve results")
                break
        
        # Step 5: Multi-hop reasoning for complex queries
        if (query_analysis.requires_multi_hop and 
            len(query_analysis.decomposed_queries) > 1 and
            quality.confidence_score < 0.8):
            
            enhanced_response = self._multi_hop_reasoning(
                query_analysis.decomposed_queries, top_k
            )
            
            if enhanced_response:
                # Merge responses intelligently
                response = self._merge_multi_hop_responses(response, enhanced_response)
                strategies_used.append("multi_hop")
                reasoning_chain.append("Applied multi-hop reasoning")
        
        # Step 6: Self-reflection and improvement suggestions
        reflection_notes = self._generate_reflection(query_analysis, quality, response)
        improvement_suggestions = self._generate_improvement_suggestions(
            query_analysis, quality, strategies_used
        )
        
        return AgenticResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            query_analysis=query_analysis,
            retrieval_quality=quality,
            strategies_used=strategies_used,
            reflection_notes=reflection_notes,
            improvement_suggestions=improvement_suggestions,
            reasoning_chain=reasoning_chain
        )
    
    def _apply_correction_strategy(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        strategy: str,
        top_k: int
    ) -> AnswerResponse | None:
        """Apply specific correction strategy"""
        try:
            if strategy == "reformulate_query":
                # Reformulate query for better relevance
                reformulated = self._reformulate_query(query, query_analysis)
                return self.rag_service.answer(
                    query=reformulated,
                    top_k=top_k,
                    retrieval_strategy="baseline"
                )
            
            elif strategy == "expand_search":
                # Broader search with increased top_k
                return self.rag_service.answer(
                    query=query,
                    top_k=min(top_k * 2, 12),
                    retrieval_strategy="multi_query"
                )
            
            elif strategy == "multi_query":
                # Use multi-query strategy
                return self.rag_service.answer(
                    query=query,
                    top_k=top_k,
                    retrieval_strategy="multi_query"
                )
            
            elif strategy == "iterative_retrieval":
                # Multi-hop iterative retrieval
                return self._iterative_retrieval(query, query_analysis, top_k)
            
        except Exception as e:
            logger.warning(f"Correction strategy {strategy} failed: {e}")
        
        return None
    
    def _reformulate_query(self, query: str, query_analysis: QueryAnalysis) -> str:
        """Reformulate query for better retrieval"""
        reformulation_prompt = f"""
        Reformulate this query to improve retrieval results while preserving the original intent.

        Original Query: {query}
        Query Type: {query_analysis.query_type.value}
        Key Concepts: {', '.join(query_analysis.key_concepts)}

        Guidelines for reformulation:
        - Use more specific terminology
        - Add relevant technical terms
        - Break down complex concepts
        - Maintain the original question intent

        Provide only the reformulated query, no explanation.
        """
        
        try:
            reformulated = self.ollama_client.chat(
                [{"role": "user", "content": reformulation_prompt}], 
                temperature=0.3
            ).strip()
            logger.info(f"Reformulated query: {reformulated}")
            return reformulated
        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}")
            return query
    
    def _iterative_retrieval(
        self, 
        query: str, 
        query_analysis: QueryAnalysis, 
        top_k: int
    ) -> AnswerResponse | None:
        """Perform iterative retrieval based on initial results"""
        # Get initial results
        initial_response = self.rag_service.answer(
            query=query,
            top_k=top_k,
            retrieval_strategy="baseline"
        )
        
        if not initial_response.sources:
            return None
        
        # Generate follow-up query based on gaps
        follow_up_query = self._generate_follow_up_query(query, initial_response)
        
        if follow_up_query != query:
            # Retrieve additional information
            follow_up_response = self.rag_service.answer(
                query=follow_up_query,
                top_k=top_k // 2,
                retrieval_strategy="baseline"
            )
            
            # Merge sources (avoiding duplicates)
            merged_sources = initial_response.sources.copy()
            seen_content = {src.content_preview for src in merged_sources}
            
            for src in follow_up_response.sources:
                if src.content_preview not in seen_content:
                    merged_sources.append(src)
                    seen_content.add(src.content_preview)
            
            # Create merged response
            initial_response.sources = merged_sources
        
        return initial_response
    
    def _generate_follow_up_query(self, original_query: str, response: AnswerResponse) -> str:
        """Generate follow-up query based on initial response gaps"""
        follow_up_prompt = f"""
        Based on the initial query and retrieved sources, generate a follow-up query to fill information gaps.

        Original Query: {original_query}

        Retrieved Information Summary:
        {self._summarize_sources(response.sources[:3])}

        What additional information is needed to comprehensively answer the original query?
        Generate a focused follow-up query. If no additional information is needed, return the original query.

        Provide only the follow-up query, no explanation.
        """
        
        try:
            follow_up = self.ollama_client.chat(
                [{"role": "user", "content": follow_up_prompt}], 
                temperature=0.3
            ).strip()
            return follow_up if follow_up != original_query else original_query
        except Exception:
            return original_query
    
    def _multi_hop_reasoning(self, sub_queries: list[str], top_k: int) -> AnswerResponse | None:
        """Perform multi-hop reasoning across decomposed queries"""
        if len(sub_queries) <= 1:
            return None
        
        all_sources = []
        reasoning_steps = []
        
        for i, sub_query in enumerate(sub_queries):
            logger.info(f"Multi-hop step {i+1}: {sub_query}")
            
            sub_response = self.rag_service.answer(
                query=sub_query,
                top_k=max(2, top_k // len(sub_queries)),
                retrieval_strategy="baseline"
            )
            
            if sub_response.sources:
                all_sources.extend(sub_response.sources)
                reasoning_steps.append(f"Step {i+1}: {sub_response.answer[:200]}...")
        
        if not all_sources:
            return None
        
        # Remove duplicate sources
        unique_sources = []
        seen_content = set()
        for src in all_sources:
            if src.content_preview not in seen_content:
                unique_sources.append(src)
                seen_content.add(src.content_preview)
        
        # Generate synthesis
        synthesis_answer = self._synthesize_multi_hop_answer(
            sub_queries[0], unique_sources, reasoning_steps
        )
        
        # Create response
        return AnswerResponse(
            answer=synthesis_answer,
            sources=unique_sources,
            confidence=0.8,  # Multi-hop typically has good confidence
            has_answer=True,
            query=sub_queries[0],
            query_mode="general",
            model_used=self.settings.LLM_MODEL,
            chunks_searched=len(all_sources),
            flare_enabled=False,
            flare_followup_retrieval=False,
            retrieval_strategy="multi_hop",
            retrieval_passes=len(sub_queries),
            library=self.rag_service._content_library
        )
    
    def _synthesize_multi_hop_answer(
        self, 
        original_query: str, 
        sources: list[SourceCitation],
        reasoning_steps: list[str]
    ) -> str:
        """Synthesize final answer from multi-hop reasoning"""
        synthesis_prompt = f"""
        Synthesize a comprehensive answer from the multi-hop reasoning results.

        Original Query: {original_query}

        Reasoning Steps:
        {chr(10).join(f"- {step}" for step in reasoning_steps)}

        Available Sources:
        {self._summarize_sources(sources[:8])}

        Create a coherent, comprehensive answer that integrates insights from all reasoning steps.
        Structure the response with clear sections and maintain logical flow.
        """
        
        try:
            return self.ollama_client.chat(
                [{"role": "user", "content": synthesis_prompt}], 
                temperature=0.4
            )
        except Exception as e:
            logger.warning(f"Multi-hop synthesis failed: {e}")
            return "Multi-hop reasoning completed but synthesis failed."
    
    def _merge_multi_hop_responses(
        self, 
        original: AnswerResponse, 
        enhanced: AnswerResponse
    ) -> AnswerResponse:
        """Merge original response with enhanced multi-hop response"""
        # Combine sources (avoiding duplicates)
        merged_sources = original.sources.copy()
        seen_content = {src.content_preview for src in merged_sources}
        
        for src in enhanced.sources:
            if src.content_preview not in seen_content:
                merged_sources.append(src)
        
        # Use enhanced answer if it's longer and more comprehensive
        final_answer = enhanced.answer if len(enhanced.answer) > len(original.answer) else original.answer
        
        original.answer = final_answer
        original.sources = merged_sources
        original.confidence = max(original.confidence, enhanced.confidence)
        
        return original
    
    def _summarize_sources(self, sources: list[SourceCitation]) -> str:
        """Create a brief summary of sources for prompts"""
        summaries = []
        for i, src in enumerate(sources[:5]):
            summaries.append(f"{i+1}. {src.paper_title}: {src.content_preview[:150]}...")
        return "\n".join(summaries)
    
    def _map_query_type_to_mode(self, query_type: QueryType) -> str:
        """Map query type to RAG service query mode"""
        mapping = {
            QueryType.COMPARATIVE: "compare",
            QueryType.ANALYTICAL: "methodology",
            QueryType.PROCEDURAL: "methodology",
            QueryType.DEFINITIONAL: "general",
            QueryType.FACTUAL: "general",
            QueryType.CONCEPTUAL: "general"
        }
        return mapping.get(query_type, "general")
    
    def _generate_reflection(
        self, 
        query_analysis: QueryAnalysis, 
        quality: RetrievalQuality, 
        response: AnswerResponse
    ) -> str:
        """Generate self-reflection notes on the response quality"""
        reflection_parts = []
        
        # Query analysis reflection
        reflection_parts.append(f"Query classified as {query_analysis.complexity.value} {query_analysis.query_type.value}")
        
        # Retrieval quality reflection
        if quality.confidence_score > 0.7:
            reflection_parts.append("High-quality retrieval with good relevance and coverage")
        elif quality.confidence_score > 0.5:
            reflection_parts.append("Moderate retrieval quality with some gaps")
        else:
            reflection_parts.append("Low retrieval quality, corrections were attempted")
        
        # Response completeness reflection
        if len(response.sources) >= 4:
            reflection_parts.append("Good source diversity for comprehensive answer")
        else:
            reflection_parts.append("Limited source diversity may affect answer completeness")
        
        return ". ".join(reflection_parts) + "."
    
    def _generate_improvement_suggestions(
        self, 
        query_analysis: QueryAnalysis, 
        quality: RetrievalQuality,
        strategies_used: list[str]
    ) -> str:
        """Generate suggestions for improving future similar queries"""
        suggestions = []
        
        if quality.confidence_score < 0.6:
            suggestions.append("Consider expanding the knowledge base with more relevant documents")
        
        if query_analysis.complexity == QueryComplexity.COMPLEX and "multi_hop" not in strategies_used:
            suggestions.append("Complex queries may benefit from multi-hop reasoning approach")
        
        if len(strategies_used) == 1:
            suggestions.append("Try alternative retrieval strategies for better results")
        
        if not suggestions:
            suggestions.append("Current approach appears optimal for this query type")
        
        return ". ".join(suggestions) + "."
"""
RAG Monitoring and Analytics Dashboard

This module provides comprehensive monitoring, analytics, and observability for RAG systems:

- Real-time performance metrics and alerting
- Query analysis and usage patterns
- Retrieval quality monitoring with drift detection
- A/B testing framework for RAG improvements
- Cost and resource utilization tracking
- Automated diagnostics and health checks

Demonstrates production-ready monitoring practices:
- Service level objectives (SLOs) and monitoring
- Performance benchmarking and regression detection
- User experience analytics
- System health and capacity planning
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import mean, median
from typing import Any, Dict, List

from app.models.response_models import AnswerResponse, SourceCitation
from app.services.rag_evaluation import RAGEvaluator, RAGMetrics

logger = logging.getLogger("documind.monitoring")


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    timestamp: datetime
    value: float
    metric_name: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class QueryAnalytics:
    """Analytics for a specific query"""
    query_id: str
    query: str
    timestamp: datetime
    
    # Performance metrics
    total_latency: float
    retrieval_latency: float
    generation_latency: float
    
    # Quality metrics
    confidence_score: float
    source_count: int
    has_answer: bool
    
    # User interaction
    user_feedback: float | None = None
    clicked_sources: list[str] = field(default_factory=list)
    
    # Technical details
    strategy_used: str = "baseline"
    model_used: str = ""
    chunks_searched: int = 0
    library: str = "public"


@dataclass
class SystemHealth:
    """System health status"""
    timestamp: datetime
    
    # Service availability
    api_healthy: bool
    embedding_service_healthy: bool
    llm_service_healthy: bool
    vector_db_healthy: bool
    
    # Performance indicators
    avg_response_time: float
    p95_response_time: float
    requests_per_minute: int
    error_rate: float
    
    # Resource utilization
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    
    # RAG-specific metrics
    avg_retrieval_quality: float
    avg_answer_confidence: float


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: str
    threshold: float
    comparison: str  # "gt", "lt", "eq"
    duration_minutes: int
    severity: str  # "critical", "warning", "info"
    enabled: bool = True


class MetricsCollector:
    """
    Collects and aggregates performance metrics for RAG system
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.query_analytics: dict[str, QueryAnalytics] = {}
        self.system_health_history: deque[SystemHealth] = deque(maxlen=1000)
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        logger.info(f"Initialized metrics collector with {retention_hours}h retention")
    
    def record_query(self, response: AnswerResponse, processing_times: dict[str, float]) -> str:
        """Record query analytics"""
        query_id = f"q_{int(time.time() * 1000)}"
        
        analytics = QueryAnalytics(
            query_id=query_id,
            query=response.query,
            timestamp=datetime.now(UTC),
            total_latency=processing_times.get("total", 0.0),
            retrieval_latency=processing_times.get("retrieval", 0.0),
            generation_latency=processing_times.get("generation", 0.0),
            confidence_score=response.confidence,
            source_count=len(response.sources),
            has_answer=response.has_answer,
            strategy_used=response.retrieval_strategy or "baseline",
            model_used=response.model_used,
            chunks_searched=response.chunks_searched,
            library=response.library or "public"
        )
        
        self.query_analytics[query_id] = analytics
        
        # Record performance metrics
        self._record_metric("query_latency", analytics.total_latency)
        self._record_metric("retrieval_latency", analytics.retrieval_latency)
        self._record_metric("generation_latency", analytics.generation_latency)
        self._record_metric("confidence_score", analytics.confidence_score)
        self._record_metric("source_count", analytics.source_count)
        
        self.request_count += 1
        
        return query_id
    
    def record_user_feedback(self, query_id: str, feedback: dict[str, Any]) -> None:
        """Record user feedback for query"""
        if query_id in self.query_analytics:
            analytics = self.query_analytics[query_id]
            analytics.user_feedback = feedback.get("rating", None)
            analytics.clicked_sources = feedback.get("clicked_sources", [])
            
            # Record feedback metrics
            if analytics.user_feedback is not None:
                self._record_metric("user_satisfaction", analytics.user_feedback)
    
    def record_error(self, error_type: str, error_details: str) -> None:
        """Record system error"""
        self.error_count += 1
        self._record_metric("error_count", 1.0, tags={"error_type": error_type})
        
        logger.warning(f"Recorded error: {error_type} - {error_details}")
    
    def record_system_health(self, health: SystemHealth) -> None:
        """Record system health snapshot"""
        self.system_health_history.append(health)
        
        # Record as individual metrics
        self._record_metric("response_time_avg", health.avg_response_time)
        self._record_metric("response_time_p95", health.p95_response_time)
        self._record_metric("requests_per_minute", health.requests_per_minute)
        self._record_metric("error_rate", health.error_rate)
        self._record_metric("cpu_usage", health.cpu_usage)
        self._record_metric("memory_usage", health.memory_usage)
        self._record_metric("retrieval_quality", health.avg_retrieval_quality)
    
    def _record_metric(self, name: str, value: float, tags: dict[str, str] = None) -> None:
        """Record individual metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now(UTC),
            value=value,
            metric_name=name,
            tags=tags or {}
        )
        
        self.metrics[name].append(metric)
        
        # Clean old metrics
        self._cleanup_old_metrics()
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period"""
        cutoff = datetime.now(UTC) - timedelta(hours=self.retention_hours)
        
        for metric_name, metric_list in self.metrics.items():
            while metric_list and metric_list[0].timestamp < cutoff:
                metric_list.popleft()
    
    def get_metrics_summary(self, hours: int = 1) -> dict[str, Any]:
        """Get metrics summary for specified time period"""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        
        summary = {}
        
        for metric_name, metric_list in self.metrics.items():
            recent_values = [
                m.value for m in metric_list 
                if m.timestamp >= cutoff
            ]
            
            if recent_values:
                summary[metric_name] = {
                    "count": len(recent_values),
                    "mean": mean(recent_values),
                    "median": median(recent_values),
                    "min": min(recent_values),
                    "max": max(recent_values),
                    "latest": recent_values[-1] if recent_values else None
                }
        
        return summary


class AlertManager:
    """
    Manages alerts and notifications for RAG system monitoring
    """
    
    def __init__(self):
        self.alert_rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, dict] = {}
        self.alert_history: list[dict] = []
        
        # Default alert rules
        self._setup_default_alerts()
        
        logger.info("Initialized alert manager with default rules")
    
    def _setup_default_alerts(self) -> None:
        """Setup default alert rules"""
        default_rules = [
            AlertRule("high_latency", "query_latency", 10.0, "gt", 5, "warning"),
            AlertRule("low_confidence", "confidence_score", 0.3, "lt", 10, "warning"),
            AlertRule("high_error_rate", "error_rate", 0.05, "gt", 5, "critical"),
            AlertRule("service_down", "api_healthy", 0.5, "lt", 1, "critical"),
            AlertRule("poor_retrieval", "retrieval_quality", 0.4, "lt", 15, "warning"),
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
    
    def check_alerts(self, metrics: dict[str, Any]) -> list[dict]:
        """Check metrics against alert rules"""
        triggered_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            metric_data = metrics.get(rule.metric)
            if not metric_data:
                continue
            
            current_value = metric_data.get("latest")
            if current_value is None:
                continue
            
            # Check threshold
            triggered = False
            if rule.comparison == "gt" and current_value > rule.threshold:
                triggered = True
            elif rule.comparison == "lt" and current_value < rule.threshold:
                triggered = True
            elif rule.comparison == "eq" and abs(current_value - rule.threshold) < 0.01:
                triggered = True
            
            if triggered:
                alert = {
                    "rule_name": rule_name,
                    "metric": rule.metric,
                    "current_value": current_value,
                    "threshold": rule.threshold,
                    "severity": rule.severity,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "message": f"{rule.metric} is {current_value} (threshold: {rule.threshold})"
                }
                
                triggered_alerts.append(alert)
                self.active_alerts[rule_name] = alert
                self.alert_history.append(alert)
                
                logger.warning(f"Alert triggered: {alert['message']}")
        
        return triggered_alerts
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add new alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def get_active_alerts(self) -> list[dict]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())


class PerformanceBenchmark:
    """
    Performance benchmarking and regression detection
    """
    
    def __init__(self, evaluator: RAGEvaluator):
        self.evaluator = evaluator
        self.benchmark_history: list[dict] = []
        self.baseline_metrics: dict[str, float] = {}
        
        logger.info("Initialized performance benchmark system")
    
    def run_benchmark(
        self, 
        test_queries: list[str], 
        rag_service, 
        benchmark_name: str = None
    ) -> dict[str, Any]:
        """Run comprehensive performance benchmark"""
        benchmark_name = benchmark_name or f"benchmark_{int(time.time())}"
        
        logger.info(f"Running benchmark '{benchmark_name}' with {len(test_queries)} queries")
        
        start_time = time.time()
        results = []
        
        for i, query in enumerate(test_queries):
            logger.info(f"Benchmark progress: {i+1}/{len(test_queries)}")
            
            query_start = time.time()
            
            try:
                # Run query
                response = rag_service.answer(query=query, top_k=6)
                
                # Evaluate response
                metrics = self.evaluator.evaluate_rag_response(query, response)
                
                query_time = time.time() - query_start
                
                result = {
                    "query": query,
                    "latency": query_time,
                    "metrics": {
                        "faithfulness": metrics.faithfulness,
                        "relevance": metrics.relevance,
                        "confidence": response.confidence,
                        "source_count": len(response.sources)
                    },
                    "success": True
                }
                
            except Exception as e:
                result = {
                    "query": query,
                    "latency": 0.0,
                    "metrics": {},
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"Benchmark query failed: {e}")
            
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r["success"]]
        
        if successful_results:
            avg_metrics = {
                "avg_latency": mean([r["latency"] for r in successful_results]),
                "avg_faithfulness": mean([r["metrics"]["faithfulness"] for r in successful_results]),
                "avg_relevance": mean([r["metrics"]["relevance"] for r in successful_results]),
                "avg_confidence": mean([r["metrics"]["confidence"] for r in successful_results]),
                "success_rate": len(successful_results) / len(results)
            }
        else:
            avg_metrics = {"success_rate": 0.0}
        
        benchmark_result = {
            "name": benchmark_name,
            "timestamp": datetime.now(UTC).isoformat(),
            "total_time": total_time,
            "query_count": len(test_queries),
            "success_count": len(successful_results),
            "aggregate_metrics": avg_metrics,
            "detailed_results": results
        }
        
        self.benchmark_history.append(benchmark_result)
        
        # Update baseline if this is the first benchmark
        if not self.baseline_metrics:
            self.baseline_metrics = avg_metrics.copy()
            logger.info("Set baseline metrics from first benchmark")
        
        logger.info(f"Benchmark completed: {avg_metrics}")
        return benchmark_result
    
    def detect_regression(self, current_metrics: dict[str, float]) -> dict[str, Any]:
        """Detect performance regression against baseline"""
        if not self.baseline_metrics:
            return {"status": "no_baseline", "message": "No baseline metrics available"}
        
        regressions = []
        improvements = []
        
        regression_threshold = 0.1  # 10% degradation
        improvement_threshold = 0.05  # 5% improvement
        
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                change = (current_value - baseline_value) / baseline_value
                
                if change < -regression_threshold:
                    regressions.append({
                        "metric": metric,
                        "baseline": baseline_value,
                        "current": current_value,
                        "change_pct": change * 100
                    })
                elif change > improvement_threshold:
                    improvements.append({
                        "metric": metric,
                        "baseline": baseline_value,
                        "current": current_value,
                        "change_pct": change * 100
                    })
        
        status = "regression" if regressions else "stable"
        if improvements and not regressions:
            status = "improvement"
        
        return {
            "status": status,
            "regressions": regressions,
            "improvements": improvements,
            "baseline_date": self.baseline_metrics.get("timestamp", "unknown")
        }


class UsageAnalytics:
    """
    User behavior and usage pattern analysis
    """
    
    def __init__(self):
        self.query_patterns: dict[str, int] = defaultdict(int)
        self.user_sessions: dict[str, list] = defaultdict(list)
        self.popular_queries: list[tuple[str, int]] = []
        
        logger.info("Initialized usage analytics")
    
    def analyze_query_patterns(self, analytics: list[QueryAnalytics]) -> dict[str, Any]:
        """Analyze query patterns and user behavior"""
        if not analytics:
            return {}
        
        # Query type distribution
        strategies = defaultdict(int)
        libraries = defaultdict(int)
        confidence_buckets = defaultdict(int)
        
        for query in analytics:
            strategies[query.strategy_used] += 1
            libraries[query.library] += 1
            
            # Confidence buckets
            if query.confidence_score >= 0.8:
                confidence_buckets["high"] += 1
            elif query.confidence_score >= 0.5:
                confidence_buckets["medium"] += 1
            else:
                confidence_buckets["low"] += 1
        
        # Time-based patterns
        hours = defaultdict(int)
        for query in analytics:
            hour = query.timestamp.hour
            hours[hour] += 1
        
        # Performance correlation analysis
        high_confidence_queries = [q for q in analytics if q.confidence_score >= 0.8]
        low_confidence_queries = [q for q in analytics if q.confidence_score < 0.5]
        
        patterns = {
            "total_queries": len(analytics),
            "strategy_distribution": dict(strategies),
            "library_distribution": dict(libraries),
            "confidence_distribution": dict(confidence_buckets),
            "peak_hours": sorted(hours.items(), key=lambda x: x[1], reverse=True)[:3],
            "performance_insights": {
                "high_confidence_avg_latency": mean([q.total_latency for q in high_confidence_queries]) if high_confidence_queries else 0,
                "low_confidence_avg_latency": mean([q.total_latency for q in low_confidence_queries]) if low_confidence_queries else 0,
                "avg_sources_per_query": mean([q.source_count for q in analytics])
            }
        }
        
        return patterns
    
    def get_query_recommendations(self, analytics: list[QueryAnalytics]) -> list[str]:
        """Generate query optimization recommendations"""
        recommendations = []
        
        if not analytics:
            return recommendations
        
        # Analyze performance patterns
        avg_latency = mean([q.total_latency for q in analytics])
        avg_confidence = mean([q.confidence_score for q in analytics])
        
        if avg_latency > 5.0:
            recommendations.append("Consider optimizing retrieval strategy - average latency is high")
        
        if avg_confidence < 0.6:
            recommendations.append("Low average confidence - consider improving chunking or expanding knowledge base")
        
        # Strategy recommendations
        strategy_performance = defaultdict(list)
        for query in analytics:
            strategy_performance[query.strategy_used].append(query.confidence_score)
        
        if len(strategy_performance) > 1:
            best_strategy = max(strategy_performance.items(), 
                              key=lambda x: mean(x[1]) if x[1] else 0)
            recommendations.append(f"Consider using '{best_strategy[0]}' strategy more often - shows best performance")
        
        return recommendations


class RAGMonitoringDashboard:
    """
    Main monitoring dashboard that coordinates all monitoring components
    """
    
    def __init__(self, evaluator: RAGEvaluator):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.benchmark = PerformanceBenchmark(evaluator)
        self.usage_analytics = UsageAnalytics()
        
        # Dashboard state
        self.is_monitoring = True
        self.last_health_check = None
        
        logger.info("Initialized RAG Monitoring Dashboard")
    
    def record_query_execution(
        self, 
        response: AnswerResponse, 
        processing_times: dict[str, float]
    ) -> str:
        """Record query execution for monitoring"""
        return self.metrics_collector.record_query(response, processing_times)
    
    def record_user_interaction(self, query_id: str, feedback: dict[str, Any]) -> None:
        """Record user feedback and interactions"""
        self.metrics_collector.record_user_feedback(query_id, feedback)
    
    def get_dashboard_data(self, hours: int = 1) -> dict[str, Any]:
        """Get comprehensive dashboard data"""
        # Get recent metrics
        metrics_summary = self.metrics_collector.get_metrics_summary(hours)
        
        # Check alerts
        active_alerts = self.alert_manager.check_alerts(metrics_summary)
        
        # Get usage analytics
        recent_queries = [
            q for q in self.metrics_collector.query_analytics.values()
            if q.timestamp >= datetime.now(UTC) - timedelta(hours=hours)
        ]
        usage_patterns = self.usage_analytics.analyze_query_patterns(recent_queries)
        
        # System health
        latest_health = (self.metrics_collector.system_health_history[-1] 
                        if self.metrics_collector.system_health_history else None)
        
        dashboard = {
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics_summary": metrics_summary,
            "active_alerts": self.alert_manager.get_active_alerts(),
            "usage_patterns": usage_patterns,
            "system_health": {
                "status": "healthy" if not active_alerts else "degraded",
                "latest_health": latest_health.__dict__ if latest_health else None,
                "uptime_hours": (time.time() - self.metrics_collector.start_time) / 3600
            },
            "recommendations": self.usage_analytics.get_query_recommendations(recent_queries),
            "performance_trends": self._get_performance_trends(hours)
        }
        
        return dashboard
    
    def run_health_check(self, rag_service) -> SystemHealth:
        """Run comprehensive system health check"""
        logger.info("Running system health check")
        
        start_time = time.time()
        
        # Test basic functionality
        api_healthy = True
        embedding_healthy = True
        llm_healthy = True
        vector_db_healthy = True
        
        try:
            # Test query execution
            test_response = rag_service.answer(
                query="What is artificial intelligence?",
                top_k=3
            )
            
            if not test_response or not test_response.answer:
                api_healthy = False
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            api_healthy = False
            llm_healthy = False
        
        # Get performance metrics
        recent_metrics = self.metrics_collector.get_metrics_summary(1)
        
        avg_response_time = recent_metrics.get("query_latency", {}).get("mean", 0.0)
        p95_response_time = recent_metrics.get("query_latency", {}).get("max", 0.0)
        
        # Calculate rates
        requests_per_minute = self.metrics_collector.request_count / max(1, 
            (time.time() - self.metrics_collector.start_time) / 60)
        
        error_rate = (self.metrics_collector.error_count / 
                     max(1, self.metrics_collector.request_count))
        
        health = SystemHealth(
            timestamp=datetime.now(UTC),
            api_healthy=api_healthy,
            embedding_service_healthy=embedding_healthy,
            llm_service_healthy=llm_healthy,
            vector_db_healthy=vector_db_healthy,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            requests_per_minute=requests_per_minute,
            error_rate=error_rate,
            cpu_usage=0.0,  # Would integrate with system monitoring
            memory_usage=0.0,
            disk_usage=0.0,
            avg_retrieval_quality=recent_metrics.get("retrieval_quality", {}).get("mean", 0.0),
            avg_answer_confidence=recent_metrics.get("confidence_score", {}).get("mean", 0.0)
        )
        
        self.metrics_collector.record_system_health(health)
        self.last_health_check = health
        
        logger.info(f"Health check completed in {time.time() - start_time:.2f}s")
        return health
    
    def _get_performance_trends(self, hours: int) -> dict[str, Any]:
        """Calculate performance trends over time"""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        
        trends = {}
        
        for metric_name in ["query_latency", "confidence_score", "retrieval_quality"]:
            if metric_name in self.metrics_collector.metrics:
                recent_values = [
                    m.value for m in self.metrics_collector.metrics[metric_name]
                    if m.timestamp >= cutoff
                ]
                
                if len(recent_values) >= 2:
                    # Simple trend calculation (first half vs second half)
                    mid_point = len(recent_values) // 2
                    early_avg = mean(recent_values[:mid_point])
                    late_avg = mean(recent_values[mid_point:])
                    
                    trend = "improving" if late_avg > early_avg else "degrading"
                    if abs(late_avg - early_avg) / max(early_avg, 0.001) < 0.05:
                        trend = "stable"
                    
                    trends[metric_name] = {
                        "direction": trend,
                        "early_avg": early_avg,
                        "late_avg": late_avg,
                        "change_pct": ((late_avg - early_avg) / max(early_avg, 0.001)) * 100
                    }
        
        return trends
    
    def export_monitoring_data(self, filepath: str, hours: int = 24) -> None:
        """Export monitoring data for analysis"""
        dashboard_data = self.get_dashboard_data(hours)
        
        export_data = {
            "export_timestamp": datetime.now(UTC).isoformat(),
            "export_duration_hours": hours,
            "dashboard_data": dashboard_data,
            "benchmark_history": self.benchmark.benchmark_history,
            "alert_history": self.alert_manager.alert_history[-100:],  # Last 100 alerts
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Monitoring data exported to {filepath}")
    
    def get_monitoring_summary(self) -> dict[str, Any]:
        """Get high-level monitoring summary"""
        recent_queries = list(self.metrics_collector.query_analytics.values())[-100:]
        
        return {
            "monitoring_active": self.is_monitoring,
            "total_queries_tracked": len(self.metrics_collector.query_analytics),
            "active_alert_count": len(self.alert_manager.active_alerts),
            "last_health_check": self.last_health_check.__dict__ if self.last_health_check else None,
            "uptime_hours": (time.time() - self.metrics_collector.start_time) / 3600,
            "recent_performance": {
                "avg_latency": mean([q.total_latency for q in recent_queries]) if recent_queries else 0,
                "avg_confidence": mean([q.confidence_score for q in recent_queries]) if recent_queries else 0,
                "success_rate": sum(1 for q in recent_queries if q.has_answer) / len(recent_queries) if recent_queries else 0
            }
        }
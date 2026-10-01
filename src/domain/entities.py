"""
src/domain/entities.py - Pure domain models for Conquer AI MLOps Model Reliability Engine.
Zero external I/O, database drivers, or vendor dependencies imported here.
Staff-Level Domain-Driven Design (DDD) & Pydantic v2 Immutable Contracts.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class ModelDeploymentRecord(BaseModel):
    """Core domain entity representing a production ML model or agent loop in Conquer AI."""
    model_id: str
    model_family: str
    environment: str = "production-k8s"
    inference_throughput_qps: float
    feature_psi_drift: float
    p99_latency_ms: float
    gpu_memory_utilization_pct: float
    days_active: float
    drift_detected: bool
    event_timestamp: Optional[str] = None


class CoxRegressionSummary(BaseModel):
    """Statistical summary of Cox Proportional Hazards regression."""
    concordance_index: float
    hazard_ratios: Dict[str, float]
    log_likelihood_ratio_p: float
    is_statistically_significant: bool
    baseline_hazard_scale: float


class ModelSurvivalSummary(BaseModel):
    """Parametric Weibull reliability metrics across Conquer AI model deployments."""
    weibull_shape_k: float
    weibull_scale_lambda: float
    median_time_to_drift_days: float
    reliability_d7_pct: float
    reliability_d14_pct: float
    reliability_d30_pct: float
    total_models_evaluated: int
    models_at_risk_count: int


class ExecutionContext(BaseModel):
    """Runtime context for MLOps reliability evaluation and Kubeflow dispatch gates."""
    execution_id: str = "default_ctx"
    risk_tolerance_alpha: float = 0.65
    psi_critical_threshold: float = 0.25
    auto_dispatch_kubeflow: bool = True


class RetrainingDispatchEvent(BaseModel):
    """Event emitted to trigger automated Kubeflow / MLflow retraining pipelines."""
    model_id: str
    model_family: str
    pipeline_endpoint: str
    urgency_tier: str  # IMMEDIATE_DRIFT, PREDICTIVE_DEGRADATION, NOMINAL
    current_survival_prob: float
    projected_compute_cost_usd: float
    recommendation: str
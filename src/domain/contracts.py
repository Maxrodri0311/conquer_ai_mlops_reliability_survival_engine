"""
src/domain/contracts.py - Inversion of Dependencies (DIP) Protocols.
All analytical, survival modeling and delivery services depend strictly on these abstract interfaces.
Enforces PEP 544 Protocol-based decoupling: Zero direct database or cloud API coupling.
"""

from typing import Protocol, Optional, List, Dict, Any
import pandas as pd
from .entities import (
    ExecutionContext,
    ModelSurvivalSummary,
    CoxRegressionSummary,
    RetrainingDispatchEvent,
)


class AnalyticalStorageProtocol(Protocol):
    """Abstract analytical storage contract (DuckDB, PostgreSQL, or In-Memory Mock)."""
    def execute_query(self, query: str) -> pd.DataFrame: ...
    def scan_dataset(self, base_path: str) -> pd.DataFrame: ...


class ModelReliabilityEngineProtocol(Protocol):
    """Abstract algorithmic engine for Cox Proportional Hazards and Weibull reliability."""
    def fit_cox_hazard_model(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> CoxRegressionSummary: ...

    def evaluate_survival_lifecycles(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> ModelSurvivalSummary: ...


class MLOpsPipelineTriggerProtocol(Protocol):
    """Abstract orchestrator for automated Kubeflow / MLflow pipeline retraining triggers."""
    def evaluate_retraining_triggers(
        self,
        df: Optional[pd.DataFrame] = None,
        context: Optional[ExecutionContext] = None,
    ) -> List[RetrainingDispatchEvent]: ...


class DeliverySinkProtocol(Protocol):
    """Abstract delivery contract for Rich Executive TUI or telemetry logs."""
    def render(
        self,
        survival_summary: ModelSurvivalSummary,
        cox_summary: CoxRegressionSummary,
        dispatch_events: List[RetrainingDispatchEvent],
    ) -> Any: ...
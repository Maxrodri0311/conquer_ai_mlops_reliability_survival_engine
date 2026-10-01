"""
tests/test_suite.py - Automated Pytest Suite.
Verifies MLOps data generation, Cox Proportional Hazards, Weibull Reliability,
Kubeflow trigger gates, and Dependency Inversion Principle (DIP).
"""

import os
import tempfile
import pytest
import pandas as pd
from src.data_generator import generate_domain_dataset
from src.core_engine import DomainAnalyticsEngine, DuckDBStorageAdapter, create_engine
from src.domain.contracts import AnalyticalStorageProtocol
from src.domain.entities import ExecutionContext


@pytest.fixture(scope="session")
def test_dataset(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data") / "test_mlops_data.parquet"
    df = generate_domain_dataset(num_records=2500, output_path=str(fn))
    return str(fn)


def test_data_generation_integrity(test_dataset):
    df = pd.read_parquet(test_dataset)
    assert len(df) == 2500
    assert "model_id" in df.columns
    assert "feature_psi_drift" in df.columns
    assert "days_active" in df.columns
    assert "drift_detected" in df.columns
    assert df.isnull().sum().sum() == 0
    # Physical invariant assertions
    assert 0.05 < df["feature_psi_drift"].mean() < 0.20
    assert df["inference_throughput_qps"].min() >= 10.0
    assert df["days_active"].min() > 0.0


def test_cox_proportional_hazards_model_fit(test_dataset):
    adapter = DuckDBStorageAdapter()
    engine = DomainAnalyticsEngine(storage=adapter, data_path=test_dataset)
    df = engine.load_dataset()
    cox = engine.fit_cox_hazard_model(df)
    
    assert cox.concordance_index > 0.50
    assert "feature_psi_drift" in cox.hazard_ratios
    # Invariant: Feature PSI drift acts as an accelerating hazard multiplier (> 1.5x)
    assert cox.hazard_ratios["feature_psi_drift"] > 1.5
    assert cox.is_statistically_significant is True


def test_weibull_survival_retention_invariants(test_dataset):
    adapter = DuckDBStorageAdapter()
    engine = DomainAnalyticsEngine(storage=adapter, data_path=test_dataset)
    df = engine.load_dataset()
    surv = engine.evaluate_survival_lifecycles(df)

    assert surv.weibull_shape_k > 0.8
    assert surv.weibull_scale_lambda > 20.0
    assert 0.0 <= surv.reliability_d14_pct <= 100.0
    assert surv.total_models_evaluated == len(df)


def test_retraining_trigger_dispatch_accuracy(test_dataset):
    adapter = DuckDBStorageAdapter()
    engine = DomainAnalyticsEngine(storage=adapter, data_path=test_dataset)
    df = engine.load_dataset()
    
    ctx = ExecutionContext(risk_tolerance_alpha=0.65, psi_critical_threshold=0.25)
    triggers = engine.evaluate_retraining_triggers(df, ctx)

    assert len(triggers) > 0
    for t in triggers:
        assert t.model_id.startswith("mdl-")
        assert "kubeflow" in t.pipeline_endpoint.lower()
        assert t.urgency_tier in ["IMMEDIATE_DRIFT", "PREDICTIVE_DEGRADATION", "NOMINAL"]


def test_core_engine_execution_with_duckdb(test_dataset):
    adapter = DuckDBStorageAdapter()
    engine = DomainAnalyticsEngine(storage=adapter, data_path=test_dataset)
    res = engine.execute_analysis()
    
    assert len(res) == 1
    assert "cox_concordance_index" in res.columns
    assert "weibull_shape_k" in res.columns
    assert res.iloc[0]["total_records"] == 2500


def test_core_engine_dependency_inversion_mock():
    """Validates that domain logic works with an in-memory mock without DuckDB or disk I/O."""
    class MockStorageAdapter:
        def execute_query(self, query: str) -> pd.DataFrame:
            return pd.DataFrame([{"total_deployments": 100, "avg_psi_drift": 0.09}])
        def scan_dataset(self, base_path: str) -> pd.DataFrame:
            # Minimal synthetic test payload for in-memory execution
            import numpy as np
            n = 200
            return pd.DataFrame({
                "model_id": [f"m-{i}" for i in range(n)],
                "model_family": ["transformer_agent"] * n,
                "environment": ["production-k8s"] * n,
                "inference_throughput_qps": np.random.uniform(50, 100, n),
                "feature_psi_drift": np.random.uniform(0.02, 0.35, n),
                "p99_latency_ms": np.random.uniform(20, 150, n),
                "gpu_memory_utilization_pct": np.random.uniform(40, 90, n),
                "days_active": np.random.uniform(1, 45, n),
                "drift_detected": np.random.choice([True, False], n),
            })

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        engine = DomainAnalyticsEngine(storage=MockStorageAdapter(), data_path=tmp.name)
        res = engine.execute_analysis()
        assert len(res) == 1
        assert res.iloc[0]["total_records"] == 200
        assert res.iloc[0]["weibull_shape_k"] > 0.5
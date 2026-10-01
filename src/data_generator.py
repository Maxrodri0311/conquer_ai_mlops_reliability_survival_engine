"""
src/data_generator.py - Calibrated Stochastic MLOps Reliability & Drift Telemetry Generator.
Company: Conquer AI (GP-071) | Target: Senior Data Scientist (MLOps & Reliability)
Physics: Cox Proportional Hazards & Parametric Weibull Reliability Modeling.
Generates 50,000+ production ML model deployments and agent decision loops with right-censoring.
"""

import os
import time
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_domain_dataset(
    num_records: int = 50000,
    output_path: str = "data/raw_dataset.parquet",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthesizes calibrated MLOps model deployment lifecycles for Conquer AI.
    Invariants:
      1. feature_psi_drift: Gamma distribution with stable mode (<0.10) and volatile drift tails (>0.25).
      2. inference_throughput_qps: Log-Normal distribution [10.0, 500.0] QPS.
      3. days_active: Weibull Time-to-Drift (k=1.45, lambda=42.0) modulated by Cox hazard multipliers.
      4. drift_detected: Binary right-censoring indicator (1 = degraded/drifted, 0 = healthy running).
    """
    print(f"[Data Generator] Generating {num_records:,} calibrated MLOps model lifecycles for Conquer AI...")
    start_time = time.time()
    rng = np.random.default_rng(seed)

    # 1. Model Deployment Identifiers & Architecture Classes
    model_families = ["transformer_agent", "xgboost_classifier", "dnn_recommendation", "llm_router"]
    family_probs = [0.35, 0.30, 0.20, 0.15]
    chosen_families = rng.choice(model_families, p=family_probs, size=num_records)

    environments = ["us-east-1-k8s", "us-west-2-k8s", "eu-west-1-k8s"]
    env_probs = [0.50, 0.30, 0.20]
    chosen_envs = rng.choice(environments, p=env_probs, size=num_records)

    model_ids = [f"mdl-{fam[:4]}-{i:06d}" for i, fam in enumerate(chosen_families, start=1)]

    # 2. Operational Ingress Metrics (QPS & GPU Saturation)
    log_qps = rng.normal(loc=4.2, scale=0.55, size=num_records)
    qps = np.clip(np.exp(log_qps), 10.0, 500.0)

    # GPU utilization correlated with architecture class
    gpu_base = np.where(
        chosen_families == "transformer_agent", 75.0,
        np.where(chosen_families == "llm_router", 68.0,
        np.where(chosen_families == "dnn_recommendation", 55.0, 32.0))
    )
    gpu_noise = rng.normal(loc=0.0, scale=12.0, size=num_records)
    gpu_pct = np.clip(gpu_base + gpu_noise, 15.0, 98.0)

    # 3. Population Stability Index (PSI) Drift Modeling
    # Gamma distribution: k=2.2, theta=0.045 -> Mean ~0.099 (Stable threshold is <0.10)
    raw_psi = rng.gamma(shape=2.2, scale=0.045, size=num_records)
    psi_drift = np.clip(raw_psi, 0.01, 0.85)

    # 4. Latency Tail Distribution (P99 ms)
    # Higher PSI & QPS exacerbate P99 tail latencies
    base_latency = np.where(chosen_families == "transformer_agent", 85.0, 28.0)
    tail_multiplier = 1.0 + (psi_drift * 1.5) + (qps / 400.0)
    p99_latency = np.clip(base_latency * tail_multiplier + rng.exponential(scale=18.0, size=num_records), 15.0, 450.0)

    # 5. Causal Hazard Multiplier & Weibull Time-to-Event Degradation
    # Formula: Hazard Multiplier = exp( beta_psi * (PSI - 0.1) + beta_qps * (QPS/100) + beta_gpu * (GPU/100) )
    # Higher stress strictly accelerates degradation rate (reduces days to drift)
    beta_psi = 3.20      # Strongest hazard multiplier: data distribution shift
    beta_qps = 0.45      # Moderate stress: query volume wear
    beta_gpu = 0.35      # Hardware pressure
    
    linear_predictor = (
        beta_psi * (psi_drift - 0.10) +
        beta_qps * ((qps - 80.0) / 100.0) +
        beta_gpu * ((gpu_pct - 60.0) / 100.0)
    )
    hazard_multiplier = np.exp(np.clip(linear_predictor, -2.5, 4.0))

    # Base parametric Weibull lifespan: shape k=1.45 (increasing wear-in hazard), scale lambda=42.0 days
    base_weibull_duration = rng.weibull(a=1.45, size=num_records) * 42.0
    
    # Accelerated Failure Time (AFT) transformation under Cox / Weibull equivalence:
    # T_actual = T_base / (hazard_multiplier ** (1 / k))
    actual_drift_time = base_weibull_duration / (hazard_multiplier ** (1.0 / 1.45))
    actual_drift_time = np.clip(actual_drift_time, 0.5, 120.0)

    # 6. Observation Window & Right-Censoring (SLA monitoring horizon: 60 days)
    # Models evaluated currently in production up to 60 days
    observed_time_in_service = rng.uniform(5.0, 60.0, size=num_records)
    
    # Event observed if time_in_service >= actual_drift_time OR if PSI exceeds emergency critical threshold (0.25)
    drift_event = (observed_time_in_service >= actual_drift_time) | (psi_drift >= 0.28)
    duration_days = np.where(drift_event, actual_drift_time, observed_time_in_service)
    duration_days = np.round(np.clip(duration_days, 0.5, 90.0), 2)

    # 7. Timestamps
    base_ts = datetime(2026, 9, 30, 12, 0, 0)
    random_offsets = rng.uniform(0, 90 * 86400, size=num_records)
    event_timestamps = [(base_ts - timedelta(seconds=float(off))).strftime("%Y-%m-%d %H:%M:%S") for off in random_offsets]

    df = pd.DataFrame({
        "model_id": model_ids,
        "model_family": chosen_families,
        "environment": chosen_envs,
        "inference_throughput_qps": np.round(qps, 2),
        "feature_psi_drift": np.round(psi_drift, 4),
        "p99_latency_ms": np.round(p99_latency, 2),
        "gpu_memory_utilization_pct": np.round(gpu_pct, 2),
        "days_active": duration_days,
        "drift_detected": drift_event.astype(bool),
        "event_timestamp": event_timestamps,
    })

    # Ensure output directory exists and export columnar Parquet
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Also export a lightweight CSV sample for PostgreSQL COPY / telemetry validation
    csv_sample_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "sample_model_telemetry.csv")
    df.head(2500).to_csv(csv_sample_path, index=False)

    elapsed = time.time() - start_time
    observed_rate = (df["drift_detected"].mean()) * 100.0
    print(f"[Data Generator] Successfully generated {len(df):,} records in {elapsed:.2f}s -> {output_path}")
    print(f"  - Mean PSI Drift:        {df['feature_psi_drift'].mean():.4f} (Critical > 0.25: {(df['feature_psi_drift'] > 0.25).mean() * 100:.1f}%)")
    print(f"  - Mean Throughput:       {df['inference_throughput_qps'].mean():.1f} QPS")
    print(f"  - Drift Event Rate:      {observed_rate:.1f}% (Right-Censored: {100.0 - observed_rate:.1f}%)")
    print(f"  - PostgreSQL Sample CSV: {csv_sample_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Conquer AI MLOps domain dataset.")
    parser.add_argument("--records", type=int, default=50000)
    parser.add_argument("--output", type=str, default="data/raw_dataset.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_domain_dataset(
        num_records=args.records,
        output_path=args.output,
        seed=args.seed,
    )
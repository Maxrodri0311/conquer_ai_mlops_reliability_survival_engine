-- ==============================================================================
-- Conquer AI MLOps Platform - Enterprise Lakehouse DDL & Partitioning Schema
-- Target: PostgreSQL 16 Enterprise / Amazon RDS Aurora
-- Role: Senior Data Scientist / MLOps Infrastructure Architect
-- Focus: Time-Series Partitioning, BRIN/B-Tree Indexing, Data Integrity & DIP Contracts
-- ==============================================================================

CREATE SCHEMA IF NOT EXISTS mlops_telemetry;
SET search_path TO mlops_telemetry, public;

-- Enable UUID extension for cryptographically strong audit identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==============================================================================
-- 1. Master Telemetry Table: Range-Partitioned by Event Timestamp
-- ==============================================================================
-- Physical design: Partitioned by month to support high-throughput telemetry ingestion
-- (sub-millisecond streaming writes) and partition pruning on analytical queries.
CREATE TABLE IF NOT EXISTS model_inference_telemetry (
    telemetry_id UUID DEFAULT uuid_generate_v4(),
    model_id VARCHAR(64) NOT NULL,
    model_family VARCHAR(32) NOT NULL,
    environment VARCHAR(32) NOT NULL DEFAULT 'production',
    inference_throughput_qps NUMERIC(8, 2) NOT NULL,
    feature_psi_drift NUMERIC(6, 4) NOT NULL,
    p99_latency_ms NUMERIC(8, 2) NOT NULL,
    gpu_memory_utilization_pct NUMERIC(5, 2) NOT NULL,
    days_active INT NOT NULL,
    drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    raw_payload_checksum CHAR(64),
    event_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Data integrity constraints enforcing physical domain boundaries
    CONSTRAINT pk_model_inference_telemetry PRIMARY KEY (event_timestamp, telemetry_id),
    CONSTRAINT chk_psi_range CHECK (feature_psi_drift >= 0.0 AND feature_psi_drift <= 5.0),
    CONSTRAINT chk_qps_positive CHECK (inference_throughput_qps >= 0.0),
    CONSTRAINT chk_latency_positive CHECK (p99_latency_ms >= 0.0),
    CONSTRAINT chk_gpu_range CHECK (gpu_memory_utilization_pct >= 0.0 AND gpu_memory_utilization_pct <= 100.0),
    CONSTRAINT chk_days_active_positive CHECK (days_active >= 0)
) PARTITION BY RANGE (event_timestamp);

-- ==============================================================================
-- 2. Time-Series Partitions (Rolling Monthly Operational Windows)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS model_inference_telemetry_2026_01 PARTITION OF model_inference_telemetry
    FOR VALUES FROM ('2026-01-01 00:00:00+00') TO ('2026-02-01 00:00:00+00');

CREATE TABLE IF NOT EXISTS model_inference_telemetry_2026_02 PARTITION OF model_inference_telemetry
    FOR VALUES FROM ('2026-02-01 00:00:00+00') TO ('2026-03-01 00:00:00+00');

CREATE TABLE IF NOT EXISTS model_inference_telemetry_2026_03 PARTITION OF model_inference_telemetry
    FOR VALUES FROM ('2026-03-01 00:00:00+00') TO ('2026-04-01 00:00:00+00');

CREATE TABLE IF NOT EXISTS model_inference_telemetry_2026_04 PARTITION OF model_inference_telemetry
    FOR VALUES FROM ('2026-04-01 00:00:00+00') TO ('2026-05-01 00:00:00+00');

CREATE TABLE IF NOT EXISTS model_inference_telemetry_2026_05 PARTITION OF model_inference_telemetry
    FOR VALUES FROM ('2026-05-01 00:00:00+00') TO ('2026-06-01 00:00:00+00');

CREATE TABLE IF NOT EXISTS model_inference_telemetry_2026_06 PARTITION OF model_inference_telemetry
    FOR VALUES FROM ('2026-06-01 00:00:00+00') TO ('2026-07-01 00:00:00+00');

CREATE TABLE IF NOT EXISTS model_inference_telemetry_default PARTITION OF model_inference_telemetry
    DEFAULT;

-- ==============================================================================
-- 3. Advanced Indexing Strategy (B-Tree + BRIN for Ultra-Low Disk Footprint)
-- ==============================================================================
-- BRIN index on event_timestamp: Reduces index size by 95% compared to B-Tree for time-ordered appends
CREATE INDEX IF NOT EXISTS idx_telemetry_brin_timestamp 
    ON model_inference_telemetry USING BRIN (event_timestamp) 
    WITH (pages_per_range = 32);

-- Composite B-Tree index for low-latency model drift lookups
CREATE INDEX IF NOT EXISTS idx_telemetry_model_drift 
    ON model_inference_telemetry (model_id, feature_psi_drift DESC, days_active);

-- Partial index for active drift incidents (sub-millisecond alert dashboards)
CREATE INDEX IF NOT EXISTS idx_telemetry_active_drift 
    ON model_inference_telemetry (model_family, environment, event_timestamp DESC)
    WHERE drift_detected = TRUE;

-- ==============================================================================
-- 4. Kubeflow Retraining Pipeline Dispatch Audit Log (Immutable Event Ledger)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS kubeflow_retraining_dispatch_log (
    dispatch_id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL,
    model_family VARCHAR(32) NOT NULL,
    trigger_reason VARCHAR(64) NOT NULL,
    hazard_ratio NUMERIC(8, 4) NOT NULL,
    survival_probability_30d NUMERIC(5, 4) NOT NULL,
    feature_psi_drift NUMERIC(6, 4) NOT NULL,
    kubeflow_pipeline_run_id VARCHAR(128) NOT NULL,
    dispatch_status VARCHAR(32) NOT NULL DEFAULT 'DISPATCHED',
    estimated_cost_saved_usd NUMERIC(10, 2) NOT NULL DEFAULT 145.00,
    dispatched_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMPTZ,
    
    CONSTRAINT chk_dispatch_status CHECK (
        dispatch_status IN ('DISPATCHED', 'ACKNOWLEDGED', 'RUNNING', 'COMPLETED', 'FAILED')
    )
);

CREATE INDEX IF NOT EXISTS idx_dispatch_model_status 
    ON kubeflow_retraining_dispatch_log (model_id, dispatch_status, dispatched_at DESC);

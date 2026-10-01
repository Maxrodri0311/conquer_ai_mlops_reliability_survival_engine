"""
src/interface.py - Rich Executive Terminal User Interface (CLI_TUI Delivery Paradigm).
Company: Conquer AI (GP-071) | Target: Senior Data Scientist (MLOps & Reliability)
Presents real-time Cox Hazard Multipliers, Weibull Reliability Curves, and Kubeflow Trigger Gates.
"""

import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

from pathlib import Path

# Ensure project root is in sys.path for standalone script execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure UTF-8 output encoding on Windows terminals to prevent cp1252 character map errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.core_engine import create_engine
from src.domain.entities import ExecutionContext

console = Console()


def run_cli():
    console.print()
    console.print(Panel(
        "[bold cyan]CONQUER AI - MLOPS MODEL RELIABILITY & DRIFT SURVIVAL PLATFORM (GP-071)[/bold cyan]\n"
        "[dim]Cox Proportional Hazards & Parametric Weibull Reliability Modeling (Kubeflow / MLflow MLOps Archetype)[/dim]",
        title="Executive MLOps Reliability Dashboard",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    engine = create_engine()
    ctx = ExecutionContext(risk_tolerance_alpha=0.65, psi_critical_threshold=0.25)
    
    df = engine.load_dataset()
    surv = engine.evaluate_survival_lifecycles(df)
    cox = engine.fit_cox_hazard_model(df)
    triggers = engine.evaluate_retraining_triggers(df, ctx)

    # 1. Panel: Weibull Reliability & Time-to-Drift Lifecycles
    t_surv = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow")
    t_surv.add_column("Reliability Dimension", style="cyan")
    t_surv.add_column("Calibrated Value", justify="right", style="bold white")
    t_surv.add_column("MLOps Engineering Impact", style="dim")

    t_surv.add_row(
        "Weibull Shape (k)",
        f"{surv.weibull_shape_k:.4f}",
        "k > 1: Increasing wear-in hazard rate over time"
    )
    t_surv.add_row(
        "Weibull Scale (lambda)",
        f"{surv.weibull_scale_lambda:.1f} days",
        "Characteristic time-to-drift across all deployments"
    )
    t_surv.add_row(
        "Median Time-to-Drift",
        f"{surv.median_time_to_drift_days:.1f} days",
        "50% of active models require retraining horizon"
    )
    t_surv.add_row(
        "Day 7 Reliability (D7)",
        f"{surv.reliability_d7_pct:.1f}%",
        "Early deployment stability benchmark"
    )
    t_surv.add_row(
        "Day 14 Reliability (D14)",
        f"{surv.reliability_d14_pct:.1f}%",
        "Key threshold before silent drift begins"
    )
    t_surv.add_row(
        "Day 30 Reliability (D30)",
        f"{surv.reliability_d30_pct:.1f}%",
        "High-risk operational window"
    )

    p_surv = Panel(
        t_surv,
        title="[bold yellow][RELIABILITY LIFECYCLES] Parametric Weibull Hazard[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    )

    # 2. Panel: Cox Proportional Hazards Multipliers
    t_cox = Table(box=box.SIMPLE, show_header=True, header_style="bold green")
    t_cox.add_column("Operational Stressor", style="cyan")
    t_cox.add_column("Hazard Ratio (exp(beta))", justify="right", style="bold white")
    t_cox.add_column("Hazard Acceleration", style="dim")

    for cov, hr in cox.hazard_ratios.items():
        desc = "Accelerates drift risk" if hr > 1.05 else "Baseline neutral"
        t_cox.add_row(cov, f"{hr:.3f}x", desc)

    t_cox.add_row(
        "Concordance Index (C-Index)",
        f"{cox.concordance_index:.4f}",
        "Discriminative accuracy of survival predictions"
    )
    t_cox.add_row(
        "Statistical Significance",
        f"p = {cox.log_likelihood_ratio_p:.6f}",
        "SIGNIFICANT (p < 0.05)" if cox.is_statistically_significant else "NOT SIGNIFICANT"
    )

    p_cox = Panel(
        t_cox,
        title="[bold green][COX PROPORTIONAL HAZARDS] Stressor Hazard Multipliers[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    )

    console.print(p_surv)
    console.print(p_cox)

    # 3. Table: Automated Kubeflow / MLflow Retraining Dispatch Triggers
    t_trig = Table(
        title="Automated Kubeflow Pipeline Retraining Triggers (Survival Gate)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    t_trig.add_column("Model ID", style="bold cyan")
    t_trig.add_column("Model Family", style="white")
    t_trig.add_column("Kubeflow Pipeline Endpoint", style="dim")
    t_trig.add_column("Urgency Tier", justify="center")
    t_trig.add_column("Survival Prob", justify="right", style="bold yellow")
    t_trig.add_column("Est. Cost Saved", justify="right", style="green")

    for ev in triggers[:6]:
        urgency_style = "[bold red]IMMEDIATE_DRIFT[/bold red]" if ev.urgency_tier == "IMMEDIATE_DRIFT" else "[bold yellow]PREDICTIVE[/bold yellow]"
        t_trig.add_row(
            ev.model_id,
            ev.model_family,
            ev.pipeline_endpoint,
            urgency_style,
            f"{ev.current_survival_prob:.1f}%",
            f"${ev.projected_compute_cost_usd:.0f} USD"
        )

    console.print(t_trig)

    # 4. Table: Model Family Reliability & Degradation Matrix across Clusters
    cohort_df = engine.execute_degradation_matrix()
    t_cluster = Table(
        title="Model Family Degradation & Reliability Matrix (Kubernetes Serving Clusters)",
        box=box.ROUNDED,
        header_style="bold blue",
    )
    t_cluster.add_column("Model Family", style="bold cyan")
    t_cluster.add_column("Serving Cluster", style="white")
    t_cluster.add_column("Total Deployed", justify="right", style="white")
    t_cluster.add_column("Avg QPS", justify="right", style="dim")
    t_cluster.add_column("Avg PSI", justify="right", style="yellow")
    t_cluster.add_column("Avg P99 (ms)", justify="right", style="dim")
    t_cluster.add_column("D14 Rel %", justify="right", style="green")
    t_cluster.add_column("Drift Rate %", justify="right", style="magenta")
    t_cluster.add_column("Critical Alerts", justify="right", style="bold red")

    for _, r in cohort_df.head(8).iterrows():
        t_cluster.add_row(
            str(r["model_family"]),
            str(r["environment"]),
            f"{int(r['total_deployments']):,}",
            f"{r['avg_qps']:.1f}",
            f"{r['avg_psi_drift']:.4f}",
            f"{r['avg_p99_ms']:.1f}",
            f"{r['rel_d14_pct']:.1f}%",
            f"{r['drift_rate_pct']:.1f}%",
            str(int(r["critical_drift_models"])),
        )

    console.print(t_cluster)
    console.print()


if __name__ == "__main__":
    run_cli()
# ==============================================================================
# Conquer AI - MLOps Infrastructure Outputs (GP-071)
# ==============================================================================

output "model_artifact_lake_arn" {
  description = "ARN of the S3 model artifact and MLflow tracking bucket"
  value       = aws_s3_bucket.model_artifact_lake.arn
}

output "ecr_repository_url" {
  description = "URL of the Amazon ECR repository for agent serving runtimes"
  value       = aws_ecr_repository.agent_serving_runtime.repository_url
}

output "kubeflow_execution_role_arn" {
  description = "IAM Role ARN for autonomous Kubeflow retraining pipelines"
  value       = aws_iam_role.kubeflow_pipeline_execution_role.arn
}

output "postgres_security_group_id" {
  description = "Security Group ID governing PostgreSQL inference telemetry store"
  value       = aws_security_group.mlops_postgres_sg.id
}
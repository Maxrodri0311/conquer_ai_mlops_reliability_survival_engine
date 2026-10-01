# ==============================================================================
# Conquer AI - Enterprise MLOps Model Reliability Infrastructure (GP-071)
# Terraform Infrastructure as Code (IaC) | Stack: AWS, ECR, S3, IAM, PostgreSQL
# ==============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  description = "Primary AWS deployment region for Conquer AI MLOps cluster"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Target deployment tier"
  type        = string
  default     = "production"
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "conquer-ai-mlops-reliability-survival-engine"
      TargetRole  = "Senior Data Scientist"
      ManagedBy   = "Terraform"
      Environment = var.environment
      Component   = "MLOps-Reliability-Plane"
    }
  }
}

# ------------------------------------------------------------------------------
# 1. AWS S3 Lakehouse & MLflow Model Artifact Store
# ------------------------------------------------------------------------------
resource "aws_s3_bucket" "model_artifact_lake" {
  bucket        = "conquer-ai-mlops-model-lake-${var.environment}"
  force_destroy = false
}

resource "aws_s3_bucket_versioning" "lake_versioning" {
  bucket = aws_s3_bucket.model_artifact_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lake_crypto" {
  bucket = aws_s3_bucket.model_artifact_lake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "lake_lifecycle" {
  bucket = aws_s3_bucket.model_artifact_lake.id

  rule {
    id     = "archive_old_model_checkpoints"
    status = "Enabled"

    filter {
      prefix = "model_artifacts/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

# ------------------------------------------------------------------------------
# 2. Amazon Elastic Container Registry (ECR) for Agent Serving Runtimes
# ------------------------------------------------------------------------------
resource "aws_ecr_repository" "agent_serving_runtime" {
  name                 = "conquer-ai/agent-serving-runtime"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "agent_serving_cleanup" {
  repository = aws_ecr_repository.agent_serving_runtime.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Retain last 30 production container images"
        selection = {
          tagStatus   = "any"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 30
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# ------------------------------------------------------------------------------
# 3. IAM Role for Kubeflow Pipelines & Autonomous Retraining Dispatch
# ------------------------------------------------------------------------------
resource "aws_iam_role" "kubeflow_pipeline_execution_role" {
  name = "conquer_ai_kubeflow_pipeline_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = ["eks.amazonaws.com", "ec2.amazonaws.com"]
        }
      }
    ]
  })
}

resource "aws_iam_policy" "kubeflow_s3_ecr_policy" {
  name        = "conquer_ai_kubeflow_s3_ecr_policy"
  description = "Allows Kubeflow Pipelines to pull model containers and write trained weights to S3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.model_artifact_lake.arn,
          "${aws_s3_bucket.model_artifact_lake.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ]
        Resource = [aws_ecr_repository.agent_serving_runtime.arn]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "kubeflow_attach" {
  role       = aws_iam_role.kubeflow_pipeline_execution_role.name
  policy_arn = aws_iam_policy.kubeflow_s3_ecr_policy.arn
}

# ------------------------------------------------------------------------------
# 4. PostgreSQL Relational Telemetry Store (RDS Specification)
# ------------------------------------------------------------------------------
resource "aws_security_group" "mlops_postgres_sg" {
  name        = "conquer-ai-mlops-postgres-sg"
  description = "Controls access to PostgreSQL inference telemetry database"

  ingress {
    description = "PostgreSQL ingress from Kubeflow EKS worker nodes"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
###############################################################################
# sagemaker_deploy_template.tf
# Terraform “template” for deploying any model to SageMaker
###############################################################################
###############################################################################
# VARIABLES (override via -var or terraform.tfvars)
###############################################################################

variable "aws_region" {
  description = "AWS region to deploy resources in"
  type        = string
  default     = "us-west-2"
}

variable "execution_role_name" {
  description = "IAM role name that SageMaker will assume"
  type        = string
  default     = "sagemaker-execution-role"
}

variable "model_name" {
  description = "Name of the SageMaker Model resource"
  type        = string
  default     = "my-model"
}

variable "ecr_image_uri" {
  description = "ECR image URI for the inference container"
  type        = string
}

variable "model_data_s3_uri" {
  description = "S3 URI of the model artifact tar.gz"
  type        = string
}

variable "container_hostname" {
  description = "Hostname inside the container"
  type        = string
  default     = "model-server"
}

variable "endpoint_config_name" {
  description = "Name of the SageMaker endpoint configuration"
  type        = string
  default     = "my-endpoint-config"
}

variable "production_variant_name" {
  description = "Variant name in the endpoint configuration"
  type        = string
  default     = "AllTraffic"
}

variable "initial_instance_count" {
  description = "Number of instances for the endpoint"
  type        = number
  default     = 1
}

variable "instance_type" {
  description = "EC2 instance type for the endpoint"
  type        = string
  default     = "ml.m5.large"
}

variable "endpoint_name" {
  description = "Name of the SageMaker endpoint"
  type        = string
  default     = "my-endpoint"
}

###############################################################################
# PROVIDER
###############################################################################

provider "aws" {
  region = var.aws_region
}

###############################################################################
# IAM ROLE & STATIC POLICY ATTACHMENTS
###############################################################################

resource "aws_iam_role" "sagemaker_execution_role" {
  name = var.execution_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "attach_s3_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "attach_sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

###############################################################################
# SAGEMAKER MODEL
###############################################################################

resource "aws_sagemaker_model" "model" {
  name               = var.model_name
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

  primary_container {
    image              = var.ecr_image_uri
    model_data_url     = var.model_data_s3_uri
    container_hostname = var.container_hostname
  }
}

###############################################################################
# ENDPOINT CONFIGURATION
###############################################################################

resource "aws_sagemaker_endpoint_configuration" "config" {
  name = var.endpoint_config_name

  production_variants {
    variant_name           = var.production_variant_name
    model_name             = aws_sagemaker_model.model.name
    initial_instance_count = var.initial_instance_count
    instance_type          = var.instance_type
  }
}

###############################################################################
# SAGEMAKER ENDPOINT
###############################################################################

resource "aws_sagemaker_endpoint" "endpoint" {
  name                 = var.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.config.name
}
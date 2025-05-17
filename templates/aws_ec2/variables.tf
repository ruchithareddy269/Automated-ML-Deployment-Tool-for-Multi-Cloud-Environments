variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "ami_id" {
  description = "AMI ID for the EC2 instance"
  type        = string
  default     = "ami-08d4f6bbae664bd41"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t2.micro"
}

variable "subnet_id" {
  description = "Subnet ID where the instance will be launched"
  type        = string
  default     = "subnet-070d54662e68443ed"
}

variable "key_name" {
  description = "SSH key pair name for accessing the instance"
  type        = string
  default     = "neel_test"
}

variable "docker_image" {
  description = "Docker image for model deployment"
  type        = string
  default     = null
}
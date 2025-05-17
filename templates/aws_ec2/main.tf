provider "aws" {
  region = var.aws_region
}

resource "aws_security_group" "ec2_sg" {
  name_prefix = "ec2-security-group-"

  # Allow SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Change this in production
  }

  # Allow HTTP access, to allow for package manager 
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  #Allow HTTPS access for docker pull 
  ingress{
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  #Allow request to come to model
  ingress{
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "EC2_SG"
  }
}

resource "aws_instance" "model_server" {
  ami             = var.ami_id
  instance_type   = var.instance_type
  subnet_id       = var.subnet_id
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  key_name        = var.key_name

  user_data = <<EOF
#!/bin/bash
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Login to Docker Hub (only if authentication is needed)
# echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

# Pull the latest Docker image
sudo docker pull ${var.docker_image}

# Run the container
sudo docker run -d -p <port_where_model_runs>:<port_where_model_runs> --name <container_name> ${var.docker_image}
EOF

  tags = {
    Name = "Model_Server"
  }
}

output "ec2_instance_public_ip" {
  value = aws_instance.model_server.public_ip
}
output "ec2_public_dns"{
    value = aws_instance.model_server.public_dns
}
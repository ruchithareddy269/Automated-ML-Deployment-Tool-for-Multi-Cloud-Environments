variable "resource_group_name" {
  description = "Name of the Resource Group"
  type        = string
}

variable "location" {
  description = "Azure Region for resources"
  type        = string
  default     = "eastus"
}

variable "virtual_network_name" {
  description = "Name of the Virtual Network"
  type        = string
  default     = "my-vnet"
}

variable "address_space" {
  description = "Address space for the Virtual Network"
  type        = string
  default     = "10.0.0.0/16"
}

variable "subnet_name" {
  description = "Name of the Subnet"
  type        = string
  default     = "my-subnet"
}

variable "subnet_address_prefix" {
  description = "Subnet address prefix"
  type        = string
  default     = "10.0.1.0/24"
}

variable "nsg_name" {
  description = "Name of the Network Security Group"
  type        = string
  default     = "my-nsg"
}

variable "public_ip_name" {
  description = "Name of the Public IP"
  type        = string
  default     = "my-public-ip"
}

variable "dns_label" {
  description = "DNS label for Public IP"
  type        = string
  default     = "mydnslabel"
}

variable "nic_name" {
  description = "Name of the Network Interface"
  type        = string
  default     = "my-nic"
}

variable "vm_name" {
  description = "Name of the VM"
  type        = string
  default     = "my-vm"
}

variable "admin_username" {
  description = "Admin username for the VM"
  type        = string
  default     = "azureuser"
}

variable "admin_password" {
  description = "Admin password for the VM"
  type        = string
  sensitive   = true
}

variable "vm_size" {
  description = "Size of the VM"
  type        = string
  default     = "Standard_B1ls"
}

variable "docker_image" {
  description = "Docker image URI"
  type        = string
  default     = "mydockerhub/yourimage:latest"
}

variable "container_name" {
  description = "Container name"
  type        = string
  default     = "model-container"
}

variable "model_port" {
  description = "Port exposed by the model container"
  type        = number
  default     = 8000
}

variable "image_publisher" {
  description = "VM image publisher"
  type        = string
  default     = "canonical"
}

variable "allowed_ssh_ips" {
  description = "Allowed IPs for SSH access"
  type        = string
  default     = "0.0.0.0/0"  # Change to your office/home IP for tighter security
}


variable "image_offer" {
  description = "VM image offer"
  type        = string
  default     = "0001-com-ubuntu-server-jammy"
}

variable "image_sku" {
  description = "VM image SKU"
  type        = string
  default     = "22_04-lts-gen2"
}

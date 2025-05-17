terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_virtual_network" "main" {
  name                = var.virtual_network_name
  address_space       = [var.address_space]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
}

resource "azurerm_subnet" "main" {
  name                 = var.subnet_name
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.subnet_address_prefix]
}

resource "azurerm_network_security_group" "main" {
  name                = var.nsg_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "Allow-SSH"
    priority                   = 1000
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }


  security_rule {
    name                       = "Allow-Model-API"
    priority                   = 1010
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = var.model_port
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

resource "azurerm_public_ip" "main" {
  name                = var.public_ip_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  allocation_method   = "Static"
  sku                 = "Standard"
  domain_name_label   = var.dns_label
}

resource "azurerm_network_interface" "main" {
  name                = var.nic_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  ip_configuration {
    name                          = "ipconfig1"
    subnet_id                     = azurerm_subnet.main.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.main.id
    # NSG is attached via the association below
  }
}

resource "azurerm_network_interface_security_group_association" "main" {
  network_interface_id      = azurerm_network_interface.main.id
  network_security_group_id = azurerm_network_security_group.main.id
}

 resource "azurerm_linux_virtual_machine" "main" {
   name                  = var.vm_name
   resource_group_name   = azurerm_resource_group.main.name
   location              = azurerm_resource_group.main.location
   size                  = var.vm_size
   admin_username                   = var.admin_username
   admin_password                   = var.admin_password
   disable_password_authentication  = false
   network_interface_ids            = [azurerm_network_interface.main.id]
  
   os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }
   source_image_reference {
     publisher = var.image_publisher
     offer     = var.image_offer
     sku       = var.image_sku
     version   = "latest"
   }

  custom_data = base64encode(<<-EOF
    #!/bin/bash
    apt-get update -y
    apt-get install -y docker.io
    systemctl enable --now docker
    docker run -d -p ${var.model_port}:${var.model_port} \
      --name ${var.container_name} ${var.docker_image}
  EOF
  )
}

output "public_ip" {
  value = azurerm_public_ip.main.ip_address
}

output "fqdn" {
  value = azurerm_public_ip.main.fqdn
}

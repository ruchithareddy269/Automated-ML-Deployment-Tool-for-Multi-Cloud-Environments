from aws_helper.ec2_deploy_wizard import DeployEC2
from k8s_helper.k8s_deploy_wizard import DeployKubernetes
from aws_helper.sagemaker_endpoint import DeploySagemakerEndpoint
from azure_helper.azure_deploy_wizard import DeployAzureVM


if __name__=="__main__":
    print("Where would you like to deploy your code?")
    print("1. AWS EC2")
    print("2. Azure VM")
    print("3. Kubernetes")
    print("4. AWS SageMaker")
    print("5. Azure ML")
    input = input("Please enter the number corresponding to your choice: ")
    if input == "1":
        print("You have selected AWS EC2.")
        # Add your AWS EC2 deployment code here
        ec2_deploy = DeployEC2()
        ec2_deploy.deploy_ec2()

    elif input == "2":
        print("You have selected Azure VM.")
        # Add your Azure VM deployment code here
        azure_vm_deploy = DeployAzureVM()
        azure_vm_deploy.deploy_azure_vm()
    elif input == "3":
        print("You have selected Kubernetes deployment.")
        k8s_deploy = DeployKubernetes()
        k8s_deploy.deploy_kubernetes()

    elif input == "4":
        print("You have selected AWS SageMaker.")
        aws_sagemaker = DeploySagemakerEndpoint()
        aws_sagemaker.deploy_to_sagemaker()
        # Add your AWS SageMaker deployment code here
    elif input == "5":
        print("You have selected Azure ML.")
        # Add your Azure ML deployment code here
    else:
        print("Invalid selection. Please try again.")
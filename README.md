# Deploy Wizard:

Deploy wizard is an CLI tool, that helps you deploy your classical models on different platforms, following the best DevOps practices using LLMs. it does all the heavy lifiting of understanding the process to deploy, creating an inference server, containizaring the model as well as generating the Infrastructure as Code needed to deploy on the specific platform.

---

## 🌟 Features

- CLI wizard to collect user inputs (Application Name, Docker Image, Port, Replicas, Cloud specific details)
- Automatic Terraform/YAML (Deployment + Service) generation via LLM
- Saves artifcats to a specified folder
- Uses actor critic loop to generate all the artifacts
- Provides instructions on how to debug and fix issues encounter when deploying the model

## Support

### Platform Supported
Currently the DeployWizard tool supports deployment for  AWS EC2, Azure VM, Kubernetes and AWS Sagemaker Endpoint. If you want to tiker with each of the process you can look athe the following files

- for EC2 /aws_helper/ec2_deploy_wizard.py ,
- For Azure VM /azure_helper/azure_deploy_wizard.py
- For Kubernets /k8s_helper/k8s_deploy_wizard.py
- For AWS SageMaker Endpoint /aws_helper/sagemaker_endpoint.py

### Demo
- EC2 https://drive.google.com/file/d/1Ep5332MPlFBJlUe-TwN6wmwxSDo403aU/view?usp=share_link
- Azure VM https://drive.google.com/file/d/1YJGgGdFh2dhpk90OGn3NVsvkUHTUQ2JN/view?usp=sharing
- Kubernetes https://drive.google.com/file/d/14vkhdl7Omg94ODmODlWwnu5zjmcdOcmz/view?usp=sharing
- Azure SageMaker Endpoint https://drive.google.com/file/d/1B9Bk5LSo3A_cphfgE3AWm8FIv9ES8u7E/view?usp=sharing


### LLM Support 
We have implemented a custom wrapper to talk to Open AI, in `talk_openai.py` file, however support can be exteneded to any LLM by implemting `Talker` present in `talk_any_llm.py`

---

## 📋 Prerequisites

- Python 3.8 or higher
- `kubectl` installed and configured
- Minikube installed and running (optional)
- OpenAI API Key (for LLM generation)
- Docker 

### Prereq for AWS EC2
1. Have a clean training file
2. The model file exported in any desired format
3. A publicly available Docker repo
4. A ssh key value pair which can be used for talking EC2
5. AWS CLI installed and user logged in
6. Know the VPC and Subnet where the model has to be deployed

### Prereq for Azure VM
1. Have a clean training file
2. The model file exported in any desired format
3. A publicly available Docker repo
4. Azure CLI installed and user logged in

renaming networking required DeployWizard can help you create it 

### Kubernetes 
1. Have a clean training file
2. The model file exported in any desired format
3. A publicly available Docker repo
4. A kubernetes cluster deployed and kubectl context configured

### AWS SageMaker Endpoint
1. Have a clean training file
2. The model file exported in any desired format
3. An ECR repo, SageMaker and an S3 bucket all set up in the same region. 


## 🚀 Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/deploy-wizard.git
cd deploy-wizard
```
2. Set up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install Required Python Packages
```
pip install -r requirements.txt
```
4. Export your OpenAI API Key
```
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxx"

```
Optionally you can save your key ia text file and point the following line to it in `talk_openai.py`
```
FILE_WITH_SECRET = '/Users/neel/Documents/Keys/OpenAIDeployWizardKey.txt'
```
5. Point the templates to proper path in your file system, by changing in all the helper packages
```
self.k8s_template_dir = "<Path Where you have cloned>/deploy_wizard/templates/k8s"
```
---
## 🧠 How to Use the Deploy Wizard

1. Run the CLI Wizard

```
python deploy_wizard.py
```
2. Answer the questions in the terminal
   

<img width="708" alt="Screenshot 2025-05-02 at 12 24 16 AM" src="https://github.com/user-attachments/assets/e3bb3545-3bf9-431c-bda8-e3141336a12d" />

3. 🐳 Docker Commands

```
docker build -t mansiiv/iris-llm:latest iris_model_inference
```
Test locally (optional)
```
docker run -p 8000:8000 mansiiv/iris-llm:latest
```
✅ Push to DockerHub

```
docker push mansiiv/iris-llm:latest
```

6. Access the Application
```
minikube service iris-llm-service --url
```
Use /predict with input:

{
  "sepal_length": 5.1,
  
  "sepal_width": 3.5,
  
  "petal_length": 1.4,
  
  "petal_width": 0.2
  
}

You should receive:

{"prediction": 0}



⚡ Notes
Make sure Minikube cluster is running before deploying.

Make sure Docker images you want to deploy are publicly accessible.

The wizard fully automates YAML generation and application — no manual Kubernetes files needed.



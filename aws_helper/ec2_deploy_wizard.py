# Import necessary modules from LangChain
from langchain.chat_models import ChatOpenAI
from talk_openai import MyOpenAI
from langchain.schema import HumanMessage
import os
from utils import *
import re
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
class SaveFileInput(BaseModel):
    file_path: str
    content: str

# Define the saving tool
@tool(args_schema=SaveFileInput)

def save_file_tool(file_path: str, content: str) -> str:
    """Save content to a file at the given path."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)
    return f"Saved to {file_path}"

class DeployEC2:
    def __init__(self, small_model="gpt-4o-mini",large_model="gpt-4o",actor_critic_iterations=2):
        self.small_model = small_model
        self.large_model = large_model
        self.actor_critic_iterations = actor_critic_iterations
        self.inference_script_generator_llm = MyOpenAI(model=large_model)
        self.inference_script_validator_llm = MyOpenAI(model=small_model)
        self.requirements_generator_llm = MyOpenAI(model=small_model)
        self.miss_llm = MyOpenAI(model=small_model)
        self.docker_generator_llm = MyOpenAI(model=small_model)
        self.terraform_generator_llm = MyOpenAI(model=large_model)
        self.terraform_validator_llm = MyOpenAI(model=small_model)
        self.infernce_dir = None
        self.terraform_dir = None
        self.python_version = get_python_version()
        self.packages = get_installed_packages()
        self.template_dir = '/Users/neel/Developer/deploy_wizard/templates/aws_ec2'
        self.main_tf_template_path = os.path.join(self.template_dir,'main.tf')
        self.vars_tf_template_path = os.path.join(self.template_dir,'variables.tf')


    def get_model_info(self):
        """Ask the user to provide model details and data shape."""
        print("To determine the input shape of a single record, you can do the following:")
        print("1. If you have a Pandas DataFrame, use df.iloc[0].shape to get the shape of a single row.")
        print("2. If using NumPy, call data[0].shape on your dataset.")
        print("3. If your data is in a list format, check the length of the first element using len(data[0]).")
        
        model_type = input("Enter the model type (Binary Classification, Multiclass Classification, Regression): ").strip()
        data_shape = input("Enter the expected shape of a single row of input data (e.g., (1, 10) for 10 features): ").strip()
        training_script_path = input("Enter the full path of training script: ").strip()
        self.infernce_dir = input("Enter the directory where you want to save the inference script: ").strip()
        return model_type, data_shape, training_script_path
        
    def generate_inference_script_with_llm(self,training_script, model_type, data_shape, feedback=None):
        """Use an LLM to generate an inference script based on the provided training script and details."""
        if feedback is None:        
            #if feedback is None, then we are generating the initial prompt
            messages = [
                {"role": "system", "content": "You are an expert Python developer."},
                {"role": "user", "content": f'''
                You are an expert Python developer. Your task is to generate an inference script that exposes a trained machine learning model as an API.

                ### Role: 
                You are to assist in writing an inference script for an AI agent that will deploy a trained model.

                ### Context:
                - The training script provided is below:
                """
                {training_script}
                """
                - The model type is {model_type}.
                - The expected input shape for predictions is {data_shape}.

                ### Action:
                Write a Python script that:
                - Loads the trained model.
                - Accepts input data via an API.
                - Preprocesses the input.
                - Makes predictions using the model.
                - Returns the prediction as JSON.
                - Uses FastAPI for API exposure.

                ### Expected Output Format:
                A complete, standalone Python script that follows best practices and is ready for execution.
                '''}
                ]
        else:
            #feedback is present so we treat it as a continuation of the conversation
            messages= feedback
        response = self.inference_script_generator_llm.invoke(messages)
        return response
    
    def evaluate_inference_script_with_llm(self,inference_script):
        """Use an LLM to evaluate the generated inference script and provide feedback."""
        messages = [
            {"role": "system", "content": "You are a code reviewer and expert in Python API development."},
            {"role": "user", "content": f"""
            You are reviewing an inference script generated for exposing a trained ML model as an API.

            ### Task:
            - Analyze the script for correctness, best practices, and completeness.
            - Identify any missing components or improvements.
            - If the script is perfect, respond with 'No changes needed.'
            - If changes are required, specify what needs to be improved.

            ### Script to review:
            ```
            {inference_script}
            ```

            ### Expected Output:
            - Either 'No changes needed.' OR a detailed improvement plan.
            """}
        ]
        
        return self.inference_script_validator_llm.invoke(messages)

    def actor_critic_inference_script(self,training_script, model_type, data_shape):
        """
        Runs the actor-critic loop:
        - Actor generates an inference script.
        - Critic evaluates the script.
        - If the script is sufficient, exit early.
        - Otherwise, Actor refines the script based on feedback.
        """
        feedback = None  # No feedback initially

        for _ in range(self.actor_critic_iterations):  # Max iterations: 1
            # Step 1: Actor generates inference script
            inference_script = self.generate_inference_script_with_llm(training_script, model_type, data_shape, feedback)

            # Step 2: Critic evaluates the script
            critic_feedback = self.evaluate_inference_script_with_llm(inference_script)

            if "no changes needed" in critic_feedback.lower():
                print("✅ Inference script is satisfactory. Exiting early.")
                return inference_script  # Early exit if script is sufficient

            # Step 3: Actor refines script using critic's feedback
            feedback = [
                {"role": "assistant", "content": inference_script},
                {"role": "user", "content": f"Revise based on this feedback:\n{critic_feedback}"}
            ]

        print("🔄 Max iterations reached. Returning final script.")
        return inference_script
    
    def extract_python_script(self,llm_response):
        """
        Extracts the main Python script from an LLM response.

        Args:
            llm_response (str): The response text containing a Python code block.

        Returns:
            str: The extracted Python script, or an empty string if no script is found.
        """
        match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
        return match.group(1) if match else ""
    
    def get_inferece_script(self):
        #get model info from user
        model_type, data_shape, training_script_path = self.get_model_info()
        #read the training script
        training_script = file_reader(training_script_path)
        #generate the inference script
        raw_inference_script = self.actor_critic_inference_script(training_script, model_type, data_shape)
        # extract the inference script from the LLM response
        inference_script = self.extract_python_script(raw_inference_script)
        # save the inference script to a file
        inference_script_path = os.path.join(self.infernce_dir, 'inference.py')
        # inference_script_path = '/Users/neel/Developer/deploy_wizard/iris_model_inference/inference.py'
        write_to_file(inference_script_path, inference_script)


    def extract_imported_libraries(self,python_script):
        """
        Extracts imported libraries from a Python script.

        Args:
            python_script (str): The Python script content.

        Returns:
            list: A list of unique libraries found in the script.
        """
        matches = re.findall(r"^\s*(?:import|from)\s+([\w\d_\.]+)", python_script, re.MULTILINE)
        return list(set(matches))  # Remove duplicates


    def generate_requirements_with_llm(self,inference_script):
        """
        Uses an LLM to generate a requirements.txt file based on the inference script.

        Args:
            inference_script (str): The Python inference script.

        Returns:s
            str: The generated requirements.txt content.
        """
        imported_libraries = self.extract_imported_libraries(inference_script)
        libraries_str = ", ".join(imported_libraries)

        messages = [
            {"role": "system", "content": "You are an expert Python package manager."},
            {"role": "user", "content": f"""
            for the inference file you generated earlier, I need to create a `requirements.txt` file.
            ### Task:
            - Generate a `requirements.txt` file based on the following libraries used in the script:
             - Also include known libraries which are not in the script but are commonly used for model inference.
            - Ensure the libraries are listed in a format suitable for `pip install`.
            - Python version is {self.python_version}.
            - To check which version to install refer to the" this list of all installed packages in the environment: {self.packages}"""}
        ]

        return self.inference_script_generator_llm.invoke(messages)
    
    def extract_requirements_txt(self,llm_response):
        """
        Extracts the main contents of the `requirements.txt` file from an LLM response.

        Args:
            llm_response (str): The response text containing the `requirements.txt` section.

        Returns:
            str: The extracted `requirements.txt` content as a string.
        """
        match = re.search(r"```(?:\w+\n)?(.*?)\n```", llm_response, re.DOTALL)
        return match.group(1).strip() if match else ""

    def get_requirements_txt(self):
        extracted_script = file_reader(os.path.join(self.infernce_dir, 'inference.py'))
        requirements_txt = self.generate_requirements_with_llm(extracted_script)
        requirements_txt_content = self.extract_requirements_txt(requirements_txt)
        requirements_txt_path = os.path.join(self.infernce_dir, 'requirements.txt')
        write_to_file(requirements_txt_path, requirements_txt_content)

    def modify_model_loading_with_llm(self,inference_script):
        """
        Uses an LLM to modify the inference script, ensuring the model is loaded from an ENV variable if set,
        while keeping the rest of the script unchanged.

        Args:
            inference_script (str): The raw inference script.

        Returns:
            str: The modified inference script with the environment-based model loading.
        """
        messages = [
            {"role": "system", "content": "You are an expert Python developer."},
            {"role": "user", "content": f"""
            The following Python script is an inference API.

            ### Task:
            - Identify the line(s) where the model is loaded (e.g., `model = joblib.load("model.pkl")`).
            - Modify only those lines to:
            - Load the model from an environment variable `MODEL_PATH` if it is set.
            - Otherwise, fall back to the original model path.
            - Keep the rest of the script **unchanged**.

            ### Expected Output:
            The entire Python script with only the necessary modification applied.

            ### Example Modification:
            **Before:**
            ```python
            import joblib
            model = joblib.load("model.pkl")
            ```

            **After:**
            ```python
            import os
            MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

            import joblib
            model = joblib.load(MODEL_PATH)
            ```

            Now, apply the same transformation to the given script.

            ### Script:
            ```
            {inference_script}
            ```
            """}
        ]

        return self.miss_llm.invoke(messages)
    
    def prepare_inference_script_for_containerization(self):
        inference_script = file_reader(os.path.join(self.infernce_dir, 'inference.py'))
        updated_inference_script = self.modify_model_loading_with_llm(inference_script)
        updated_inference_script = self.extract_python_script(updated_inference_script)
        # Save the updated inference script
        inference_script_path = os.path.join(self.infernce_dir, 'inference.py')
        write_to_file(inference_script_path, updated_inference_script)
    
    def generate_dockerfile(self,options, feedback=None):
        if feedback is None:
            messages = [
                {
                    "role": "system",
                    "content": "You are a Docker expert."
                },
                {
                    "role": "user",
                    "content": f'''
            You are an expert Dockerfile Generation Agent. Your task is to generate a **Dockerfile** that encapsulates a complete model inference environment.

            ### Role:
            Assist in writing a Dockerfile for a containerized model inference application using **only relative paths**.

            ### Context:
            - **Working Directory**: The Dockerfile is located inside the project directory where the model inference script, requirements file, and model file are also stored.
            - **Python Version**: {self.python_version}
            - **Inference Script**:
                - Location (relative to the Dockerfile): `{options['inference_script_path']}`
                - Contents:
                ```
                {options['inference_script_content']}
                ```
            - **Requirements File**:
                - Location (relative to the Dockerfile): `{options['requirements_txt_path']}`
                - Contents:
                ```
                {options['requirements_txt_content']}
                ```
            - **Model File**:
                - Location (relative to the Dockerfile): `{options['model_path']}`

            ### Action:
            Generate a **Dockerfile** that:
            - Uses an appropriate base image (e.g., `python:3.9-slim`).
            - Sets the working directory to `/app`.
            - Installs Python dependencies from the provided requirements file.
            - **Copies all necessary files using relative paths (i.e., no absolute paths)**.
            - Ensures that the model file is placed inside `/app/model/` and creates the directory if needed.
            - Configures any necessary environment variables (e.g., `MODEL_PATH=/app/model/{options['model_path'].split('/')[-1]}`).
            - Exposes a port if the inference script serves an API (e.g., FastAPI exposing `8000`).
            - Specifies the command or entrypoint to run the inference script.
            - Follows **best practices** for caching and cleanup (e.g., using `--no-cache-dir` for `pip install`).

            ### Additional Constraints:
            - **Do not use absolute paths.**  
            - **Assume all files are within the same directory as the Dockerfile when running `docker build .`.**
            - **Ensure the COPY commands properly reflect this.**
            
            ### Expected Output:
            Provide a complete **Dockerfile** as a code block with Dockerfile syntax highlighting.
            '''
                }
            ]
        else:
            messages = feedback
        response = self.docker_generator_llm.invoke(messages)
        return response
    
    def extract_dockerfile(self,llm_response):
        """
        Extracts the main Python script from an LLM response.

        Args:
            llm_response (str): The response text containing a Python code block.

        Returns:
            str: The extracted Dockerfile, or an empty string if no script is found.
        """
        match = re.search(r"```Dockerfile\n(.*?)\n```", llm_response, re.DOTALL)
        return match.group(1) if match else ""
    
    #local testind and deployedment
    def dockerfile_testing(self):
        print("We will now try validating that the Dockerfile is correctly set up for building and running the model inference application.")
        print("This will involve building the Docker image locally and running a container to test the model predictions.")
        print("Let's start by building the Docker image.")
        message ="Can you provide me instructions on how to build this docker image and run it, locally?"
        while True:
            response = self.docker_generator_llm.invoke(message)
            print(response)
            message = input("Enter your response, if everything is working say gg: ")
            if message == 'gg':
                break
                
    def push_to_dockerhub(self):
        print("We will now try pushing the Docker image to Docker Hub.")
        print("This will involve tagging the image and pushing it to your Docker Hub repository.")
        print("Let's start by pushing the Docker image.")
        message ="Can you provide me instructions on how to push this docker image to docker hub?"
        while True:
            response = self.docker_generator_llm.invoke(message)
            print(response)
            message = input("Enter your response, if everything is working say gg: ")
            if message == 'gg':
                break
    
    def dockerizing_the_model(self):
        context_for_dockerfile = {
        "inference_script_path": os.path.join(self.infernce_dir, 'inference.py'),
        "inference_script_content": file_reader(os.path.join(self.infernce_dir, 'inference.py')),
        "requirements_txt_path": os.path.join(self.infernce_dir, 'requirements.txt'),
        "requirements_txt_content": file_reader(os.path.join(self.infernce_dir, 'requirements.txt'))}
        #context_for_dockerfile["model_path"] = "/Users/neel/Developer/deploy_wizard/iris_model_inference/iris_model.pkl"
        context_for_dockerfile["model_path"] = input(f"Enter the model path, make sure it is in the director {self.infernce_dir}: ").strip()
        self.prepare_inference_script_for_containerization()
        raw_dockerfile = self.generate_dockerfile(context_for_dockerfile)
        dockerfile = self.extract_dockerfile(raw_dockerfile)
        dockerfile_path = os.path.join(self.infernce_dir, 'Dockerfile')
        write_to_file(dockerfile_path, dockerfile)
        #test the dockerfile locall
        self.dockerfile_testing()
        #pushing to docker hub
        self.push_to_dockerhub()      
    
    def generate_ec2_terraform_file(self,params,feedback=None):
        if not feedback:
            messages = [
                {"role": "system", "content": "You are an expert Terraform developer."},
                {"role": "user",
                "content": f'''
                ## Role:
                You are an **expert DevOps assistant** specializing in Terraform and AWS EC2 deployments. Your task is to generate valid Terraform files (`main.tf` and `variables.tf`) while maintaining correct syntax.

                ## Problem Statement:
                A user wants to deploy a **machine learning model** as an API on an **AWS EC2 instance using Docker**. The Terraform script should:
                - Create an **EC2 instance** inside a specified **VPC and Subnet**.
                - Configure a **Security Group** to allow **SSH, HTTP, HTTPS, and API traffic**.
                - Pull a Docker image and run it on the EC2 instance.
                - Replace placeholders with provided variable values.
                - For rules in securtiy group if no ips are specified then consider all ips 0.0.0.0/0

                ## **Terraform Templates**
                ### `main.tf`
                {params['main_tf']} 
                +/n/n
                ### `variables.tf`
                {params['vars_tf']} 

                ##Provided values:
                - **VPC ID**: {params['vpc_id']}
                - **Region**: {params['region']}
                - **Subnet ID**: {params['subnet_id']}
                - **Instance Type**: {params['instance_type']}
                - **Key Pair Name**: {params['key_pair_name']}
                - **AMI ID**: {params['ami_id']}
                - **Model Port**: {params['model_port']}
                - **Model Name**: {params['model_name']}   
                - **Image Name**: {params['image_name']}
                - **Container Name**: {params['container_name']}
                - Make sure these are set as default values in values.yaml

                ## Expected Output:
                - **Replace all placeholders** in `main.tf` and `variables.tf` with actual values.
                - **Maintain correct Terraform syntax**.
                - **Return two separate files (`main.tf` and `variables.tf`)**.
                - **Ensure the Terraform script is valid and deployable**.
                - **Output must be structured** as separate code blocks for `main.tf` and `variables.tf`.
                - It should be ``hcl \n ### filename`` syntax highlighted.
                '''}
            ]
        else:
            messages = feedback
        response = self.terraform_generator_llm.invoke(messages)
        return response

    def evaluate_terraform_with_llm(self,terraform_code):
        """Use an LLM to evaluate the generated Terraform script and provide feedback."""
        messages = [
            {"role": "system", "content": "You are a Terraform expert and code reviewer specializing in AWS infrastructure."},
            {"role": "user", "content": f"""
            You are reviewing a Terraform script that provisions an AWS EC2 instance.

            ### Task:
            - Analyze the script for correctness, best practices, and security compliance.
            - Identify any missing components or improvements.
            - If the script is perfect, respond with 'No changes needed.'
            - If changes are required, specify what needs to be improved.
            - Make Sure all variables are set as default values in values.yaml based on the values given
            - If some value is explicitly mentioned in the script then it should be set as default value in values.yaml
            - Dont use any new variables in the script, or use values not provided in the prompt


            ### Script to review:
            ```
            {terraform_code}
            ```

            ### Expected Output:
            - Either 'No changes needed.' OR a detailed improvement plan.
            """}
        ]

        return self.terraform_validator_llm.invoke(messages)


    def actor_critic_terraform(self,params):
        """
        Runs the actor-critic loop for Terraform:
        - Actor generates Terraform code.
        - Critic evaluates the script.
        - If the script is sufficient, exit early.
        - Otherwise, Actor refines the script based on feedback.
        - Max iterations: 2
        """
        feedback = None  # No feedback initially
        # Step 1: Actor generates Terraform script
        terraform_script = self.generate_ec2_terraform_file(params, feedback)
        for _ in range(self.actor_critic_iterations):  # Max iterations: 2
            # Step 2: Critic evaluates the Terraform script
            critic_feedback = self.evaluate_terraform_with_llm(terraform_script)

            if "no changes needed" in critic_feedback.lower():
                print("✅ Terraform script is satisfactory. Exiting early.")
                return terraform_script  # Early exit if script is sufficient

            # Step 3: Actor refines Terraform script using critic's feedback
            feedback = [
                {"role": "assistant", "content": terraform_script},
                {"role": "user", "content": f"Revise based on this feedback:\n{critic_feedback}"}
            ]

            terraform_script = self.generate_ec2_terraform_file(params, feedback)


        print("🔄 Max iterations reached. Returning final Terraform script.")
        return terraform_script

    def extract_and_save_hcl_files(self,response_str: str, save_dir: str = './k8s_files'):
        llm = ChatOpenAI(model="gpt-4o-mini")

        tools = [save_file_tool]

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
        )

        response = agent.invoke(
            f"""
            1. Extract all Terraform code blocks from the given text.
            2. For each extracted tf of tfvar files:
                - If it main, save it as '{self.terraform_dir}/main.tf'.
                - If it is a variables file, save it as '{self.terraform_dir}/variables.tf'.
                - Otherwise, save it as '{self.terraform_dir}/other.tf'.
            3. Use the `save_file_tool` tool to save the content.
            4. Do not output anything else except confirming tool usage.

            ### Text:
            {response_str}
            """
        )
        return response

    def terraform_validation(self):
        msg = "Can you provde the things I will have to change before running the terraform script and instructuion on running it"
        response = self.terraform_generator_llm.invoke(msg)
        print(response)
    
    def terraform_deployment_validator(self):
        msg = "Can you provide me instructions on how to deploy this terraform script?"
        response = self.terraform_generator_llm.invoke(msg)
        print(response)
        while True:
            msg = input("Enter your response, if everything is working say gg: ")
            if msg == 'gg':
                break
            response = self.terraform_generator_llm.invoke(msg)
            print(response)
    
    def get_deployment_parameter(self):
        base_main_tf = file_reader(self.main_tf_template_path)
        base_vars_tf = file_reader(self.vars_tf_template_path)

        params = {}
        # self.terraform_dir = "/Users/neel/Developer/ec2_iris/aws_terraform"
        self.terraform_dir = input("Enter the directory where you want to save the terraform files: ")
        params["vpc_id"] = input("Enter VPC ID: ")
        params["region"] = input("Enter AWS Region: ")
        params["subnet_id"] = input("Enter Subnet ID: ")
        params["instance_type"] = input("Enter Instance Type (e.g., t2.micro): ")
        params["security_group_id"] = input("Enter Security Group ID: ")
        params["key_pair_name"] = input("Enter EC2 Key Pair Name: ")
        params["ami_id"] = input("Enter AMI ID: ")
        params["model_port"] = int(input("Enter Model Port (e.g., 8000): "))
        params["model_name"] = input("Enter Model Name: ")
        params["image_name"] = input("Enter Docker Image Name (e.g., user/image:tag): ")
        params["container_name"] = input("Enter Container Name: ")
        params["main_tf"] = base_main_tf
        params["vars_tf"] = base_vars_tf
        # params = {
        # "vpc_id": "vpc-0f0aea174086b6625",
        # "region": "us-west-1",
        # "subnet_id": "subnet-070d54662e68443ed",
        # "instance_type": "t2.micro",
        # "security_group_id": "sg-03aa3023dd84cf4a5",
        # "key_pair_name": "neel_test",
        # "ami_id": "ami-08d4f6bbae664bd41",
        # "model_port": 8000,
        # "model_name": "iris_model",
        # "image_name": "neel26d/iris_model_inference:latest",
        # "container_name": "iris_model",
        # "main_tf": base_main_tf,
        # "vars_tf": base_vars_tf}
        return params

    def orchestrate_terraform_deployment(self):
        params = self.get_deployment_parameter()
        raw_terraform_script = self.actor_critic_terraform(params)
        self.extract_and_save_hcl_files(raw_terraform_script)
        self.terraform_validation
        self.terraform_deployment_validator()

    def deploy_ec2(self):
        # Step 1: Get model info and generate inference script
        self.get_inferece_script()
        
        # Step 2: Generate requirements.txt
        self.get_requirements_txt()
        
        # Step 3: Dockerize the model
        self.dockerizing_the_model()
        
        # Step 4: Orchestrate Terraform deployment
        self.orchestrate_terraform_deployment()
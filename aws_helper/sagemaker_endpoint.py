# Import necessary modules from LangChain
from langchain.chat_models import ChatOpenAI
from talk_openai import MyOpenAI
from langchain.schema import HumanMessage
import os
import subprocess
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

class DeploySagemakerEndpoint:
    def __init__(self, small_model="gpt-4o-mini",large_model="gpt-4o",actor_critic_iterations=2):
        self.small_model = small_model
        self.large_model = large_model
        self.actor_critic_iterations = actor_critic_iterations
        self.inference_dir = None
        self.inference_llm = MyOpenAI(model=large_model)
        self.inference_critic_llm = MyOpenAI(model=small_model)
        self.actor_critic_iterations = actor_critic_iterations
        self.docker_generator_llm = MyOpenAI(model=large_model)
        self.python_version = get_python_version()
        self.packages = get_installed_packages()
        self.local_image_name = None
        self.aws_account_id = None
        self.ecr_repo_name = None
        self.aws_region = None
        self.ecr_image_full_name = None
        self.s3_model_uri = None
        self.terrraform_generator_llm = MyOpenAI(model="gpt-4o")
        self.terraform_dir = None
        self.terraform_critical_llm = MyOpenAI(model="gpt-4o")
        self.ecr_repo_name
        self.template_dir = "/Users/neel/Developer/deploy_wizard/templates/aws_sagemaker_endpoint"
        self.template_training_script_path = os.path.join(self.template_dir, "inference.py")
        self.template_dockerfile_path = os.path.join(self.template_dir, "Dockerfile")
        self.template_main_tf_path = os.path.join(self.template_dir, "main.tf")
        self.template_serve_path = os.path.join(self.template_dir, "serve")

    def get_model_info(self):
        """Ask the user to provide model details and data shape."""
        print("To determine the input shape of a single record, you can do the following:")
        print("1. If you have a Pandas DataFrame, use df.iloc[0].shape to get the shape of a single row.")
        print("2. If using NumPy, call data[0].shape on your dataset.")
        print("3. If your data is in a list format, check the length of the first element using len(data[0]).")
        
        model_type = input("Enter the model type (Binary Classification, Multiclass Classification, Regression): ").strip()
        data_shape = input("Enter the expected shape of a single row of input data (e.g., (1, 10) for 10 features): ").strip()
        training_script_path = input("Enter the full path of training script: ").strip()
        self.inference_dir = input("Enter the directory where you want to save the inference script: ").strip()
        return model_type, data_shape, training_script_path
    
    def generate_sagemaker_inference_bundle(self,training_script, model_type, data_shape, feedback=None):
        """Use an LLM to generate inference.py + serve for a SageMaker endpoint."""
        if feedback is None:
            messages = [
                {"role": "system", "content": "You are an expert Python developer."},
                {"role": "user", "content": f'''
            You are to generate *two* files for AWS SageMaker custom inference:

            1) **inference.py**  
            - From the training script identity the name of the model file and the format in which the model is saved. And use similar name and format for loading the model
            - Use FastAPI, listen on 0.0.0.0:8080  
            - **GET /ping** → return JSON {{ "status": "OK" }} with HTTP status code **200**  
            - **POST /invocations** → accept `application/json` with key `"instances"`,  
                preprocess into the shape {data_shape},  
                load the trained model from `/opt/ml/model`,  
                call `model.predict(...)`,  
                and return `{{"predictions": [...]}}`.  
            - Follow best practices: exception handling, logging, and specify `status_code=200` on ping.
            - The model will be saved is `/opt/ml/model/`, you will understand how the model is saved from the training script.

            2) **serve**  
            - A bash‐executable script that SageMaker will invoke to launch your FastAPI app  
            - e.g. `uvicorn inference:app --host 0.0.0.0 --port 8080`

            ### Context:
            The training script  used is:
            \"\"\"
            {training_script}
            \"\"\"
            Model type: {model_type}.

            Provide the *complete* contents of both files (with proper shebang for `serve`).
            ''' }
                    ]
        else:
            messages = feedback
        response = self.inference_llm.invoke(messages)
        return response
        
    def evaluate_inference_script_with_llm(self,inference_script):
        """Use an LLM to evaluate the generated inference script and provide feedback."""
        messages = [
            {"role": "system", "content": "You are a code reviewer and expert in Python API development."},
            {"role": "user", "content": f"""
            You are reviewing an inference script generated for exposing a trained ML model as an API on AWS SageMaker.

            ### Task:
            - Analyze the script for correctness, best practices, and completeness. (it should have a /ping and /invocations endpoint)
            - Identify any missing components or improvements.
            - should load the model from /opt/ml/model/ and call model.predict(...)
            - If the script is perfect, respond with 'No changes needed.'
            - If changes are required, specify what needs to be improved.
            - there should be a serve script that is executable and launches the FastAPI app.

            ### Script to review:
            ```
            {inference_script}
            ```

            ### Expected Output:
            - Either 'No changes needed.' OR a detailed improvement plan.
            """}
        ]
        
        return self.inference_critic_llm.invoke(messages)

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
                inference_script = self.generate_sagemaker_inference_bundle(training_script, model_type, data_shape, feedback)

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
    
    def extract_and_save_inference_files(self,response_str: str, save_dir: str = ''):
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
            1. Extract the inference script and serve script from the response string.
            2. For each extracted script:
                - If it is the inference script, save it as '{self.inference_dir}/inference.py'.
                - If it is the serve script, save it as '{self.inference_dir}/serve'.
                - Otherwise, save it as '{self.inference_dir}/<appropriate file_name and extension>'.
            3. Use the `save_file_tool` tool to save the content.
            4. Do not output anything else except confirming tool usage.

            ### Text:
            {response_str}
            """
        )
        return response
    
    def get_inferece_script(self):
        #get model info from user
        model_type, data_shape, training_script_path = self.get_model_info()
        #read the training script
        training_script = file_reader(training_script_path)
        print("Training script read successfully.")
        print("Generating inference budle...")
        #generate the inference script
        raw_inference_script = self.actor_critic_inference_script(training_script, model_type, data_shape)
        # extract the inference script from the LLM response
        response = self.extract_and_save_inference_files(raw_inference_script)
        print(response)
    
    #generating the requirements.txt file
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


    def generate_requirements_with_llm(self):
        print("Generating requirements.txt...")
        inference_script_path = os.path.join(self.inference_dir, "inference.py")
        inference_script = file_reader(inference_script_path)
        libraries_str = self.extract_imported_libraries(inference_script)
        messages = [
                {"role": "system", "content": "You are an expert Python package manager."},
                {"role": "user", "content": f"""
                for the inference file you generated earlier, I need to create a `requirements.txt` file.
                ### Task:
                - Generate a `requirements.txt` file based on the following libraries used in the script:
                - Also include known libraries which are not in the script but are commonly used for model inference.
                - Ensure the libraries are listed in a format suitable for `pip install`.
                {libraries_str}
                - Python version is {self.python_version}.
                - To check which version to install refer to the" this list of all installed packages in the environment: {self.packages}
        """}
            ]

        return self.inference_llm.invoke(messages)
    
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
        extracted_script = file_reader(os.path.join(self.inference_dir, 'inference.py'))
        requirements_txt = self.generate_requirements_with_llm()
        requirements_txt_content = self.extract_requirements_txt(requirements_txt)
        requirements_txt_path = os.path.join(self.inference_dir, 'requirements.txt')
        write_to_file(requirements_txt_path, requirements_txt_content)


    
    #dockerization
    def generate_dockerfile(self,feedback=None):
        docker_template = file_reader(self.template_dockerfile_path)
        if feedback is None:
            messages = [
                {
                    "role": "system",
                    "content": "You are a Docker expert."
                },
                {
                    "role": "user",
                    "content": f'''
            You are an expert Dockerfile Generation Agent. Your task is to generate a **Dockerfile** for a containerized an inference script and its artifacts for AWS SageMaker Endpoint.

            
            ### Context:
            - **Working Directory**: The Dockerfile is located inside the project directory where the model inference script, requirements file, and a serve script are located.
            - **Python Version**: {self.python_version}
            - I am providing you a template for the Dockerfile, you can use it as a reference.
            - {docker_template}


            ### Additional Constraints:
            - **Do not use absolute paths.**  
            - **Assume all files are within the same directory as the Dockerfile when running `docker build .`.**
            - **Ensure the COPY commands properly reflect this.**
            
            ### Expected Output:
            Provide a complete **Dockerfile** as a code block with Dockerfile syntax highlighting.
            No need to include entrypoint or command for copying the model file
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

    
    def get_image_information(self):
        """Get image information from the user."""
        self.image_name = input("Enter the name of the docker image: ").strip()
        model_dir = input("Enter the directory where your model file is located: ").strip()
        self.local_image_name = self.image_name
        return {
            "image_name": self.image_name,
            "model_dir": model_dir
        }
    
    def docker_validation(self):
        print("Dockerfile validation")
        options = self.get_image_information()
        msg = f"""Now I want to validate the dockerfile you generated. 
        I need you help to provide me instrustion on 
        1. How to build the docker image, with the name {options['image_name']}, 
        2. How to run the docker image with 
            - by mountinng my model file present inside {options['model_dir']} to /opt/ml/model
            - providing entrypoint as the serve script
        3. Once that is done, I want to test the docker image by running a curl command to the /ping endpoint
        4. I also want to test the /invocations endpoint by providing a sample input
        5. If I face any issues, I will ask you for help
        Please provide me the instructions step by step."""
        while True:
            response = self.docker_generator_llm.invoke(msg)
            print("AI", response)
            input_text = input("Human (gg to exit): ")
            print("Human: ", input_text)
            if input_text.lower() == "gg":
                break
    
    def get_aws_ecr_information(self):
        """Get ECR information from the user."""
        print("To push the Docker image to AWS ECR, please provide the following information:")
        self.ecr_repo_name = input("Enter the name of the ECR repository: ").strip()
        self.aws_account_id = input("Enter your AWS account ID: ").strip()
        self.aws_region = input("Enter the AWS region for ECR: ").strip()        
        self.ecr_image_full_name = f"{self.aws_account_id}.dkr.ecr.{self.aws_region}.amazonaws.com/{self.ecr_repo_name}:latest"
        
    def provide_commmands_to_push_ecr(self):
        self.get_aws_ecr_information()
        options = {
            "account_id": self.aws_account_id,
            "region": self.aws_region,
            "ecr_repository_name": self.ecr_repo_name
        }
        print("Make sure you have the AWS CLI installed and configured with the necessary permissions.")
        print("Make sure ou have an ECR repository created. In the same region as your Sagemaker")
        print("Make sure you have Docker installed and running.")
        print("Login to ECR")
        print(f"aws ecr get-login-password --region {options['region']} | docker login --username AWS --password-stdin {options['account_id']}.dkr.ecr.{options['region']}.amazonaws.com")
        print("Tagging the locally build image for ECR")
        print(f"docker tag {self.local_image_name}:latest {options['account_id']}.dkr.ecr.{options['region']}.amazonaws.com/{options['ecr_repository_name']}:latest")
        print("Pushing the image to ECR")
        print(f"docker push {options['account_id']}.dkr.ecr.{options['region']}.amazonaws.com/{options['ecr_repository_name']}:latest")
    
    def dockerizing_the_model(self):
        print("Generating Dockerfile...")
        raw_dockerfile = self.generate_dockerfile()
        dockerfile = self.extract_dockerfile(raw_dockerfile)
        dockerfile_path = os.path.join(self.inference_dir, 'Dockerfile')
        print('Saving Dockerfile to:', dockerfile_path)
        write_to_file(dockerfile_path, dockerfile)
        print("Dockerfile generated successfully.")
        self.docker_validation()
        #pushing the docker image to ECR
        self.provide_commmands_to_push_ecr()
    

    #uploading the model to AWS S3
    def get_s3_information(self):
        """Get S3 bucket information from the user."""
        print("To upload the model to AWS S3, please provide the following information:")
        model_path = input("Enter the full path of the model file: ").strip()
        bucket_name = input("Enter the name of the S3 bucket: ").strip()
        s3_key = input("Enter the S3 key (path) for the model file: ").strip()
        return model_path,bucket_name, s3_key
    
    def compress_and_upload_to_s3(self,model_path, bucket_name, s3_key):
        """
        Compress a model file to tar.gz and upload it to an S3 bucket.

        Args:
            model_path (str): Path to the model file or directory.
            bucket_name (str): Name of the S3 bucket.
            s3_key (str): Key (path) in the S3 bucket where the file will be uploaded.

        Returns:
            str: The S3 URI of the uploaded file.
        """
        print("Uploading model to S3...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The specified model path does not exist: {model_path}")
        
        # Compress the model to a tar.gz file
        compressed_file = f"{os.path.splitext(model_path)[0]}.tar.gz"
        try:
            subprocess.run(['tar', '-czf', compressed_file, '-C', os.path.dirname(model_path), os.path.basename(model_path)],
                        check=True)
            print(f"Model compressed to: {compressed_file}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error compressing the model: {e}")
        
        # Upload the compressed file to S3
        try:
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            subprocess.run(['aws', 's3', 'cp', compressed_file, s3_uri], check=True)
            print(f"File uploaded to S3: {s3_uri}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error uploading the file to S3: {e}")
        finally:
            # Clean up the compressed file
            if os.path.exists(compressed_file):
                os.remove(compressed_file)
                print(f"Temporary file removed: {compressed_file}")
        
        return s3_uri
    
    def upload_model_to_s3(self):
        model_path, bucket_name, s3_key = self.get_s3_information()
        self.s3_model_uri = self.compress_and_upload_to_s3(model_path, bucket_name, s3_key)
        print(f"Model uploaded to S3 at: {self.s3_model_uri}")
    
    #terraforming the model
    def get_terraform_information(self):
        """Get Terraform information from the user."""
        print("To deploy the model on AWS SageMaker Endpoint, please provide the following information:")
        self.terraform_dir = input("Enter the directory where you want to save the Terraform files: ").strip()
        model_name = input("Enter the name of the model: ").strip()
        instance_type = input("Enter the instance type (e.g., ml.t2.medium): ").strip()
        return {
            "model_name": model_name,
            "instance_type": instance_type,
        }

    def terraform_generator_actor(self,options, feedback=None):
        terraform_template = file_reader(self.template_main_tf_path)
        if feedback is None:
            messages = [
                {
                    "role": "system",
                    "content": "You are a Terraform expert."
                },
                {
                    "role": "user",
                    "content": f'''
            You are an expert Terraform Generation Agent. Your task is to generate a **Terraform** script for deploying a model on AWS SageMaker Endpoint.
            ### Context:
            I need a terraform file to deploy a model on AWS SageMaker Endpoint. The terraform code should be able to deploy the following:
            1. An IAM role with the necessary permissions for SageMaker and S3, this role will be be used by the model.
            2. A SageMaker model with name {options['model_name']} that uses the ECR image {options['ecr_image']} and S3 URI {options['model_in_s3']}.
            3. An endpoint configuration for the SageMaker model.
            4. An endpoint for the SageMaker model.
            5. For IAM role and its polcies, use what is there in the template. You have to give it a name that is similar to the model name.
            6. Use the instance type {options['instance_type']} for the endpoint.
            7. Use the region {options['region']} for the resources.
            8. Overwrite the values in the template with the values provided in the options.

            Here is a template for the terraform file, you can use it as a reference.
            {terraform_template}'''}
            ]
        else:
            messages = feedback
        response = self.terrraform_generator_llm.invoke(messages)
        return response

    def terraform_critic_evaluator(self,response):
        messages = [
            {"role": "system", "content": "You are a Terraform expert."},
            {"role": "user", "content": f"""
            You are reviewing a terraform script generated for deploying a model on AWS SageMaker Endpoint.

            ### Task:
            - Analyze the script for correctness, best practices, and completeness.
            - Identify any missing components or improvements.
            - If the script is perfect, respond with 'No changes needed.'
            - If changes are required, specify what needs to be improved.
            - Dont menion the IAM role, it is already created and will be used by the model.
            

            ### Script to review:
            ```
            {response}
            ```

            ### Expected Output:
            - Either 'No changes needed.' OR a detailed improvement plan.
            """}
        ]
        
        return response.invoke(messages)

    def actor_critic_terraform_script(self,options):
            """
            Runs the actor-critic loop:
            - Actor generates a terraform script.
            - Critic evaluates the script.
            - If the script is sufficient, exit early.
            - Otherwise, Actor refines the script based on feedback.
            """
            feedback = None  # No feedback initially

            for _ in range(1):  # Max iterations: 1
                # Step 1: Actor generates inference script
                terraform_script = self.terraform_generator_actor(options, feedback)

                # Step 2: Critic evaluates the script
                # critic_feedback = terraform_critic_evaluator(terraform_script)

                # if "no changes needed" in critic_feedback.lower():
                #     print("✅ Terraform script is satisfactory. Exiting early.")
                #     return terraform_script  # Early exit if script is sufficient

                # # Step 3: Actor refines script using critic's feedback
                # feedback = [
                #     {"role": "assistant", "content": terraform_script},
                #     {"role": "user", "content": f"Revise based on this feedback:\n{critic_feedback}"}
                # ]

            print("🔄 Max iterations reached. Returning final script.")
            return terraform_script

    def extract_and_save_terraform_files(self,response_str: str, save_dir: str = ''):
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
            1. Extract all terraform code from the response string.
            2. For each extracted script:
                 - save it with appropriate name and extension like main.tf, etc. at {self.terraform_dir}
            3. Before saving them kindly check if the file are syntactically valid.
            4. Use the `save_file_tool` tool to save the content.
            5. Do not output anything else except confirming tool usage.

            ### Text:
            {response_str}
            """
        )
        return response
    
    def instructions_to_run_terraform(self):
        message = f"Can you provide me the instructions to run the terraform script you generated? I want to know the commands to run in order to deploy the model on AWS SageMaker Endpoint. I also want to know how to destroy the resources once I am done."
        response= self.terrraform_generator_llm.invoke(message)
        print(response)

    def orchestrate_terraform_deployment(self):
        # Get Terraform information from the user
        options = self.get_terraform_information()
        options["model_in_s3"] = self.s3_model_uri
        options["ecr_image"] = self.ecr_image_full_name
        options["region"] = self.aws_region
        # Generate the Terraform script using actor-critic approach
        terraform_script = self.actor_critic_terraform_script(options)
        # Extract and save the Terraform script to a file
        self.extract_and_save_terraform_files(terraform_script)
        # Provide instructions to run the Terraform script
        self.instructions_to_run_terraform()

    def deploy_to_sagemaker(self):
        # Step 1: Get model information and generate inference script
        self.get_inferece_script()
        # Step 2: Generate requirements.txt
        self.get_requirements_txt()
        # Step 3: Dockerize the model
        self.dockerizing_the_model()
        # Step 4: Upload the model to S3
        self.upload_model_to_s3()
        # Step 5: Terraform the model
        self.orchestrate_terraform_deployment()
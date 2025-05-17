from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from pydantic import BaseModel
import os
import subprocess
import re
from utils import file_reader, write_to_file
from talk_openai import MyOpenAI
from jinja2 import Template

class SaveFileInput(BaseModel):
    file_path: str
    content: str

@tool(args_schema=SaveFileInput)
def save_file_tool(file_path: str, content: str) -> str:
    """
    Save content to a file at the specified path.

    Args:
        file_path (str): The path where the file will be saved.
        content (str): The content to write into the file.

    Returns:
        str: Confirmation message with the file path.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)
    return f"Saved to {file_path}"


class DeployKubernetes:
    def __init__(self, small_model="gpt-4o-mini", large_model="gpt-4o", actor_critic_iterations=1):
        self.small_model = small_model
        self.large_model = large_model
        self.actor_critic_iterations = actor_critic_iterations
        self.inference_llm = MyOpenAI(model=large_model)
        self.validator_llm = MyOpenAI(model=small_model)
        self.requirements_llm = MyOpenAI(model=small_model)
        self.docker_llm = MyOpenAI(model=small_model)
        self.yaml_generator_llm = ChatOpenAI(model=large_model)
        self.inference_dir = None
        self.k8s_yaml_dir = None
        self.k8s_template_dir = "/Users/neel/Developer/deploy_wizard/templates/k8s"
        self.templated_deployment_path = os.path.join(self.k8s_template_dir, "deployment.yaml")
        self.templated_service_path = os.path.join(self.k8s_template_dir, "service.yaml")

    def get_model_info(self):
        print("\n📦 Let's collect your model information:")
        model_type = input("Model type (e.g., Classification, Regression): ").strip()
        data_shape = input("Input data shape (e.g., (1, 4)): ").strip()
        training_script_path = input("Full path to training script: ").strip()
        self.inference_dir = input("Directory to save inference script and Dockerfile: ").strip()
        return model_type, data_shape, training_script_path

    def generate_inference_script_with_llm(self, training_script, model_type, data_shape, feedback=None):
        if feedback is None:
            messages = [
                {"role": "system", "content": "You are an expert Python API developer."},
                {"role": "user", "content": f'''
Given this training script:
\"\"\"
{training_script}
\"\"\"
Model type: {model_type}  
Input shape: {data_shape}  

Write a complete FastAPI app that:
- Loads the trained model
- Accepts input as JSON
- Returns prediction as JSON
- Uses `/predict` endpoint
                '''}
            ]
        else:
            messages = feedback
        return self.inference_llm.invoke(messages)

    def evaluate_inference_script_with_llm(self, inference_script):
        messages = [
            {"role": "system", "content": "You are a Python reviewer."},
            {"role": "user", "content": f"Review this script:\n```\n{inference_script}\n```"}
        ]
        return self.validator_llm.invoke(messages)

    def actor_critic_inference_script(self, training_script, model_type, data_shape):
        feedback = None
        for _ in range(self.actor_critic_iterations):
            script = self.generate_inference_script_with_llm(training_script, model_type, data_shape, feedback)
            review = self.evaluate_inference_script_with_llm(script)
            if "no changes needed" in review.lower():
                return script
            feedback = [
                {"role": "assistant", "content": script},
                {"role": "user", "content": f"Revise based on this feedback:\n{review}"}
            ]
        return script

    def extract_python_script(self, llm_response):
        match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
        return match.group(1) if match else llm_response.strip()

    def get_inference_script(self):
        model_type, data_shape, training_script_path = self.get_model_info()
        training_script = file_reader(training_script_path)
        raw_response = self.actor_critic_inference_script(training_script, model_type, data_shape)
        script = self.extract_python_script(raw_response)
        path = os.path.join(self.inference_dir, "inference.py")
        write_to_file(path, script)

    def extract_imported_libraries(self, script):
        matches = re.findall(r"^\s*(?:import|from)\s+([\w\d_\.]+)", script, re.MULTILINE)
        return list(set(matches))

    def generate_requirements_txt(self):
        script_path = os.path.join(self.inference_dir, "inference.py")
        script = file_reader(script_path)
        imported_libs = self.extract_imported_libraries(script)
        libraries = ", ".join(imported_libs)

        messages = [
            {"role": "system", "content": "You are a Python dependency resolver."},
            {"role": "user", "content": f"""
    Generate a `requirements.txt` file based on these imports:
    {libraries}

    - Include all common ML & API runtime libraries even if not directly imported.
    - DO NOT return markdown formatting (no ```).
    Just return plain requirements.txt format.
            """}
        ]

        response = self.requirements_llm.invoke(messages)

        # Step 1: Extract plain text (remove ``` if present)
        raw_lines = response.strip().splitlines()
        clean_lines = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("```")]

        # Step 2: Ensure critical packages are included
        must_have = ["fastapi", "uvicorn", "scikit-learn", "joblib", "pydantic","jinja2"]
        for lib in must_have:
            if not any(lib in line for line in clean_lines):
                clean_lines.append(lib)

        final_reqs = "\n".join(clean_lines)
        path = os.path.join(self.inference_dir, "requirements.txt")
        write_to_file(path, final_reqs)


    def generate_dockerfile(self):
        script = file_reader(os.path.join(self.inference_dir, "inference.py"))
        reqs = file_reader(os.path.join(self.inference_dir, "requirements.txt"))
        model_path = input(f"Enter the model path (relative to {self.inference_dir}): ").strip()

        messages = [
            {"role": "system", "content": "You are a Dockerfile generation assistant."},
            {"role": "user", "content": f'''
    Create a Dockerfile for a FastAPI app with:
    - `inference.py` as the app
    - `requirements.txt` containing:
    {reqs}
    - `model.pkl` located at: {model_path}

    Use best practices: python:3.9-slim, install deps, copy code and model, expose 8000, run via uvicorn.
    Output only the Dockerfile contents, no explanation or markdown.
            '''}
        ]

        response = self.docker_llm.invoke(messages)

        # Clean output: extract only Dockerfile instructions
        valid_instructions = ["FROM", "RUN", "COPY", "CMD", "ENV", "EXPOSE", "WORKDIR", "ENTRYPOINT"]
        dockerfile_lines = []

        for line in response.splitlines():
            line = line.strip()
            if any(line.startswith(cmd) for cmd in valid_instructions):
                dockerfile_lines.append(line)

        dockerfile_cleaned = "\n".join(dockerfile_lines)
        path = os.path.join(self.inference_dir, "Dockerfile")
        write_to_file(path, dockerfile_cleaned)


    def dockerfile_testing(self):
        print("➡️ You can now build your Docker image:")
        print(f"docker build -t yourdockerhub/your-image-name:latest {self.inference_dir}")
        input("Press Enter when the image is built and tested locally...")

    def push_to_dockerhub(self):
        print("➡️ Now push your Docker image to DockerHub:")
        print("docker push yourdockerhub/your-image-name:latest")
        input("Press Enter after the image is pushed...")

    def get_k8s_parameters(self):
        print("\n📦 Let's collect your Kubernetes deployment details:")
        params = {}
        params["app_name"] = input("Application name (e.g., iris-inference-api): ").strip()
        params["docker_image"] = input("Docker image (e.g., user/image:tag): ").strip()
        params["container_port"] = input("Container port (e.g., 8000): ").strip()
        params["replicas"] = input("Number of replicas (e.g., 1): ").strip()
        self.k8s_yaml_dir = input("Directory to save Kubernetes YAMLs (e.g., ./k8s_files): ").strip()
        print("✅ Parameters collected.\n")
        return params

    def generate_k8s_yamls_from_templates(self, params):
        os.makedirs(self.k8s_yaml_dir, exist_ok=True)

        variables = {
            "app_name": params["app_name"],
            "docker_image": params["docker_image"],
            "container_port": params["container_port"],
            "replicas": params["replicas"]
        }

        deployment_yaml = self.render_template(self.templated_deployment_path, variables)
        service_yaml = self.render_template(self.templated_deployment_path, variables)

        write_to_file(os.path.join(self.k8s_yaml_dir, 'deployment.yaml'), deployment_yaml)
        write_to_file(os.path.join(self.k8s_yaml_dir, 'service.yaml'), service_yaml)

        print("✅ Kubernetes YAMLs generated using templates.\n")

    def render_template(self, template_path, variables):
        with open(template_path, 'r') as file:
            content = file.read()
        template = Template(content)
        return template.render(**variables)
    
    def extract_and_save_yaml_files(self, llm_response: str):
            agent = initialize_agent(
                tools=[save_file_tool],
                llm=ChatOpenAI(model="gpt-4o-mini"),
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True,
            )
            return agent.invoke(
                f"""
        Extract all YAML blocks and save them into:
        - deployment.yaml
        - service.yaml
        Inside folder: {self.k8s_yaml_dir}

        Use save_file_tool to write the files.

        ### Input:
        {llm_response}
                    """
            )

    def apply_k8s_yamls(self):
        print(f"🚀 Applying Kubernetes YAMLs from {self.k8s_yaml_dir}...")
        try:
            subprocess.run(f"kubectl apply -f {self.k8s_yaml_dir}", shell=True, check=True)
            print("✅ Kubernetes resources created successfully!")
        except subprocess.CalledProcessError as e:
            print("❌ Error during kubectl apply.")
            print(e)

    def deploy_kubernetes(self):
        self.get_inference_script()
        self.generate_requirements_txt()
        self.generate_dockerfile()
        self.dockerfile_testing()
        self.push_to_dockerhub()
        params = self.get_k8s_parameters()
        self.generate_k8s_yamls_from_templates(params)
        self.apply_k8s_yamls()

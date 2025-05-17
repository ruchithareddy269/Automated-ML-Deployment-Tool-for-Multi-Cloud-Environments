# Import necessary modules from LangChain and supporting libraries
from langchain.chat_models import ChatOpenAI
from talk_openai import MyOpenAI
from langchain.schema import HumanMessage
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel
import os, re
from utils import file_reader, write_to_file
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI

# -----------------------------
# Tool to save files to disk
# -----------------------------
class SaveFileInput(BaseModel):
    file_path: str
    content: str

@tool(args_schema=SaveFileInput)
def save_file_tool(file_path: str, content: str) -> str:
    """Save content to a file at the given path."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)
    return f"Saved to {file_path}"

# -----------------------------------
# EC2-like Azure Deploy Wizard
# -----------------------------------
class DeployAzureVM:
    def __init__(self, small_model="gpt-4o-mini", large_model="gpt-4o", actor_critic_iterations=1):
        # LLMs for different generation/validation tasks
        self.small_model = small_model
        self.large_model = large_model
        self.actor_critic_iterations = actor_critic_iterations
        # Inference
        self.inference_script_generator_llm = MyOpenAI(model=large_model)
        self.inference_script_validator_llm = MyOpenAI(model=small_model)
        # Requirements
        self.requirements_generator_llm = MyOpenAI(model=small_model)
        # Model loading modifications
        self.miss_llm = MyOpenAI(model=small_model)
        # Docker
        self.docker_generator_llm = MyOpenAI(model=small_model)
        # Terraform
        self.terraform_generator_llm = MyOpenAI(model=large_model)
        self.terraform_validator_llm = MyOpenAI(model=small_model)
        # Directories
        self.inference_dir = None
        self.terraform_dir = None
        #path to templates
        self.template_dir = '/Users/neel/Developer/deploy_wizard/templates/azure_vm'
        self.main_tf_template_path = os.path.join(self.template_dir,'main.tf')
        self.vars_tf_template_path = os.path.join(self.template_dir,'variables.tf')

    # -----------------------------
    # 1. Inference Script Methods
    # -----------------------------
    def get_model_info(self):
        print("To determine input shape: e.g., Pandas df.iloc[0].shape or numpy data[0].shape.")
        model_type = input("Enter model type (Binary Classification, Multiclass Classification, Regression): ").strip()
        data_shape = input("Enter expected shape of a single row of input data (e.g., (1,10)): ").strip()
        training_script_path = input("Enter full path of training script: ").strip()
        self.inference_dir = input("Enter directory to save inference artifacts: ").strip()
        return model_type, data_shape, training_script_path

    def generate_inference_script_with_llm(self, training_script, model_type, data_shape, feedback=None):
        if feedback is None:
            messages = [
                {"role":"system","content":"You are an expert Python developer."},
                {"role":"user","content":f"""
You are to generate an inference FastAPI script for a trained ML model.

Training script:\n{training_script}
- Model type: {model_type}
- Input shape: {data_shape}

Write a standalone Python script that:
- Loads the trained model
- Exposes an API via FastAPI to accept, preprocess, predict, and return JSON
"""}
            ]
        else:
            messages = feedback
        return self.inference_script_generator_llm.invoke(messages)

    def evaluate_inference_script_with_llm(self, inference_script):
        messages = [
            {"role":"system","content":"You are a code reviewer and Python API expert."},
            {"role":"user","content":f"""
Review this inference script for best practices and completeness. Return 'No changes needed.' or detailed improvements.
```python
{inference_script}
```
"""}
        ]
        return self.inference_script_validator_llm.invoke(messages)

    def actor_critic_inference_script(self, training_script, model_type, data_shape):
        feedback = None
        for _ in range(self.actor_critic_iterations):
            raw = self.generate_inference_script_with_llm(training_script, model_type, data_shape, feedback)
            inference_script = self.extract_python_script(raw)
            critique = self.evaluate_inference_script_with_llm(inference_script)
            if "no changes needed" in critique.lower():
                print("✅ Inference script satisfactory.")
                return inference_script
            feedback = [
                {"role":"assistant","content":inference_script},
                {"role":"user","content":"Revise based on feedback:\n" + critique}
            ]
        print(" Max inference iterations reached.")
        return inference_script

    def extract_python_script(self, llm_response):
        match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
        return match.group(1).strip() if match else llm_response

    def get_inference_script(self):
        model_type, data_shape, training_script_path = self.get_model_info()
        training_script = file_reader(training_script_path)
        script = self.actor_critic_inference_script(training_script, model_type, data_shape)
        path = os.path.join(self.inference_dir, 'inference.py')
        write_to_file(path, script)

    # -----------------------------
    # 2. Requirements.txt Methods
    # -----------------------------
    def extract_imported_libraries(self, script: str):
        libs = re.findall(r"^(?:import|from)\s+([\w\.]+)", script, re.MULTILINE)
        return list(set(libs))

    def generate_requirements_with_llm(self, inference_script):
        libs = self.extract_imported_libraries(inference_script)
        messages = [
            {"role":"system","content":"You are a Python dependency expert."},
            {"role":"user","content":"Generate a requirements.txt for these imports and common inference libs: " + ", ".join(libs)}
        ]
        return self.requirements_generator_llm.invoke(messages)

    def extract_requirements_txt(self, llm_response):
        match = re.search(r"```(?:txt)?\n(.*?)```", llm_response, re.DOTALL)
        return match.group(1).strip() if match else llm_response

    def get_requirements_txt(self):
        script = file_reader(os.path.join(self.inference_dir, 'inference.py'))
        raw = self.generate_requirements_with_llm(script)
        req_txt = self.extract_requirements_txt(raw)
        path = os.path.join(self.inference_dir, 'requirements.txt')
        write_to_file(path, req_txt)

    # -----------------------------
    # 3. Dockerfile Methods
    # -----------------------------
    def generate_dockerfile(self, options, feedback=None):
        if feedback is None:
            messages = [
                {"role":"system","content":"You are a Dockerfile expert."},
                {"role":"user","content":"Only output the Dockerfile itself (no explanations). Use python:3.9-slim, WORKDIR /app, install from requirements.txt, copy inference.py and your model file, expose port {options['model_port']}, and CMD [\"python\",\"inference.py\"]."}
            ]
        else:
            messages = feedback
        return self.docker_generator_llm.invoke(messages)

    def extract_dockerfile(self, llm_response):
        match = re.search(r"```Dockerfile\n(.*?)```", llm_response, re.DOTALL)
        return match.group(1).strip() if match else llm_response

    def dockerfile_testing(self):
        print("Run: docker build -t model-image .")
        print("Run: docker run -p ${var.model_port}:${var.model_port} model-image")

    def push_to_dockerhub(self):
        print("docker tag model-image user/repo:tag")
        print("docker push user/repo:tag")

    def dockerizing_the_model(self):
        options = {
            'inference_script_path': os.path.join(self.inference_dir, 'inference.py'),
            'inference_script_content': file_reader(os.path.join(self.inference_dir, 'inference.py')),
            'requirements_txt_path': os.path.join(self.inference_dir, 'requirements.txt'),
            'requirements_txt_content': file_reader(os.path.join(self.inference_dir, 'requirements.txt'))
        }
        options['model_path'] = input("Enter model file path in inference dir: ").strip()
        options['model_port'] = input("Enter model port (default from template): ")

        # 1) Generate the raw LLM response (which may include commentary + code fence)
        raw = self.generate_dockerfile(options)

        # 2) Extract *only* the Dockerfile code block
        match = re.search(r"```dockerfile\n(.*?)```", raw, re.DOTALL)
        dockerfile = match.group(1).strip() if match else raw

        # 3) Save just the Dockerfile instructions
        write_to_file(os.path.join(self.inference_dir, 'Dockerfile'), dockerfile)

        # 4) Continue with testing & push
        self.dockerfile_testing()
        self.push_to_dockerhub()


    # ---------------------------------
    # 4. Azure Terraform Methods
    def generate_azure_terraform_files(self, params: dict, feedback=None) -> str:
        if not feedback:
            system_msg = "You are an expert Terraform and Azure infrastructure specialist."
            # 1) Read in the full template (so the LLM sees the custom_data block)
            #template_main_tf = file_reader('templates_azure_vm/main.tf')
            raw_main = file_reader(self.main_tf_template_path)
            # remove any "tags = var.tags" from subnets (subnets cannot be tagged)
            template_main_tf = re.sub(r'\n\s*tags\s*=\s*var\.tags', '', raw_main)
            template_vars_tf = file_reader(self.vars_tf_template_path)

            # 2) Show the raw templates
            user_msg = (
                "I have the following Terraform template files:\n\n"
                "### main.tf template:\n"
                "```hcl\n"
                f"{template_main_tf}"
                "\n```\n\n"
                "### variables.tf template:\n"
                "```hcl\n"
                f"{template_vars_tf}"
                "\n```\n\n"
                "Please replace **only** the `var.*` placeholders in these templates with the values below, "
                "and preserve **every other line exactly as-is**, including the entire `custom_data` heredoc.\n\n"
                "#### Values to inject:\n"
            )

            # 3) List concrete values
            for k, v in params['values'].items():
                user_msg += f"- {k}: {v}\n"

            # 4) Strict rules to avoid past errors
            user_msg += (
                "\n## Additional Notes:\n"
                "- Do **NOT** remove or alter the `custom_data = base64encode(<<-EOF ... EOF)` block.\n"
                "- Do **NOT** emit any `tags` argument or block for `azurerm_subnet` (subnets can’t be tagged).\n"
                "- Do **NOT** hard-code port numbers anywhere: leave your SSH port (if any) as `var.ssh_port` and your model port as `var.model_port`.\n"
                "- Maintain valid HCL syntax and output exactly two code blocks (one for `main.tf`, one for `variables.tf`).\n"
                "- Follow AzureRM v3.x+ resource schemas as in the template.\n"
            )

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg}
            ]
        else:
            messages = feedback

        return self.terraform_generator_llm.invoke(messages)




    def evaluate_terraform(self, code: str) -> str:
        messages = [
            {"role":"system","content":"You are a Terraform Azure security expert."},
            {"role":"user","content":f"Review this Terraform code. Return 'No changes needed.' or list improvements.\n```hcl\n{code}\n```"}
        ]
        return self.terraform_validator_llm.invoke(messages)

    def actor_critic_terraform(self, params: dict) -> str:
        feedback = None
        tf_code = self.generate_azure_terraform_files(params, feedback)
        for _ in range(self.actor_critic_iterations):
            critique = self.evaluate_terraform(tf_code)
            if "no changes needed" in critique.lower():
                print("✅ Terraform code validated.")
                return tf_code
            feedback = [{"role":"assistant","content":tf_code},{"role":"user","content":"Revise based on feedback:\n"+critique}]
            tf_code = self.generate_azure_terraform_files(params, feedback)
        print("🔄 Max terraform iterations reached.")
        return tf_code

    def extract_and_save_hcl(self, response_str: str) -> None:
        agent = initialize_agent(tools=[save_file_tool], llm=ChatOpenAI(model="gpt-4o-mini"), agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
        agent.invoke(f"Extract HCL blocks for main.tf and variables.tf from text and save them under {self.terraform_dir}.\n\nText:\n{response_str}")

    def terraform_validation(self):
        msg = "Can you provide me the changes needed before running the Terraform script for Azure?"
        response = self.terraform_validator_llm.invoke(msg)
        print(response)

    def terraform_deployment_validator(self):
        msg = "Can you provide me instructions on how to deploy this Terraform script to Azure?"
        response = self.terraform_validator_llm.invoke(msg)
        print(response)
        while True:
            user = input("Enter 'gg' when done testing or input further questions: ")
            if user.lower() == 'gg':
                break
            response = self.terraform_validator_llm.invoke(user)
            print(response)

    def get_deployment_parameters(self):
        # Load your Terraform templates
        main_tf = file_reader(self.main_tf_template_path)
        vars_tf = file_reader(self.vars_tf_template_path)

        # Ask the user where to save the generated files
        self.terraform_dir = input("Enter the directory to save Terraform files: ").strip()

        # Prompt for each parameter
        answers = {
            "resource_group_name":   input("Resource Group Name: ").strip(),
            "location":              input("Azure Region (e.g. eastus): ").strip(),
            "virtual_network_name":  input("Virtual Network Name: ").strip(),
            "address_space":         input("VNet Address Space (e.g. 10.0.0.0/16): ").strip(),
            "subnet_name":           input("Subnet Name: ").strip(),
            "subnet_address_prefix": input("Subnet Address Prefix (e.g. 10.0.1.0/24): ").strip(),
            "nsg_name":              input("Network Security Group Name: ").strip(),
            "public_ip_name":        input("Public IP Name: ").strip(),
            "dns_label":             input("Public IP DNS Label: ").strip(),
            "nic_name":              input("Network Interface Name: ").strip(),
            "vm_name":               input("VM Name: ").strip(),
            "admin_username":        input("Admin Username: ").strip(),
            "admin_password":        input("Admin Password: ").strip(),
            "vm_size":               input("VM Size (e.g. Standard_B1ls): ").strip(),
            "docker_image":          "neel26d/iris_model_inference:latest",
            "container_name":        input("Container Name: ").strip(),
            "model_port":            int(input("Model Port (e.g. 8000): ").strip()),
            "image_publisher":       input("Image Publisher (e.g. Canonical): ").strip(),
            "image_offer":           input("Image Offer (e.g. UbuntuServer): ").strip(),
            "image_sku":             input("Image SKU (e.g. 18.04-LTS): ").strip(),
        }

        # Sanitize container_name: lowercase, hyphens only, max 63 chars
        raw = answers["container_name"]
        safe = re.sub(r'[^a-z0-9-]', '-', raw.lower())
        safe = re.sub(r'-+', '-', safe).strip('-')
        answers["container_name"] = safe[:63]

        return {
            "main_tf": main_tf,
            "vars_tf":  vars_tf,
            "values":   answers
        }


    def orchestrate_terraform_deployment(self):
        params = self.get_deployment_parameters()
        raw_tf = self.actor_critic_terraform(params)
        self.extract_and_save_hcl(raw_tf)
        self.terraform_validation()
        self.terraform_deployment_validator()

    # -----------------------------
    # 5. Top-level Deploy Method
    # -----------------------------
    def deploy_azure_vm(self):
        # Step 1: Generate inference script
        self.get_inference_script()
        # Step 2: Create requirements.txt
        self.get_requirements_txt()
        # Step 3: Dockerize model
        self.dockerizing_the_model()
        # Step 4: Azure Terraform deployment
        self.orchestrate_terraform_deployment()

# Example usage:
# if __name__ == '__main__':
#     wizard = DeployAzureVM()
#     wizard.deploy_azure_vm()

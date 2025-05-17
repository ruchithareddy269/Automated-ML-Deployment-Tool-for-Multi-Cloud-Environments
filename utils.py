import os
import subprocess
import sys

def read_openai_key(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        return str(e)

def file_reader(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError("File not found. Please provide a valid path.")
    try:
        with open(input_file, 'r') as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

#get python version
def get_python_version():
    try:
        # Check if python3 is available
        return subprocess.check_output([sys.executable, '--version']).decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        # python3 is not installed
        print(f"Error: {e}")
        return None
    
#get all packages installed in the system
def get_installed_packages():
    try:
        # Get the list of installed packages
        result = subprocess.check_output(['pip', 'freeze']).decode('utf-8').strip()
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

def get_file_path():
    """Prompt the user for the training Python file path."""
    file_path = input("Enter the path to the training Python file: ").strip()
    return file_path

def write_to_file(file_path, content):
    """Write the content to the specified file."""
    #create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(content)
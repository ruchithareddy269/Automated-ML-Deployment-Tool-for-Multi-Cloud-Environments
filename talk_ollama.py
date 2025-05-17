from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class Ollama:
    def __init__(self, model_name):
        self.llm = ChatOllama(model="gemma2:2b",temperature=0)
        self.messages = [("system",'Your are an expert DevOps Engineer')]
    
    def invoke(self, message):
        self.messages.append(('human', message))
        response = self.llm.invoke(message)
        self.messages.append(('ollama', response.content))
        return response.content


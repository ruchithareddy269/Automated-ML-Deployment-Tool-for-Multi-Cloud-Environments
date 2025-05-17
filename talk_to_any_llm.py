from abc import ABC, abstractmethod

class Talker(ABC):
    @abstractmethod
    def __init__(self,model):
        pass
    
    @abstractmethod
    def invoke(self, message):
        pass    

    def read_file(self, file_name):
        try:
            with open(file_name, 'r') as file:
                return file.read().strip()
        except Exception as e:
            return str(e)
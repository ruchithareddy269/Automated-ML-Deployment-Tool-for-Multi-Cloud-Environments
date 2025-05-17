from talk_to_any_llm import Talker
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import os

class MyOpenAI(Talker):
    FILE_WITH_SECRET = '/Users/neel/Documents/Keys/OpenAIDeployWizardKey.txt'
    
    def __init__(self, model="gpt-4o-mini"):
        os.environ["OPENAI_API_KEY"] = self.read_openai_key()
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.messages = []
    
    def read_openai_key(self):
        try:
            with open(self.FILE_WITH_SECRET, 'r') as file:
                return file.read().strip()
        except Exception as e:
            return str(e)
        
    def invoke(self, message):
        """
        Invokes the LLM with either a single message or a list of formatted messages.
        
        - If `message` is a `str`, it is treated as a user message and added to the conversation.
        - If `message` is a `list[dict]`, it is treated as a full message history and used directly.
        """

        if isinstance(message, str):  # Single user message
            human_message = HumanMessage(content=message)
            self.messages.append(human_message)
            response = self.llm.invoke(self.messages)  # Maintain context
        elif isinstance(message, list) and all(isinstance(m, dict) for m in message):  # List of messages
            self.messages = self._convert_messages(message)  # Reset context
            response = self.llm.invoke(self.messages)
        else:
            raise ValueError("Invalid message format. Must be a string or list of dicts.")

        ai_message = AIMessage(content=response.content)
        self.messages.append(ai_message)
        
        return response.content

    def _convert_messages(self, message_list):
        """ Converts a list of dicts into LangChain message objects. """
        converted_messages = []
        for msg in message_list:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                converted_messages.append(SystemMessage(content=content))
            elif role == "user":
                converted_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                converted_messages.append(AIMessage(content=content))
        return converted_messages
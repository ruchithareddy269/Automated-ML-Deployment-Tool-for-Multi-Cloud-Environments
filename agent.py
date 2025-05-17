from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import subprocess

class CodeAssistantAgent:
    def __init__(self):
        # Initialize memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key="", temperature=0)

        # Define tools
        self.tools = [
            Tool(name="Generate Code", func=self.generate_code, description="Generates new Python or Terraform code."),
            Tool(name="Modify Code", func=self.modify_code, description="Modifies existing code.")
        ]

        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            memory=self.memory,
            agent="chat-conversational-react-description"
        )

    def generate_code(self, tool_input: str) -> str:
        """Generate code based on user input."""
        prompt = f"""
                You are an expert assistant. Generate code with correct indentation and formatting.
                Request: {tool_input}
                Ensure the response is properly formatted for direct use in Python or Terraform.
                """
        return self.llm.invoke(prompt)

    def modify_code(self, tool_input: str) -> str:
        """Modify existing code based on user instructions."""
        prompt = f"Modify this code: {tool_input}"
        return self.llm.invoke(prompt)


    def process_input(self, user_input: str) -> str:
        """Process user input and delegate to the appropriate tool."""
        if "generate" in user_input.lower():
            return self.agent.invoke(f"Generate Code: {user_input}")
        elif "modify" in user_input.lower():
            return self.agent.invoke(f"Modify Code: {user_input}")
        else:
            return self.agent.invoke(user_input)

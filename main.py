from langchain.agents import AgentExecutor
from dotenv import load_dotenv
from agent import OpenAIFunctionsAgent

# Load environment variables from .env file
load_dotenv()

# Example question with two students
question = "What are the data for Ana and Bianca?"

# Create the agent
agent = OpenAIFunctionsAgent()

# Create the executor to run the agent with the tools
executor = AgentExecutor(
    agent=agent.agent,
    tools=agent.tools,
    verbose=True
)

# Run the agent and print the answer
response = executor.invoke({"input": question})
print(response['output'])
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
from agent import OpenAIFunctionsAgent

# Load environment variables from .env file
load_dotenv()

# Example questions
question = "Give me the data for Ana and Bianca."
question = "Make an academic profile for Ana"
question = "Compare the academic profiles of Ana and Bianca"
question = "Give the data for USP"
question = "Between USP and UFRJ, which one has the best option for Bianca?"

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
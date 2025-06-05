import openai
import json
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain.tools import BaseTool
from langchain import hub
from langchain.agents import Tool, create_openai_tools_agent, AgentExecutor
from pydantic import Field, BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the OpenAI API with Azure credentials from environment variables
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_type = os.getenv("AZURE_OPENAI_API_TYPE")

# Tool class to fetch student data from the CSV file
def get_student_data(student):
    # Read the CSV file containing student data
    data = pd.read_csv("docs/students.csv")
    # Filter the DataFrame for the requested student username
    student_data = data[data["USUARIO"] == student]
    if student_data.empty:
        return {}
    # Return the first matching record as a dictionary
    return student_data.iloc[:1].to_dict()

class StudentDataTool(BaseTool):
    name: str = "student_data"
    description: str = "Tool to get a student's data from the database."

    def _run(self, input: str) -> str:
        # Extract the student's name from the question (e.g., 'Ana' from 'What are the data for Ana?')
        student = input.lower()
        # Query the CSV for the student's data
        data = get_student_data(student)
        print("Found:")
        print(data)
        # Return the result as a JSON string
        return json.dumps(data, ensure_ascii=False)

    def _call(self, input: str) -> str:
        return self._run(input)

# Initialize the AzureChatOpenAI LLM with Azure credentials
llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1-mini",
        openai_api_version=openai.api_version,
        openai_api_key=openai.api_key,
        azure_endpoint=openai.api_base
    )

# Instantiate the tool and register it for the agent
student_data_tool = StudentDataTool()
tools = [
    Tool(name=student_data_tool.name,
         func=student_data_tool.run,
         description=student_data_tool.description)
]

# Load the agent prompt from the LangChain Hub
prompt = hub.pull("hwchase17/openai-functions-agent")
#print(prompt)

# Create the agent and executor for running tool-augmented LLM tasks
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

question = "What are the data for Ana?"
response = executor.invoke({"input": question})
print(response['output'])
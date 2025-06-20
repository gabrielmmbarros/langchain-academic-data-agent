from langchain_openai import AzureChatOpenAI
from langchain.agents import create_openai_tools_agent, create_react_agent
from langchain import hub
from langchain.agents import Tool
import os
from student import AcademicProfile, StudentDataTool
from university import UniversityDataTool, AllUniversitiesTool

class OpenAIFunctionsAgent:
    def __init__(self):
        # Create the LLM with Azure credentials
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4.1-mini",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )   

        # Create the tool for getting tools from student file
        student_data_tool = StudentDataTool()
        academic_profile = AcademicProfile()
        university_data = UniversityDataTool()
        all_universities = AllUniversitiesTool()

        self.tools = [
            Tool(
                name=student_data_tool.name,
                func=student_data_tool.run,
                description=student_data_tool.description
            ),
            Tool(
                name=academic_profile.name,
                func=academic_profile.run,
                description=academic_profile.description
            ),
            Tool(
                name=university_data.name,
                func=university_data.run,
                description=university_data.description
            ), 
            Tool(
                name=all_universities.name,
                func=all_universities.run,
                description=all_universities.description
            )
        ]

        # Load the OpenAI functions agent prompt from the LangChain Hub
        #prompt = hub.pull("hwchase17/openai-functions-agent")
        prompt = hub.pull("hwchase17/react")

        # Create the agent with the LLM, tools, and prompt
        #self.agent = create_openai_tools_agent(llm, self.tools, prompt)
        self.agent = create_react_agent(llm, self.tools, prompt)
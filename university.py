import json
import os
import pandas as pd
from typing import List
from langchain_openai import AzureChatOpenAI
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import Field, BaseModel

# Function to get data for a specific university from the CSV
def get_university_data(university: str):
    data = pd.read_csv("docs/universities.csv")
    data["NOME_FACULDADE"] = data["NOME_FACULDADE"].str.lower()
    # Find the row where the university name matches
    university_data = data[data["NOME_FACULDADE"] == university]
    if university_data.empty:
        return {}
    return university_data.iloc[:1].to_dict()

# Function to get data for all universities from the CSV
def get_all_universities_data():
    data = pd.read_csv("docs/universities.csv")
    return data.to_dict()

# Pydantic model for extracting university name
class UniversityExtractor(BaseModel):
    university: str = Field("The name of the university in lowercase.")

# Tool that uses the LLM to extract the university name and get their data
class UniversityDataTool(BaseTool):
    name: str = "UniversityDataTool"
    description: str = """
    Tool to get a university's data from the database.
    The argument of this tool is the name of the university.
    """

    def _run(self, input: str) -> str:
        # Create the LLM with Azure credentials
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4.1-mini",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # Set up the output parser to extract the university name
        parser = JsonOutputParser(pydantic_object=UniversityExtractor)

        # Prompt template for extracting university name
        template = PromptTemplate(
            template="""
                You have to analyze the following input and extract the name of the university, in lowercase.
                INPUT:
                --------------------
                {input}
                --------------------
                Output format:
                {output_format}
                """,
            input_variables=["input"],
            partial_variables={"output_format": parser.get_format_instructions()}
        )

        # Running the chain: prompt -> LLM -> parser
        chain = template | llm | parser

        response = chain.invoke({"input": input})
        university = response['university']

        # Get the university data from the CSV
        data = get_university_data(university)

        # Return the data as JSON
        return json.dumps(data)
    
# Tool to load all universities data
class AllUniversitiesTool(BaseTool):
    name: str = "AllUniversitiesTool"
    description: str = """Loads the data of all universities. No input parameter is required."""

    def _run(self, input: str):
        universities = get_all_universities_data()
        return universities
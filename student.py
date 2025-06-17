import json
import pandas as pd
import os
from langchain_openai import AzureChatOpenAI
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import Field, BaseModel
from typing import List

# Function to get student data from the CSV file and return the first match as a dictionary
def get_student_data(student):
    data = pd.read_csv("docs/students.csv")
    # Find the row where the username matches
    student_data = data[data["USUARIO"] == student]
    if student_data.empty:
        return {}
    return student_data.iloc[:1].to_dict()

# Model to help the LLM extract the student's name from the question
class StudentExtractor(BaseModel):
    student: str = Field("The student's username, exactly as in the USUARIO column, all lowercase, no accents or extra spaces. Example: john, carlos, joana, carla.")

# Tool that uses the LLM to extract the student name and get their data
class StudentDataTool(BaseTool):
    name: str = "StudentDataTool"
    description: str = """
    Tool to get a student's data from the database.
    The argument of this tool is the student's name.
    """

    def _run(self, input: str) -> str:
        # Create the LLM with Azure credentialsAdd commentMore actions
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4.1-mini",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # Set up the output parser to extract the student name
        parser = JsonOutputParser(pydantic_object=StudentExtractor)

        template = PromptTemplate(
            template="""
                You have to analyze the following input and extract the provided name, in lower case.
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
        student = response['student']
        student = student.lower().strip()

        # Get the student data from the CSV
        data = get_student_data(student)

        # Return the data as JSON
        return json.dumps(data)

# Represents a grade for a specific knowledge area
class Grade(BaseModel):
    area: str = Field("Name of the knowledge area")
    grade: float = Field("Grade in the knowledge area")

# Represents the academic profile of a student, including name, graduation year, grades, and a summary
class StudentAcademicProfile(BaseModel):
    name: str = Field("Student's name")
    graduation_year: int = Field("Year of graduation")
    grades: List[Grade] = Field("List of grades for subjects and knowledge areas")
    summary: str = Field("Summary of the main characteristics that make this student unique and a great potential candidate for universities. Example: only this student has ...")

# Tool that uses the LLM to generate an academic profile for a student based on their data
class AcademicProfile(BaseTool):
    name: str = "AcademicProfile"
    description: str = (
        """Creates an academic profile for a student."
        "This tool requires all student data as input.
        "I am unable to fetch the student's data.
        You need to fetch the student's data before calling me.
        """
    )

    def _run(self, input: str) -> str:
        # Create the LLM with Azure credentials
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4.1-mini",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # Set up the output parser for the academic profile
        parser = JsonOutputParser(pydantic_object=StudentAcademicProfile)

        # Prompt to instruct the LLM to create the academic profile
        template = PromptTemplate(
            template = """
            - Format the student data into an academic profile.
            - With the data, identify suggested university options and courses that match the student's interests.
            - Highlight the student's profile, focusing on what makes sense for the student's target institutions.

            Persona: You are a career consultant and need to give detailed, rich, but direct advice to the student about options and possible consequences.
            Current information:

            {student_data}
            {output_format}
            """,
            input_variables=["student_data"],
            partial_variables={"output_format": parser.get_format_instructions()}
        )

        # Run the chain: prompt -> LLM -> parser
        chain = template | llm | parser
        response = chain.invoke({"student_data": input})
        return response
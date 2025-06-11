import json
import pandas as pd
import os
from langchain_openai import AzureChatOpenAI
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import Field, BaseModel

# Function to get student data from the CSV file and return the first match as a dictionary
def get_student_data(student):
    data = pd.read_csv("docs/students.csv")
    # Find the row where the username matches
    student_data = data[data["USUARIO"] == student]
    print('Dados:', student_data)
    if student_data.empty:
        return {}
    return student_data.iloc[:1].to_dict()

# Model to help the LLM extract the student's name from the question
class StudentExtractor(BaseModel):
    student: str = Field("The student's username, exactly as in the USUARIO column, all lowercase, no accents or extra spaces. Example: john, carlos, joana, carla.")

# Tool that uses the LLM to extract the student name and get their data
class StudentDataTool(BaseTool):
    name: str = "StudentDataTool"
    description: str = "Tool to get a student's data from the database."

    def _run(self, input: str) -> str:
        # Create the LLM with Azure credentials
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4.1-mini",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # Set up the output parser to extract the student name
        parser = JsonOutputParser(pydantic_object=StudentExtractor)

        template = PromptTemplate(
            template="""You should analyze the {input} and extract the provided username.
                Output format:
                {output_format}""",
            input_variables=["input"],
            partial_variables={"output_format": parser.get_format_instructions()}
        )

        # Running the chain: prompt -> LLM -> parser
        chain = template | llm | parser
        response = chain.invoke({"input": input})
        print("[DEBUG] LLM extraction response:", response)  # Debug print
        student = response['student'].lower()

        # Get the student data from the CSV
        data = get_student_data(student)

        # Return the data as JSON
        return json.dumps(data)
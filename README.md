
## LangChain - Academic Data Agent

## 🔨 What does this project do?

This project is a set of assistants (agents) that help you find the best universities for different student profiles. We built it in Python using LangChain and OpenAI's language model (LLM). The main steps are:

1. Get user data
2. Create a user profile
3. Find universities that match the profile

There is also a hub to manage all the agents and tools.

![](img/amostra.gif)

## ✔️ Main Technologies Used

We used these main tools and techniques:

- Python Object-Oriented Programming (OOP)
- OpenAI GPT API
- LangChain chains
- OpenAI Agents
- ReAct Agents
- Reading and working with CSV files

## 🛠️ How to Run the Project

After downloading the project, open it in Visual Studio Code (or another code editor). Then, set up your environment:

### Create a virtual environment on Windows:

```bash
python -m venv venv-langchain2
venv-langchain2\Scripts\activate
```

### Create a virtual environment on Mac/Linux:

```bash
python3 -m venv venv-langchain2
source venv-langchain2/bin/activate
```

Next, install the required packages:

```bash
pip install -r requirements.txt
```

## 🔑 Set your API Key

You need an OpenAI API key. Create a file called `.env` in the project folder and add this line:

```
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your real OpenAI API key.

## 📁 Data Files

The project uses CSV files with student and university data. You can find them in the `docs/` folder:

- `students.csv`: Example student profiles
- `universities.csv`: Example university data

Feel free to add or edit data in these files to test different scenarios.


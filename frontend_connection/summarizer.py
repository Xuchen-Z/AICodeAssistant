import feedback as fb
import code_access as ca
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

MODEL_NAME = "codellama:7b"
model = Ollama(model=MODEL_NAME)

file_path = "../tests/error.py"
lines = ca.read_script(file_path)

summarizer = Agent(
    role = "Python code summarizer",
    goal = "Accurately summarize Python code based what the code does. give every function and method a breif summary within 300 words",
    backstory = "Your are an assistant whose only job is to summarize Python code accurately. \
    Your job is to help the user to better understand their Python code. \
    The Python code may have errors, but please provide a summary based on the overall code, making reasonable assumptions where needed.",
    verbose = False,
    allow_delegation = False,
    llm = model    
)

summarizer_task = Task(
    description=f"Write a concise paragraph summarizing the purpose and main functionality of the provided Python code: \n'{lines}'",
    agent=summarizer,
    expected_output=(
        "A breif summary that:\n"
        "1. States the code's overall purpose\n"
        "2. Uses plain English (no markdown)\n"
        "Keep it under 300 words and avoid technical jargon."
    )
)

summary = summarizer_task.execute()
print(summary)
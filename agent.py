# from langchain_community.llms import LlamaCpp
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

MODEL_NAME = "codellama:7b"

model = Ollama(model=MODEL_NAME)

# with open("your_file.txt", "r") as file:
#     lines = file.readlines()  # Reads file into a list of lines

lines = 'print("hello world")\n;\nprint("Hi");\nprint("Bye")'

summarizer = Agent(
    role = "Python code summarizer",
    goal = "Accurately summarize Python code based what the code does. give every function and method a breif summary within 300 words",
    backstory = "Your are an AI assistant whose only job is to summarize Python code accurately. \
    The Python code might contain errors, do not be afraid to point it out. \
    Your job is to help the user to better understand their Python code.",
    verbose = True,
    allow_delegation = False,
    llm = model    
)


debugger = Agent(
    role = "Python code debugger",
    goal = "Accurately debug and fix every error in Python code. \
    If any error was found, provide a copy of the original Python code that has all errors fixed. \
    If no error was found, then output 'no fixes were made' only without making changes to the code.",
    backstory = "Your are an AI assistant whose only job is to find and fix errors in Python code. \
    Don't be afraid to point out any errors that you have noticed. \
    There is a chance that the code is error free, in that case no fixes are required.",
    verbose = True,
    allow_delegation = False,
    llm = model    
)

debug_and_fix_errors = Task(
    description = f"Find and fix all errors in the Python code '{lines}'",
    agent = debugger,
    expected_output = "Output a copy of Python code with all the errors fixed, or 'no error was found' if the code is error free.",
)

defect_crew = Crew(
    agents = [debugger],
    tasks = [debug_and_fix_errors],
    verbose = 2,
    process = Process.sequential
)

output = defect_crew.kickoff()
print(output)
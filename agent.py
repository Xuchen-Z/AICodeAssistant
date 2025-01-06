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
    The Python code may have errors. Please provide a summary based on the overall code, making reasonable assumptions where needed. \
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

feedback_summarizer = Agent(
    role = "User feedback summarizer",
    goal = "Accurately summarize user feedback in a concise manner. \
    Ensure the summary captures the key points and sentiment of the feedback without omitting critical details. \
    If the feedback is already concise and clear, output the summary directly without modifications.",
    backstory = "Your are a summarizer of an AI assistant whose only job is to summarize the feedback received from user. \
    Always ensure that the summary is accurate and concise.",
    verbose = True,
    allow_delegation = False,
    llm = model
)

debug_and_fix_errors = Task(
    description = f"Find and fix all errors in the Python code: '{lines}'",
    agent = debugger,
    expected_output = "Output a copy of Python code with all the errors fixed, or 'no fixes were made' if the code is error free.",
)

user_feedback_summarize = Task(
    description = f"Summarize the feedback recieved by an AI coding assistant: '{lines}'",
    agent = feedback_summarizer,
    expected_output = "Output the concise sumarry of the feedback recieved.",
)

defect_crew = Crew(
    agents = [debugger],
    tasks = [debug_and_fix_errors],
    verbose = 2,
    process = Process.sequential
)

feedback_crew = Crew(
    agents = [feedback_summarizer],
    tasks = [user_feedback_summarize],
    verbose = 2,
    process = Process.sequential
)

output = defect_crew.kickoff()
print(output)
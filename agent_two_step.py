# from langchain_community.llms import LlamaCpp
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

MODEL_NAME = "codellama:7b"

model = Ollama(model=MODEL_NAME)
# response = model("Hi!")

# with open("your_file.txt", "r") as file:
#     lines = file.readlines()  # Reads file into a list of lines

lines = 'print("hello world")\n;\nprint("Hi")\n;\n'

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


classifier = Agent(
    role = "Python code debugger",
    goal = "Accurately locate every error in Python code. classifies each error into one of these types: syntax error, exception, \
    or no errors. Also provide a line number along with it if it is not 'no errors'.",
    backstory = "Your are an AI assistant whose only job is to locate errors in Python code. \
    Don't be afraid to point out any errors that you have noticed.",
    verbose = True,
    allow_delegation = False,
    llm = model    
)

fixer = Agent(
    role = "Python code fixer",
    goal = "Locate each errors and its cause by analyzing the definition of the error type provided, \
    and the code around the specified line number. Provide the fixed Python code that has no error. \
    If no error was found, then output 'no fixes were made' without making fixes to the code.",
    backstory = "Your are an AI assistant whose only job is to locate and provide a fix to the error based on the \
    type of the error and its line number. \
    Relate to the code around the line number provided to ensure the error was understand correctly. \
    Both error type and line number will be provided to you by the 'Python code debugger' agent. \
    There is a chance that there are 'no errors', in that case no fixes are required.",
    verbose = True,
    allow_delegation = False,
    llm = model    
)

classify_errors = Task(
    description = f"Locate and classfy errors in the Python code '{lines}'",
    agent = classifier,
    expected_output = "A line number to the error code and one of these options: 'syntax error', 'exception', or 'no errors'.",
)

fixes_to_errors = Task(
    description = f"Provide fixes to the errors in Python code '{lines}' based on the error types and the line numbers \
    of each error provided by the 'classifier' agent.",
    agent = fixer,
    expected_output = "fixes to the Python code based on the error types and the line numbers of each error \
    provided by the 'classifier' agent. If the error types is 'no errors', then output 'no fixes were made'.",
)

defect_crew = Crew(
    agents = [classifier, fixer],
    tasks = [classify_errors, fixes_to_errors],
    verbose = 2,
    process = Process.sequential
)

output = defect_crew.kickoff()
print(output)
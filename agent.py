import feedback as fb
# from langchain_community.llms import LlamaCpp
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

MODEL_NAME = "codellama:7b"

model = Ollama(model=MODEL_NAME)

# with open("your_file.txt", "r") as file:
#     lines = file.readlines()  # Reads file into a list of lines

lines = 'print("hello world")\n;\nprint("Hi");\nprint("Bye")'
feedback = 'Check the indentation in the code is correct and would not cause errors.'

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
    goal = "Thoroughly inspect the provided Python code, \
    identify and fix all syntax, runtime, and logical errors maintain the original code's functionality, \
    provide the corrected code as your output. \
    If any error was found, provide a copy of the original Python code that has all errors fixed. \
    If no error was found, then output 'no fixes were made' only without making changes to the code.",
    backstory = "Your are an expert code debugger whose only job is to find and fix errors in Python code. \
    Don't be afraid to point out any errors that you have noticed. \
    There is a chance that the code is error free, in that case no fixes are required.",
    verbose = True,
    allow_delegation = False,
    llm = model    
)

feedback_summarizer = Agent(
    role = "User feedback summarizer",
    goal = "Summarize user feedback into a single, clear imperative sentence that captures the main action requested or suggested. \
    Exclude any additional explanations or context.",
    backstory = "You are a concise feedback summarizer. \
    Your task is to convert user feedback into a direct, imperative statement that reflects the core suggestion or issue. \
    Remove any extra information and focus solely on the essential action.",
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
    description = f"Summarize the feedback: '{feedback}'",
    agent = feedback_summarizer,
    expected_output = "Output a concise sumarry of the feedback recieved without extra information or mentioning this is a feedback.",
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

# new_feedback = feedback_crew.kickoff()
# fb.init_feedback_log()
# json_str = fb.process_feedback(new_feedback)
# fb.save_feedback(json_str)

# print(new_feedback)
import feedback as fb
# from langchain_community.llms import LlamaCpp
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

MODEL_NAME = "codellama:7b"

model = Ollama(model=MODEL_NAME)

# with open("your_file.txt", "r") as file:
#     lines = file.readlines()  # Reads file into a list of lines

lines = 'print("hello world")\n;\nprint("Hi");\nprint("Bye")'
feedback = 'Remove all unnecessary information and explianations, keeping the Python code only.'

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
    goal = "Identify and fix all syntax, runtime, and logical errors in the provided Python code while maintaining its functionality. \
    It is possible that the code is error free, in that case output an exact copy of the code.\
    Otherwise output the corrected code without any extra comments or explainations.",
    backstory = "You are an expert code debugger whose job is to find and fix errors in Python code. \
    You never add extra comments, information, or explainations to your output.\
    You always provide the corrected code with fixes applied.",
    verbose = False,
    allow_delegation = False,
    llm = model    
)

feedback_summarizer = Agent(
    role = "User feedback summarizer",
    goal = "Summarize user feedback into a single, clear imperative sentence that captures the main action requested or suggested. \
    Exclude any additional explanations or context.",
    backstory = "You are a concise feedback summarizer. \
    Your task is to convert user feedback into a direct, imperative statement that reflects the core suggestion or issue. \
    Do not include any explanations or additional information.",
    verbose = False,
    allow_delegation = False,
    llm = model
)

adjust_output_agent = Agent(
    role="Output Adjuster",
    goal="Modify the output from a python code debugger to align with the user's requirements. \
    Make necessary adjustments only if accapable.",
    backstory="You are an expert in refining python code based on user requirements. \
    Your job is to adjust the provided python code base on user feedbacks while keeping the functionality and correctness of the code. \
    Provide code adjustments only when it is accapable. \
    Output the original code without adjustments if not accapable, or if the requirement is 'No requirement'.",
    verbose=False,
    allow_delegation=False,
    llm=model
)


# DEBUG TASK
debug_and_fix_errors = Task(
    description = f"Find and fix all errors in the Python code: '{lines}'",
    agent = debugger,
    expected_output = "Output a copy of the original Python code with all errors fixed. \
    No extra comments, information or explainations should be provided.",
)

defect_crew = Crew(
    agents = [debugger],
    tasks = [debug_and_fix_errors],
    verbose = 2,
    process = Process.sequential
)

# DEBUG TASK ONLY
# output = defect_crew.kickoff()
# print(output)



# FEEDBACK SUMMARIZE TASK
user_feedback_summarize = Task(
    description = f"Summarize the feedback: '{feedback}'",
    agent = feedback_summarizer,
    expected_output = "Output a concise sumarry of the feedback recieved without extra information or mentioning this is a feedback.",
)

feedback_crew = Crew(
    agents = [feedback_summarizer],
    tasks = [user_feedback_summarize],
    verbose = 2,
    process = Process.sequential
)

# ADD TO FEEDBACK LOG
# new_feedback = feedback_crew.kickoff()
# fb.init_feedback_log()
# json_str = fb.process_feedback(new_feedback)
# fb.save_feedback(json_str)
# print(new_feedback)


# REFINE RESPONSE TASK
requirements = fb.extract_instructions('./feedback_log.json')
adjust_output_task = Task(
    description=f"Refine the 'debugger' agent's output based on the requirements: '{requirements}'.",
    agent=adjust_output_agent,
    expected_output="Output a copy of the refined response that aligns with the user's requirements."
)

defect_with_feedback_crew = Crew(
    agents = [debugger, adjust_output_agent],
    tasks = [debug_and_fix_errors, adjust_output_task],
    verbose = 2,
    process = Process.sequential
)

# REFINE DEBUGGER OUTPUT WITH FEEDBACKS
output = defect_with_feedback_crew.kickoff()
print(output)
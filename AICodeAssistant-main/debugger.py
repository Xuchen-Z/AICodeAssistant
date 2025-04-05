import feedback as fb
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

MODEL_NAME = "codellama:7b"
model = Ollama(model=MODEL_NAME)

feedback_path = "./feedback_log.json"

def process_code(lines):
    requirements = fb.extract_instructions(feedback_path)
    # AGENTS ---------------------------------------------------------------------------------------------------------------------------------
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
        llm = model,
        max_iterations=None,  # Disable iteration limit
        timeout=None  # Disable timeout
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
        llm=model,
        max_iterations=None,  # Disable iteration limit
        timeout=None  # Disable timeout
    )

    # TASKS  -----------------------------------------------------------------------------------------------------------------------

    debug_and_fix_errors = Task(
        description = f"Find and fix all errors in the Python code: \n'{lines}'",
        agent = debugger,
        expected_output = "Output a copy of the original Python code with all errors fixed. \
        No extra comments, information or explainations should be provided.",
        max_steps=None  # Remove limit on steps
    )

    adjust_output_task = Task(
        description=f"Refine the 'debugger' agent's output based on the requirements: '{requirements}'.",
        agent=adjust_output_agent,
        expected_output="Output a copy of the refined response that aligns with the user's requirements.",
        max_steps=None  # Remove limit on steps
    )


    # CREW -----------------------------------------------------------------------------------------------------------------------------
    defect_crew = Crew(
        agents = [debugger],
        tasks = [debug_and_fix_errors],
        verbose = 0,
        process = Process.sequential,
        timeout=None  # Remove global timeout
    )

    defect_with_feedback_crew = Crew(
        agents = [debugger, adjust_output_agent],
        tasks = [debug_and_fix_errors, adjust_output_task],
        verbose = 0,
        process = Process.sequential,
        timeout=None  # Remove global timeout
    )


    # FUNCTION CALLS --------------------------------------------------------------------------------------------------

    fixed_code = ""
    if requirements == "No requirement.":
        fixed_code = defect_crew.kickoff()
    else:
        fixed_code = defect_with_feedback_crew.kickoff()

    return fixed_code
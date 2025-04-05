import feedback as fb
import code_access as ca
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

MODEL_NAME = "codellama:7b"
model = Ollama(model=MODEL_NAME)

feedback_path = "./feedback_log.json"


def process_code(lines):
    # AGENTS ---------------------------------------------------------------------------------------------------------------------------------
    requirements = fb.extract_instructions(feedback_path)
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

    error_classifier = Agent(
        role="Strict Error Type Classifier",
        goal="Output ONLY 'syntax', 'runtime', or 'logical' - nothing else",
        backstory="You are a strict classifier that categorizes Python errors. \
        You NEVER add explanations or formatting. Your responses are ALWAYS exactly \
        one lowercase word from: syntax, runtime, logical.",
        verbose=False,
        allow_delegation=False,
        llm=model,
        max_iterations=3,
        stop=["\n", ".", ","],
        max_iterations=None,  # Disable iteration limit
        timeout=None  # Disable timeout
    )

    rag_refiner = Agent(
        role="Code Refinement Specialist",
        goal="Adjust the output from a python code debugger based on previous failure cases. \
        Make necessary adjustments only if accapable.",
        backstory="You are an expert in refining python code based on previous failure cases. \
        Your job is to adjust the provided python code base on previous failure cases while keeping the functionality and correctness of the code. \
        Adjust errors in the Python code only if they closely resemble the errors in the provided failure case examples. \
        Output the original code without adjustments if not accapable.",
        verbose=False,
        allow_delegation = False,
        llm=model,
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

    classification_task = Task(
        description=f"""
        Analyze this Python code and STRICTLY classify the error type:
        
        ### CODE START ###
        {lines}
        ### CODE END ###
        
        Follow these steps:
        1. FIRST check for syntax errors
        2. If no syntax errors, check for runtime errors (undefined variables, division by zero)
        3. If neither, check for logical errors (incorrect calculations, loop conditions)
        
        FORMATTING RULES:
        - Output MUST be exactly 1 lowercase word: 'syntax', 'runtime', or 'logical'
        - Never explain or justify your answer
        - If uncertain, choose based on earliest detectable error
        
        Example valid responses:
        syntax
        runtime
        logical
        """,
        agent=error_classifier,
        expected_output="Exactly one lowercase word from: syntax, runtime, logical",
        max_steps=None  # Remove limit on steps
    )

    adjust_output_task = Task(
        description=f"Refine the 'debugger' agent's output based on the requirements: '{requirements}'.",
        agent=adjust_output_agent,
        expected_output="Output a copy of the refined response that aligns with the user's requirements.",
        max_steps=None  # Remove limit on steps
    )

    # obtain error type, extract examples
    error_type_response = classification_task.execute()
    examples = ca.get_rag_examples(error_type_response, 1)

    rag_refine_task = Task(
        description=(
            "Refine code using these patterns:\n" + examples + 
            "\nApply ONLY relevant improvements from patterns"
        ),
        agent=rag_refiner,
        expected_output="Final optimized code",
        max_steps=None  # Remove limit on steps
    )

    # CREW -----------------------------------------------------------------------------------------------------------

    rag_defect_crew = Crew(
        agents=[debugger, rag_refiner],
        tasks=[debug_and_fix_errors, rag_refine_task],
        verbose=0,
        process = Process.sequential,
        timeout=None  # Remove global timeout
    )

    rag_defect_with_feedback_crew = Crew(
        agents=[debugger, rag_refiner, adjust_output_agent],
        tasks=[debug_and_fix_errors, rag_refine_task, adjust_output_task],
        verbose=0,
        process = Process.sequential,
        timeout=None  # Remove global timeout
    )


    # FUNCTION CALLS --------------------------------------------------------------------------------------------------

    fixed_code = ""
    if requirements == "No requirement.":
        fixed_code = rag_defect_crew.kickoff()
    else:
        fixed_code = rag_defect_with_feedback_crew.kickoff()

    return fixed_code
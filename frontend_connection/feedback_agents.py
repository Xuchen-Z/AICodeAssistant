import feedback as fb
import code_access as ca
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

MODEL_NAME = "codellama:7b"
model = Ollama(model=MODEL_NAME)

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

user_feedback_summarize = Task(
    description = f"Summarize the feedback: '{feedback}'",
    agent = feedback_summarizer,
    expected_output = "Output a concise sumarry of the feedback recieved without extra information or mentioning this is a feedback.",
)

feedback_crew = Crew(
    agents = [feedback_summarizer],
    tasks = [user_feedback_summarize],
    verbose = 0,
    process = Process.sequential
)

new_feedback = feedback_crew.kickoff()
fb.init_feedback_log()
json_str = fb.process_feedback(new_feedback)
fb.save_feedback(json_str)
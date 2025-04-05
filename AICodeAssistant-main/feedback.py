import json

# Sample Json structure for feedback storage

# {
#   "instructions": [
#     "Ignore indentation errors",
#     "Focus on logical errors"
#   ]
# }

FEEDBACK_LOG = "feedback_log.json"

def init_feedback_log():
    print("init_feedback_log")
    try:
        with open(FEEDBACK_LOG, "r") as f:
            pass
    except FileNotFoundError:
        with open(FEEDBACK_LOG, "w") as f:
            json.dump({}, f)


def save_feedback(json_str):
    print("safe_feedback")
    with open(FEEDBACK_LOG, "w") as f:
        f.write(json_str)


def process_feedback(new_instruction):
    print("process feedback")
    try:
        with open(FEEDBACK_LOG, "r") as f:
            existing_feedback = json.load(f)
    except FileNotFoundError:
        existing_feedback = {"instructions": []}

    if "instructions" not in existing_feedback:
        existing_feedback["instructions"] = []

    existing_feedback["instructions"].append(new_instruction.strip())

    updated_feedback_json = json.dumps(existing_feedback)
    return updated_feedback_json

def extract_instructions(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        instructions = data.get("instructions", [])
        if instructions:
            return " ".join(instructions)
        else:
            return "No requirement."
    except FileNotFoundError:
        return f"No requirement."
    except json.JSONDecodeError as e:
        return f"No requirement."
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
    try:
        with open(FEEDBACK_LOG, "r") as f:
            pass
    except FileNotFoundError:
        with open(FEEDBACK_LOG, "w") as f:
            json.dump([], f)


def save_feedback(json_str):
    with open(FEEDBACK_LOG, "w") as f:
        f.write(json_str)


def load_feedback_history():
    try:
        with open(FEEDBACK_LOG, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"instructions": []}


def process_feedback(new_instruction, existing_feedback_json):
    existing_feedback = json.loads(existing_feedback_json)

    if "instructions" not in existing_feedback:
        existing_feedback["instructions"] = []

    existing_feedback["instructions"].append(new_instruction.strip())

    updated_feedback_json = json.dumps(existing_feedback)
    return updated_feedback_json
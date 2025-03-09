import json

def read_script(file_path):
    try:
        # Open the file in read mode with newline parameter to preserve line endings
        with open(file_path, 'r', encoding='utf-8', newline='') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "Error: The specified file was not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_rag_examples(input_str: str, max_examples: int = 1) -> str:
    """
    Retrieve examples for all detected error types with per-type limits
    
    Args:
        input_str: Text containing error type mentions
        max_examples: Maximum examples PER ERROR TYPE to return
    
    Returns:
        Formatted examples string or empty string
    """
    try:
        # Detect which error types are mentioned
        input_lower = input_str.lower()
        detected_types = [
            t for t in ["syntax", "runtime", "logical"] 
            if t in input_lower
        ]
        
        if not detected_types:
            return ""

        # Load failure cases
        with open('./failure_cases.json', 'r') as f:
            cases = json.load(f)

        # Collect up to max_examples for EACH detected type
        collected_examples = []
        for error_type in detected_types:
            type_cases = [
                c for c in cases 
                if c.get('error_type', '').lower() == error_type
            ][:max_examples]
            collected_examples.extend(type_cases)

        if not collected_examples:
            return ""

        # Format examples with clear type headers
        return "\n\n".join(
            f"▣ {case['error_type'].upper()} ERROR EXAMPLE\n"
            f"• Buggy Code of Example:\n```python\n{case['buggy_code']}\n```\n"
            f"• Corrected Version of Example:\n```python\n{case['fixed_code']}\n```"
            for case in collected_examples
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        return ""
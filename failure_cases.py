import os
import json

def generate_failure_cases_json(root_dir: str, output_path: str = "./failure_cases.json"):
    """
    Converts folder structure of failure cases to JSON format.
    
    Args:
        root_dir: Path to parent directory containing error type folders
        output_path: Path to save the resulting JSON file
    """
    failure_cases = []
    error_types = ["logic", "runtime", "syntax"]  # Match your folder names
    
    for error_type in error_types:
        type_dir = os.path.join(root_dir, error_type)
        
        if not os.path.exists(type_dir):
            continue
            
        # Get all error files in directory
        error_files = [f for f in os.listdir(type_dir) 
                      if f.endswith("_error.py")]
        
        for err_file in error_files:
            # Get base number (e.g., "32" from "32_error.py")
            base_number = err_file.split("_")[0]
            gt_file = f"{base_number}_gt.py"
            
            # Build paths
            err_path = os.path.join(type_dir, err_file)
            gt_path = os.path.join(type_dir, gt_file)
            
            if not os.path.exists(gt_path):
                continue  # Skip if ground truth is missing
                
            # Read file contents
            try:
                with open(err_path, 'r') as f:
                    buggy_code = f.read().strip()
                
                with open(gt_path, 'r') as f:
                    fixed_code = f.read().strip()
                    
                failure_cases.append({
                    "error_type": error_type,
                    "buggy_code": buggy_code,
                    "fixed_code": fixed_code,
                    "source_file": f"{base_number}.py"  # Optional metadata
                })
                
            except Exception as e:
                print(f"Error processing {base_number}: {str(e)}")
                continue
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(failure_cases, f, indent=2)
        
    print(f"Generated {len(failure_cases)} cases in {output_path}")


generate_failure_cases_json(
    root_dir="./tests/test_cases/failure",
    output_path="./failure_cases.json"
)
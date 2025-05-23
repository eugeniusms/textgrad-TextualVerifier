import re

def step_formatter(reasoning_path):
    """
    Extract individual steps from a reasoning path.
    
    Args:
        reasoning_path (str): The reasoning path with step tags
        
    Returns:
        list: List of individual reasoning steps
    """
    # Use regex to extract content between <Step> and </Step> tags
    step_pattern = r"<Step>(.*?)</Step>"
    steps = re.findall(step_pattern, reasoning_path, re.DOTALL)
    
    # Clean up extracted steps
    return [step.strip() for step in steps]
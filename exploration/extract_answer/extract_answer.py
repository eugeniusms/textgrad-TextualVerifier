import re

def extract_answer(steps, query=None):
    """
    Extract the final answer from the reasoning path based on the query type.
    
    Args:
        steps (list): List of reasoning steps
        query (str, optional): The original query to determine the answer format
        
    Returns:
        str: The extracted answer in the appropriate format
    """
    # Check if this is a multiple choice question
    is_multiple_choice = False
    expected_format = None
    
    if query:
        # Check if the query mentions multiple choice
        if "multiple choice" in query.lower():
            is_multiple_choice = True
        
        # Check if there's an expected answer format
        format_match = re.search(r"format:\s*'([^']+)'", query)
        if format_match:
            expected_format = format_match.group(1)
            
        # Check if the query contains options like A), B), etc.
        if re.search(r'[A-Z]\)', query) or re.search(r'\([A-Z]\)', query):
            is_multiple_choice = True
    
    # For multiple choice questions, look for an answer in the format "Answer: X"
    if is_multiple_choice:
        # Check all steps, with priority given to the last steps
        for step in reversed(steps):
            # Common patterns for multiple choice answers
            patterns = [
                r'Answer:\s*([A-D])',  # "Answer: A"
                r'answer\s+is\s+([A-D])',  # "answer is A"
                r'option\s+([A-D])',  # "option A"
                r'([A-D])\)',  # "A)"
                r'\(([A-D])\)'  # "(A)"
            ]
            
            for pattern in patterns:
                answer_match = re.search(pattern, step, re.IGNORECASE)
                if answer_match:
                    letter = answer_match.group(1).upper()
                    
                    # Return in the expected format if specified
                    if expected_format:
                        return expected_format.replace('$LETTER', letter)
                    else:
                        return f"Answer: {letter}"
    
    # For numeric answers
    # Check all steps, with priority given to the last steps
    for step in reversed(steps):
        # Look for "the answer is X" pattern
        answer_pattern = re.search(r'(answer|result)\s+is\s+([0-9.e^-]+)', step, re.IGNORECASE)
        if answer_pattern:
            return answer_pattern.group(2)
        
        # Look for "Therefore, X" pattern
        therefore_pattern = re.search(r'therefore,?\s+([0-9.e^-]+)', step, re.IGNORECASE)
        if therefore_pattern:
            return therefore_pattern.group(1)
        
        # Look for any number with eV (electron volt) for physics problems
        ev_pattern = re.search(r'([0-9.e^-]+)\s*eV', step)
        if ev_pattern:
            return ev_pattern.group(1) + " eV"
    
    # If no specific pattern is found, look for the last numeric value in the last step
    last_step = steps[-1]
    number_matches = re.findall(r'([0-9.e^-]+)', last_step)
    if number_matches:
        return number_matches[-1]
    
    # If still no answer found, return the last step
    return last_step
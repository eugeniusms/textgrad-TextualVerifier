def cot_prompter(query):
    initial_reasoning_path = f"""
        Mark the beginning and end of each reasoning step with <Step> 
        and </Step> tags. Q: q. A: Let's think step by step.

        {query}
    """
    return initial_reasoning_path
    
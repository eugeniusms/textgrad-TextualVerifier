"""
Prompts for the TextualVerifier system.
All prompts are process-focused to encourage better reasoning rather than direct corrections.
"""

COT_PROMPT = """
Break down this instance into clear calculation steps.
Focus only on the mathematical/logical steps needed.
Mark each step with <Step> and </Step> tags.

Instance: {instance}

Let's think step by step:"""

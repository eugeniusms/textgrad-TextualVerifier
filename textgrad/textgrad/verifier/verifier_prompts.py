"""
Prompts for the TextualVerifier system.
All prompts are process-focused to encourage better reasoning rather than direct corrections.
"""

DEFAULT_VERIFICATION_TASK_PROMPTS = [
    "Verify any miss on calculation, if any misses please revise calculation based on misses."
]

COT_PROMPT = """
Break down this calculation into clear steps.
Focus only on the mathematical/logical steps needed.
Mark each step with <STEP> and </STEP> tags.

Calculation: {calculation}

Let's think step by step:"""

VARIANT_GENERATION_PROMPT = """"
You are verifying whether the calculation correctly follows from applying the instruction to the instance.

Instance: 
{instance}

Instruction: 
{instruction}

Previous Context:
{previous_context}

Calculation:
{calculation}

Verification Tasks:
{verification_task_prompt}

Provide ONLY the improved calculation, no additional text or formatting.
"""

MAJORITY_VOTING_PROMPT = """
Original calculation: 
{calculation}

Generated variants:
{generated_variants}

Use majority voting to identify the variant that appears most frequently.
Select that single variant as the replacement for the original calculation.
Provide ONLY the selected variant/merged variant, with no additional text or formatting.
"""
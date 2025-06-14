"""
Prompts for the TextualVerifier system.
All prompts are process-focused to encourage better reasoning rather than direct corrections.
"""

COT_PROMPT = """
Break down this instance into clear calculation steps.
Focus only on the mathematical/logical steps needed.
Mark each step with <STEP> and </STEP> tags.

Instance: {instance}

Let's think step by step:"""

VARIANT_GENERATION_PROMPT_STEP_BASED = """
Given instance and instruction.

Instance: {instance}

Instruction: {instruction}

The result of instruction to instance is:
{calculation}

Now I build verified calculation, currently I have:
{context}

Verification:
{verification_prompt}
"""

VARIANT_GENERATION_PROMPT = """"
You are verifying whether the calculation correctly follows from applying the instruction to the instance.

Instance: 
{instance}

Instruction: 
{instruction}

Calculation:
{calculation}

Verification Tasks:
{verification_prompt}
"""

MAJORITY_VOTING_PROMPT = """
"""
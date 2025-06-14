"""
Prompts for the TextualVerifier Experiment (using v2 modified due to feasibility of experiment).
All prompts are process-focused to encourage better reasoning rather than direct corrections.
"""

# Updated for Experiment
VARIANT_GENERATION_PROMPT_WITH_CONTEXT = """
You are verifying the correctness of a step-by-step solution.

Original solution:
{}

Verification instruction:
{}

Previously verified steps (cumulative context):
{}

Current step to verify:
{}

IS_LAST_STEP: {}

Instructions for you:
- Focus only on verifying or revising **the current step** using the context above.
- If the current step is correct and logically consistent, keep it as is.
- If the step has errors, unclear logic, or poor structure, revise it to improve reasoning and clarity.
- Preserve the **exact original format**:
  Step {} variant {}: <your verified or revised step goes here>
- DO NOT solve the problem or compute the final answer.
- DO NOT include or duplicate any previous steps.
- ONLY output the verified or revised current step.

If IS_LAST_STEP is true:
- Append the answer in a new line at the end using this exact format:
  # Answer <answer>

Important:
- Do NOT provide explanations.
- Do NOT include any commentary.
- Output only the revised step and optional answer if applicable.
"""

# ENHANCED: Now includes cumulative context for consistency evaluation
VOTING_PROMPT_WITH_CONTEXT = """
CUMULATIVE CONTEXT (Previously verified steps):
{}

Original step:
{}

Candidate revisions:
{}

Instruction:
Select the most methodologically sound and contextually consistent candidate revision.

Evaluation Criteria:
- Logical continuation of the previous verified steps
- Clear and systematic reasoning
- Strong error-prevention and clarity of process
- Maintains alignment with the reasoning chain established so far

ONLY return the best revised version in the original format:
"""

DECISION_PROMPT = """
You are given two reasoning steps for the same problem:

Original version:
{}
---
Verified version:
{}

Your task is to choose which version has better logical and mathematical reasoning.

Follow these steps:
1. Carefully compare the structure, logic, and correctness of the reasoning in both versions.
2. Identify any flaws, gaps, or invalid steps in either version.
3. Focus on methodology quality — not just the final answer.

Use the following decision rules:
- Respond with **[REPLACE]** if the Verified version clearly improves or corrects flaws in the Original.
- Respond with **[SUFFICIENT]** if the Original version is already valid and the Verified version does not add meaningful improvement.

Only respond with one of the following tokens: [REPLACE] or [SUFFICIENT]
"""
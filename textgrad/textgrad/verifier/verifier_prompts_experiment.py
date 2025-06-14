"""
Prompts for the TextualVerifier Experiment (using v2 modified due to feasibility of experiment).
All prompts are process-focused to encourage better reasoning rather than direct corrections.
"""

# Updated for Experiment
VARIANT_GENERATION_PROMPT_WITH_CONTEXT = """
Original solution: {}
Instruction: {}

CUMULATIVE CONTEXT (Previous verified steps):
{}

Current step to verify: {}

IS_LAST_STEP: {}

Instruction:
- Verify the current step using the context of the previously verified steps.
- If the step is correct and consistent, return it as is.
- If the step contains errors or lacks clarity, revise it to improve the logic and coherence.
- Maintain the **same format**: Step {} variant {}: <verified or revised step content>
- Do NOT solve the problem or compute final answers—focus only on logical verification and revision of the current step.
- ONLY revise the current step. DO NOT include or duplicate other steps.

ONLY return the updated solution in the same format. 
MUST add # Answer <answer> if IS_LAST_STEP is true.
DO NOT add any explanation or additional commentary.
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
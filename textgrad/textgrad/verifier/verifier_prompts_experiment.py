"""
Prompts for the TextualVerifier Experiment (using v2 modified due to feasibility of experiment).
All prompts are process-focused to encourage better reasoning rather than direct corrections.
Key improvements:
- Clearer format specifications
- Better context utilization
- Reduced redundancy
- Stricter output control
"""

SEED_RANDOM_APPROACHES = [
    "Focus primarily on mathematical rigor and computational accuracy.",
    "Emphasize clarity and step-by-step logical progression.",
    "Consider alternative mathematical methods and approaches.",
    "Prioritize elegant and efficient problem-solving techniques.",
    "Look for potential edge cases and verify all assumptions."
]

SEED_RANDOM_IMPROVEMENT = [
    "Make the solution more detailed with additional explanations.",
    "Simplify and streamline the mathematical presentation.",
    "Add more rigorous mathematical justification.",
    "Focus on computational efficiency and clarity.",
    "Enhance with alternative solution perspectives."
]

VARIANT_GENERATION_PROMPT_WITH_CONTEXT = """
You are a mathematical expert reviewing and improving a step in a solution.

PROBLEM: {problem}

SOLUTION APPROACH: {approach}

VERIFIED PREVIOUS STEPS:
{context}

CURRENT STEP TO ANALYZE AND IMPROVE:
{current_step}

IS_FINAL_STEP: {is_final}

TASK:
First, carefully analyze the current step for any mathematical errors, logical issues, or improvements needed. Then create an improved version.

ANALYSIS CHECKLIST:
1. Are all mathematical calculations correct?
2. Is the mathematical reasoning sound?
3. Does it follow logically from previous steps?
4. Are the mathematical concepts applied correctly?
5. Is the notation and terminology accurate?
6. For combinatorics: Are the right formulas used (e.g., circular vs linear permutations)?

COMMON ERRORS TO CHECK:
- Using n! instead of (n-1)! for circular arrangements
- Incorrect factorial calculations
- Wrong application of combinatorial formulas
- Arithmetic errors in calculations
- Logical inconsistencies with the problem setup

IMPROVEMENT PROCESS:
1. If errors are found: Correct them while explaining the fix
2. If no errors: Improve clarity, add detail, or use alternative approach
3. Make the step more mathematically rigorous and clear
4. Ensure perfect accuracy in all calculations

REQUIREMENTS:
- Write as if you are the original solver
- Use clear, direct mathematical language
- Ensure logical flow from previous steps
- Maintain or improve mathematical correctness
- Do not include meta-commentary about the analysis process
- Focus on the mathematical content

{final_instruction}

OUTPUT FORMAT:
Provide only the corrected/improved step content, no additional text or formatting.
"""

VOTING_PROMPT_WITH_CONTEXT = """
You need to select the best mathematical step from the candidates provided.

CONTEXT FROM PREVIOUS VERIFIED STEPS:
{context}

ORIGINAL STEP:
{original_step}

CANDIDATE IMPROVEMENTS:
{candidates}

SELECT the candidate that:
1. Is mathematically most accurate
2. Follows logically from previous steps  
3. Uses clear, direct language
4. Contains no meta-commentary

OUTPUT FORMAT:
Return only the selected candidate content (no "Candidate X:" prefix, no explanations).
{voting_instruction}
"""

DECISION_PROMPT = """
Compare the mathematical correctness and logical flow of these two solution versions:

ORIGINAL SOLUTION STEPS:
{original_steps}

VERIFIED SOLUTION STEPS: 
{verified_steps}

TASK: Determine which version is better.

DECISION RULES:
- Use REPLACE if verified version fixes mathematical errors or significantly improves clarity
- Use SUFFICIENT if original version is already mathematically sound

OUTPUT: Respond with exactly one word: REPLACE or SUFFICIENT
"""

def get_final_step_instruction(is_final_step: bool) -> str:
    """Helper function to format final step instruction"""
    if is_final_step:
        return "\n- Since this is the final step, end with the final answer in the format: Answer: [numerical_value]"
    return ""

def get_voting_final_instruction(step_index: int, total_steps: int) -> str:
    """Helper function to format voting final instruction"""
    if step_index == total_steps - 1:
        return "\nEnsure the selected step ends with: Answer: [numerical_value]"
    return ""
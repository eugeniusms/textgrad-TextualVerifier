"""
Prompts for the TextualVerifier Experiment (using v2 modified due to feasibility of experiment).
All prompts are process-focused to encourage better reasoning rather than direct corrections.
Key improvements:
- Clearer format specifications
- Better context utilization
- Reduced redundancy
- Stricter output control
"""

VARIANT_GENERATION_PROMPT_WITH_CONTEXT = """
You are improving a step in a mathematical solution. Your job is to rewrite the current step with a DIFFERENT approach or explanation style.

PROBLEM: {problem}

SOLUTION APPROACH: {approach}

VERIFIED PREVIOUS STEPS:
{context}

CURRENT STEP TO IMPROVE:
{current_step}

IS_FINAL_STEP: {is_final}

TASK:
Create a DIFFERENT version of this step that:
1. Uses different wording or mathematical notation
2. May use an alternative mathematical approach
3. Provides a different level of detail (more or less)
4. Maintains mathematical correctness
5. Is DISTINCT from the original step

REQUIREMENTS:
- Write as if you are the original solver
- Use clear, direct mathematical language  
- Ensure logical flow from previous steps
- Do not reference other steps explicitly
- Do not include meta-commentary or judgments
- Keep mathematical notation consistent
- MAKE IT DIFFERENT from the original step
{final_instruction}

OUTPUT FORMAT:
Provide only the improved step content, no additional text or formatting.
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
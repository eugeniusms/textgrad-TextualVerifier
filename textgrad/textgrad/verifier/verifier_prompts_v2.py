"""
Prompts for the TextualVerifier system.
All prompts are process-focused to encourage better reasoning rather than direct corrections.
"""

COT_PROMPT = """
Break down this problem into clear calculation steps.
Focus only on the mathematical/logical steps needed.
Mark each step with <Step> and </Step> tags.

Problem: {}

Let's think step by step:"""

# ENHANCED: Now includes cumulative context for better step coherence
VARIANT_GENERATION_PROMPT_WITH_CONTEXT = """
Original problem: {}
Instruction: {}

CUMULATIVE CONTEXT (Previous verified steps):
{}

Current step: {}

Analyze this step considering the previous context and identify potential process improvements:
- Are there methodological issues?
- Could the approach be more systematic?
- Does this step build logically on previous verified steps?
- What alternative reasoning methods exist that maintain consistency?

Provide process-focused guidance for improving this step while maintaining coherence with previous steps.
DO NOT give direct numerical answers.
Variant {}: """

# ENHANCED: Now includes cumulative context for consistency evaluation
VOTING_PROMPT_WITH_CONTEXT = """
CUMULATIVE CONTEXT (Previous verified steps):
{}

Original step: {}

Process improvement suggestions:
{}

Which approach provides the best methodological guidance considering the previous context?
Focus on:
- Clearest reasoning process that builds on previous steps
- Best error-prevention strategies
- Most systematic approach that maintains coherence
- Consistency with established reasoning chain

Return the best process-focused guidance:"""

MERGE_STEPS_PROMPT = """
Instruction: {}

Process improvement guidance from verification steps:
{}

Synthesize this guidance into coherent process-focused feedback that:
- Identifies methodology errors
- Suggests systematic improvements
- Guides better reasoning approaches
- Does NOT provide direct numerical answers

Provide the merged process feedback:"""

DECISION_PROMPT = """
Compare these two evaluations:

Original evaluation: {}

Process-focused verification: {}

Your task: Provide feedback that focuses on PROCESS improvements, not direct answers.

Guidelines:
- Identify methodology errors and reasoning flaws
- Suggest systematic approaches for improvement
- Guide better problem-solving processes
- Avoid giving direct numerical corrections
- Focus on teaching better reasoning skills

Classify and provide process-focused feedback:
1. ENHANCE - Original feedback can be improved with process guidance
2. REPLACE - Original feedback is insufficient, use process-focused version
3. SUFFICIENT - Original feedback is already process-focused

Respond with: [DECISION]: [PROCESS_FOCUSED_FEEDBACK]"""
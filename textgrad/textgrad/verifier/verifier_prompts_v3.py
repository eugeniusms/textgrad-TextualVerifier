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

# V3 ENHANCEMENT: Consolidated verification prompt for all steps in one call
CONSOLIDATED_VERIFICATION_PROMPT = """
Original problem: {}
Instruction: {}

Steps to verify:
{}

Your task: Verify and improve ALL steps in a single comprehensive analysis.

For each step, provide process-focused guidance that:
- Analyzes methodological issues
- Suggests systematic improvements  
- Ensures logical coherence between steps
- Maintains consistency throughout the reasoning chain
- Does NOT provide direct numerical answers

Format your response as:
<VerifiedStep1>Your improved guidance for step 1</VerifiedStep1>
<VerifiedStep2>Your improved guidance for step 2</VerifiedStep2>
<VerifiedStep3>Your improved guidance for step 3</VerifiedStep3>
...and so on for all steps

Focus on process improvements and reasoning methodology:"""

# V3 ENHANCEMENT: Voting on complete solutions rather than individual steps
CONSOLIDATED_VOTING_PROMPT = """
Original steps: 
{}

Complete verification variants:
{}

Which variant provides the best overall methodological guidance?
Focus on:
- Clearest reasoning process across all steps
- Best error-prevention strategies throughout
- Most systematic approach with strong coherence
- Consistency in the complete reasoning chain

Return the best complete process-focused guidance:"""

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
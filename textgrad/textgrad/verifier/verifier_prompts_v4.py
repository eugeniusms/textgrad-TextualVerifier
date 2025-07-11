"""
Prompts for the TextualVerifier system.
All prompts are process-focused to encourage better reasoning rather than direct corrections.
"""

# V4 ENHANCEMENT: Concise prompt using <StepN></StepN> format
COT_PROMPT = """
Breakdown the initial solution step by step without changing math calculation/logical reasoning.
Label each step as <Step1></Step1>, <Step2></Step2>, etc. Please be concise each steps.

Problem: {}

Let's think step by step:"""

# V4 ENHANCEMENT: Concise prompt using <StepN></StepN> format
CONSOLIDATED_VERIFICATION_PROMPT = """
Original problem: {}
Instruction: {}

Steps to verify:
{}

Your task: 
For every steps: verify math calculation & logical reasoning based on original problem & previous steps.
Then deliver concise error analysis feedback for each steps.
If no error state "No error".

Format your response as:
<FeedbackStep1>Your concise error feedback for step 1</FeedbackStep1>
<FeedbackStep2>Your concise error feedback for step 2</FeedbackStep2>
...and so on for all steps
"""

# V4 ENHANCEMENT: Voting with concise prompt
GROUPING_STEP_FOR_VOTING_PROMPT = """
Original steps: 
{}

Step verification feedback variants:
{}

Grouping each FeedbackStep by their Number in this format:
<FeedbackStep1Variant1></FeedbackStep1Variant1>
<FeedbackStep1Variant2></FeedbackStep1Variant1>
...
<FeedbackStep1Variant`$TotalVariant`></FeedbackStep1Variant`$TotalVariant`>

For all steps.
"""

# V4 ENHANCEMENT: Voting with concise prompt
CONSOLIDATED_VOTING_PROMPT = """
Grouped feedback variant by steps:
{}

For each FeedbackStep$Number, perform majority voting across variants.
If the majority feedback is 'No error', skip the step (do not output it).
If the majority feedback is any other message, include it as:

<Step$Number>majority_feedback</Step$Number>

Output only the steps with feedback other than 'No error'.
"""

# V4 ENHANCEMENT: 
# I want (1) be verified by (2), so if anything wrong in (1) -> update it, 
# if true just stay the existing, if anything left behind in (1) add from (2), 
# the result is in (1) format
SUMMARIZED_VERIFICATION_RESULT = """
You are given two types of feedback for a solution:

(1) Feedback generated by a general evaluator (e.g., a model's loss-based calculation or initial reasoning trace).  
(2) Feedback generated by a verificator (e.g., step-by-step logical or rule-based analysis).

Your task:
- Verify the feedback in (1) using the verification details in (2).
- If all step in (2) is "No error", leave it unchanged.
- If any part of (1) is incorrect according to (2), update it.
- If any important information is missing in (1) but present in (2), add it clearly.
- If any part in (1) is already correct and validated by (2), leave it unchanged.
- Return the result in the same concise and summarized format as (1).

Treat both (1) and (2) as structured forms of feedback — (1) from an evaluator, and (2) from a verificator.

Inputs:
(1)
{}

(2)
{}

Output:
Improved version of (1), verified and refined based on (2).
"""
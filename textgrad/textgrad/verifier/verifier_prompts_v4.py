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
<FeedbackStep2>Your concise feedback for step 2</FeedbackStep2>
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

For each FeedbackStep$Number vote majority of feedbacks,
For step with majority is 'No error', write 'No error' in that step.

Create output format:

Voting Results:
<Step1></Step1>
<Step2></Step2>
...
<Step`$TotalStep`></Step`$TotalStep`>
"""

# V4 ENHANCEMENT: Voting with concise prompt
# HIGHLIGHT ORIGINAL STEP WITH CORRECTION PER STEP CONCISE IN TEXT NO FORMAT
SUMMARIZED_VERIFICATION_RESULT = """
Subject: {}

Correction: {}

Your task:  
Merge the subject and correction into concise summary that merges the subject and corrections, ensuring nothing is left out:
"""
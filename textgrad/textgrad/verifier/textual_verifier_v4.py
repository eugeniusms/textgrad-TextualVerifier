import re
from textgrad.variable import Variable
from typing import Union, List
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .verifier import Verifier
from .verifier_prompts_v4 import (
    COT_PROMPT,
    CONSOLIDATED_VERIFICATION_PROMPT,
    GROUPING_STEP_FOR_VOTING_PROMPT,
    CONSOLIDATED_VOTING_PROMPT,
    SUMMARIZED_VERIFICATION_RESULT,
)

"""
[UPDATE TextualVerifier V4]

Situation: 
TextualVerifierV2 shows good stability in transition and significant accuracy,
but incurs high LLM cost due to multiple verification steps. TextualVerifierV4
shows good speed (fast-running) low LLM cost, but incurs unstability in iteration.
Also, previous approach have a good zero-shot but unstable final solution.

Objective:
Combining two approaches to increase stability while maintaining speed and low LLM cost.
Simplify & resizing prompt to increase accuracy gradually not jump perfect zero-shot.

Action List:
1.  Add layer function inside verification. In verification step-by-step add
    function to verify step-by-step formatted and voting formatted step-by-step too.
2.  Simplify & Resizing Prompt, make it concise and quality improving gradually.
3.  Add GroupingFeedback in the voting stage for LLM feedback clarity.

Hypothesis:
It will resulting:
1. Faster as TextualVerifierV4.
2. Stable accuracy as TextualVerifierV2.
3. Low zero-shot accuracy, but increasing gradually to final solution.
"""
class TextualVerifierV4(Verifier):
    def __init__(self,
                verifier_engine: Union[str, EngineLM], 
                step_eval_iterations: int = 3, 
                logger: bool = False):
        self.engine = validate_engine_or_get_default(verifier_engine)
        self.step_eval_iterations = step_eval_iterations
        self.logger = logger
        
    def verify(self, instance: Variable, prompt: Variable, calculation: Variable) -> Variable:
        if self.logger:
            print("INFO:textgrad:TextualVerifierV4:verify Starting Cost-Optimized Verification...")
        
        # Step 1: Generate reasoning steps from instance
        reasoning_steps = self._generate_cot_steps(instance.value)
        
        # Step 2: V4 ENHANCEMENT - Concise function and prompting
        verified_feedbacks = self._verify_all_steps_in_one_call(
            instance.value, prompt.value, reasoning_steps
        )

        # Step 3: V4 ENHANCEMENT - Vote on this verify function
        best_verification_feedback = self._vote_on_complete_solutions(
            reasoning_steps, verified_feedbacks
        )
        
        print("CALCULATION", calculation)
        # Step 4: Make summary
        final_result = self._get_summary_voted_steps(
            calculation, best_verification_feedback
        )
        
        if self.logger:
            print("[V] Cost-optimized verification complete!")
        return Variable(final_result, requires_grad=True, role_description="verification result")
    
    def _generate_cot_steps(self, instance: str) -> str:
        if self.logger:
            print("INFO:textgrad:TextualVerifierV4:generate_cot_steps Generating CoT steps...")
        
        cot_prompt = COT_PROMPT.format(instance)
        steps = self.engine(cot_prompt)
        # print(steps)
        
        if self.logger:
            print(f"INFO:textgrad:TextualVerifierV4:generate_cot_steps Generated some steps")
        return steps

    def _verify_all_steps_in_one_call(self, instance: str, prompt: str, formatted_steps: str) -> List[str]:
        if self.logger:
            print("INFO:textgrad:TextualVerifierV4:verify_all_steps_in_one_call Consolidated verification...")
        
        # Generate multiple complete solution variants in parallel
        complete_variants = self._generate_verification_feedback_variants(
            instance, prompt, formatted_steps
        )

        return complete_variants
    
    def _generate_verification_feedback_variants(self, instance: str, prompt: str, steps_text: str) -> List[str]:
        variants = []
        
        for iteration in range(self.step_eval_iterations):
            if self.logger:
                print(f"Generating complete solution variant {iteration + 1}/{self.step_eval_iterations}")
            
            variant_prompt = CONSOLIDATED_VERIFICATION_PROMPT.format(
                instance,
                prompt,
                steps_text,
                iteration + 1,
                iteration + 1
            )
            
            variant = self.engine(variant_prompt)
            # print(f"VARIANT {iteration}", variant)
            variants.append(variant.strip())
        
        return variants
    
    def _vote_on_complete_solutions(self, original_steps: str, complete_variants: List[str]) -> str:
        variants_text = "\n".join(f"Variants {i+1}:\n{var}\n" for i, var in enumerate(complete_variants))
        # print("VARIANTS TEXT:", variants_text)

        grouping_variant_by_steps_prompt = GROUPING_STEP_FOR_VOTING_PROMPT.format(
            original_steps,
            variants_text
        )
        grouped_variant_by_steps = self.engine(grouping_variant_by_steps_prompt)
        
        voting_prompt = CONSOLIDATED_VOTING_PROMPT.format(grouped_variant_by_steps)
        best_voted_feedback = self.engine(voting_prompt)

        return best_voted_feedback.strip()

    def _get_summary_voted_steps(self, calculation: str, best_voted_feedback: str) -> str:

        print("VERIFIER", best_voted_feedback)
        
        summarized_verification_result = SUMMARIZED_VERIFICATION_RESULT.format(
            calculation,
            best_voted_feedback,
        )
        summary = self.engine(summarized_verification_result)

        return summary
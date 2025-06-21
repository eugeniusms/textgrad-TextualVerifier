import re
from textgrad.variable import Variable
from typing import Union, List
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .verifier import Verifier
from .verifier_prompts_v3 import (
    COT_PROMPT,
    CONSOLIDATED_VERIFICATION_PROMPT,
    CONSOLIDATED_VOTING_PROMPT,
    MERGE_STEPS_PROMPT,
    DECISION_PROMPT
)

"""
[UPDATE TextualVerifier V3]

Situation: 
TextualVerifierV2 shows good stability in transition and significant accuracy,
but incurs high LLM cost due to multiple verification steps.

Objective:
Reduce LLM cost while maintaining overall accuracy and reasoning stability.

Action List:
1.  Instead of verifying each step individually with the LLM,
    consolidate all CoT-prompted steps into a single LLM call.
    This is formatted as a list output optimized for TGD (TextGrad) processing.

    Complexity Reduction:
        From: O(M x N) -> (generated_steps x step_eval_iterations)
        To:   O(N)     -> (single step_eval_iterations)

2.  Instead of voting variants step-by-step in individual LLM call,
    consolidate all variant steps into single LLM call and vote the variant directly.

    Complexity Reduction:
        From: O(N) -> (generated_steps)
        To: O(1)   -> (1 LLM call)

Hypothesis:
It will reduce rough complexity:
FROM: O(M x N) + O(N) + O(3)
TO: O(N) + O(4)
"""
class TextualVerifierV3(Verifier):
    """
    Cost-optimized verifier that consolidates all step verification into single LLM calls.
    Reduces complexity from O(M x N) to O(N) while maintaining verification quality.
    """

    def __init__(self,
                verifier_engine: Union[str, EngineLM], 
                step_eval_iterations: int = 3, 
                logger: bool = False):
        """
        Initialize the cost-optimized verifier.
        
        Args:
            verifier_engine: The LLM engine for verification
            step_eval_iterations: How many complete solution variants to generate
            logger: Enable logging for debugging
        """
        self.engine = validate_engine_or_get_default(verifier_engine)
        self.step_eval_iterations = step_eval_iterations
        self.logger = logger
        
    def verify(self, instance: Variable, prompt: Variable, calculation: Variable) -> Variable:
        """
        Main verification function with cost optimization.
        
        Args:
            instance: Original problem/question
            prompt: The instruction/prompt used
            calculation: Result from instance + prompt (what we want to verify)
            
        Returns:
            Variable: Verified/improved calculation
        """
        if self.logger:
            print("INFO:textgrad:TextualVerifierV3:verify Starting Cost-Optimized Verification...")
        
        # Step 1: Generate reasoning steps from instance
        reasoning_steps = self._generate_cot_steps(instance.value)
        
        # Step 2: Format steps properly
        formatted_steps = self._format_steps(reasoning_steps)
        
        # Step 3: V3 ENHANCEMENT - Verify ALL steps in a single consolidated call
        verified_steps = self._verify_all_steps_in_one_call(
            instance.value, prompt.value, formatted_steps
        )
        
        # Step 4: Merge all verified steps
        merged_calculation = self._merge_verified_steps(prompt.value, verified_steps)
        
        # Step 5: Make final decision
        final_result = self._make_decision(calculation.value, merged_calculation)
        
        if self.logger:
            print("[V] Cost-optimized verification complete!")
        return Variable(final_result, requires_grad=True, role_description="verified calculation")
    
    def _generate_cot_steps(self, instance: str) -> List[str]:
        """Generate Chain of Thought steps from the instance."""
        if self.logger:
            print("INFO:textgrad:TextualVerifierV3:generate_cot_steps Generating CoT steps...")
        
        cot_prompt = COT_PROMPT.format(instance)
        response = self.engine(cot_prompt)
        steps = self._extract_steps_from_response(response)
        
        if self.logger:
            print(f"INFO:textgrad:TextualVerifierV3:generate_cot_steps Generated {len(steps)} steps")
        return steps
    
    def _extract_steps_from_response(self, response: str) -> List[str]:
        """Extract steps from LLM response."""
        step_pattern = r"<Step>(.*?)</Step>"
        steps = re.findall(step_pattern, response, re.DOTALL)
        
        cleaned_steps = [step.strip() for step in steps if step.strip()]
        
        if not cleaned_steps:
            lines = response.split('\n')
            cleaned_steps = [line.strip() for line in lines 
                           if line.strip() and len(line.strip()) > 10]
        
        return cleaned_steps[:5]  # Limit to 5 steps max
    
    def _format_steps(self, steps: List[str]) -> List[str]:
        """Format steps for better processing."""
        if self.logger:
            print("INFO:textgrad:TextualVerifierV3:format_steps Formatting steps...")
        
        formatted = []
        for i, step in enumerate(steps):
            formatted_step = f"Step {i+1}: {step}"
            formatted.append(formatted_step)
        
        return formatted

    def _verify_all_steps_in_one_call(self, instance: str, prompt: str, formatted_steps: List[str]) -> List[str]:
        """
        V3 ENHANCEMENT: Verify ALL steps in a single consolidated LLM call.
        
        Complexity Reduction:
        - V2: O(M x N) calls → (num_steps x step_eval_iterations) 
        - V3: O(N) calls → (step_eval_iterations only)
        """
        if self.logger:
            print("INFO:textgrad:TextualVerifierV3:verify_all_steps_in_one_call Consolidated verification...")
        
        # Format all steps into a single text block
        steps_text = "\n".join(formatted_steps)
        
        # Generate multiple complete solution variants in parallel
        complete_variants = self._generate_complete_solution_variants(
            instance, prompt, steps_text
        )
        
        # Vote on the best complete solution
        best_complete_solution = self._vote_on_complete_solutions(
            steps_text, complete_variants
        )
        
        # Extract individual verified steps from the best solution
        verified_steps = self._extract_verified_steps_from_solution(best_complete_solution)
        
        if self.logger:
            print(f"Verified {len(verified_steps)} steps in single consolidated call")
        
        return verified_steps
    
    def _generate_complete_solution_variants(self, instance: str, prompt: str, steps_text: str) -> List[str]:
        """
        Generate multiple variants of the complete solution in parallel.
        Each variant analyzes ALL steps together for consistency.
        """
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
            variants.append(variant.strip())
        
        return variants
    
    def _vote_on_complete_solutions(self, original_steps: str, complete_variants: List[str]) -> str:
        """
        V3 ENHANCEMENT: Vote on the best complete solution considering all steps together.
        
        Complexity Reduction:
        - V2: O(N) calls → (num_steps) 
        - V3: O(1) call 
        """
        variants_text = "\n".join(f"Variant {i+1}:\n{var}\n" for i, var in enumerate(complete_variants))
        
        voting_prompt = CONSOLIDATED_VOTING_PROMPT.format(
            original_steps,
            variants_text
        )
        
        best_solution = self.engine(voting_prompt)
        return best_solution.strip()
    
    def _extract_verified_steps_from_solution(self, complete_solution: str) -> List[str]:
        """
        Extract individual verified steps from the complete solution.
        Looks for <VerifiedStep1>, <VerifiedStep2>, etc. patterns.
        """
        verified_steps = []
        
        # Look for <VerifiedStepN>...</VerifiedStepN> patterns
        step_pattern = r"<VerifiedStep(\d+)>(.*?)</VerifiedStep\d+>"
        matches = re.findall(step_pattern, complete_solution, re.DOTALL)
        
        if matches:
            # Sort by step number and extract content
            sorted_matches = sorted(matches, key=lambda x: int(x[0]))
            verified_steps = [match[1].strip() for match in sorted_matches]
        else:
            # Fallback: split by common patterns if no tags found
            lines = complete_solution.split('\n')
            verified_steps = [line.strip() for line in lines 
                            if line.strip() and len(line.strip()) > 10][:5]
        
        return verified_steps
    
    def _merge_verified_steps(self, prompt: str, verified_steps: List[str]) -> str:
        """Merge all verified steps into one coherent calculation."""
        if self.logger:
            print("INFO:textgrad:TextualVerifierV3:merge_verified_steps Merging verified steps...")
        
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(verified_steps))
        merge_prompt = MERGE_STEPS_PROMPT.format(prompt, steps_text)
        
        merged = self.engine(merge_prompt)
        return merged.strip()
    
    def _make_decision(self, original_calculation: str, merged_calculation: str) -> str:
        """Make final decision using DECISION_PROMPT template."""
        if self.logger:
            print("INFO:textgrad:TextualVerifierV3:make_decision Making final decision...")
        
        decision_prompt = DECISION_PROMPT.format(original_calculation, merged_calculation)
        decision_response = self.engine(decision_prompt)
        
        # Parse decision based on the prompt template format
        if "REPLACE" in decision_response.upper():
            if self.logger:
                print("[X] Replacing with cost-optimized verification")
            return merged_calculation
        elif "ENHANCE" in decision_response.upper():
            if self.logger:
                print("[-] Enhancing original with cost-optimized improvements")
            final_feedback = decision_response.split(":", 1)[-1].strip()
            return final_feedback if final_feedback else merged_calculation
        else:  # SUFFICIENT
            if self.logger:
                print("[V] Keeping original calculation")
            return original_calculation
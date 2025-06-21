import re
from textgrad.variable import Variable
from typing import Union, List
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .verifier import Verifier
from .verifier_prompts import (
    COT_PROMPT,
    VARIANT_GENERATION_PROMPT,
    VOTING_PROMPT,
    MERGE_STEPS_PROMPT,
    DECISION_PROMPT
)

class TextualVerifierV1(Verifier):
    """
    A verifier that uses an LLM to evaluate and improve reasoning steps.
    """

    def __init__(self,
                verifier_engine: Union[str, EngineLM], 
                step_eval_iterations: int = 3, 
                logger: bool = False):
        """
        Initialize the verifier.
        
        Args:
            verifier_engine: The LLM engine for verification
            step_eval_iterations: How many variants to generate per step
        """
        self.engine = validate_engine_or_get_default(verifier_engine)
        self.step_eval_iterations = step_eval_iterations
        self.logger = logger
        
    def verify(self, instance: Variable, prompt: Variable, calculation: Variable) -> Variable:
        """
        Main verification function.
        
        Args:
            instance: Original problem/question
            prompt: The instruction/prompt used
            calculation: Result from instance + prompt (what we want to verify)
            
        Returns:
            Variable: Verified/improved calculation
        """
        if self.logger:
            print("INFO:textgrad:TextualVerifier:verify Starting Textual Verification...")
        
        # Step 1: Generate reasoning steps from instance
        reasoning_steps = self._generate_cot_steps(instance.value)
        
        # Step 2: Format steps properly
        formatted_steps = self._format_steps(reasoning_steps)
        
        # Step 3: Verify each step iteratively
        verified_steps = self._verify_each_step(instance.value, prompt.value, formatted_steps)
        
        # Step 4: Merge all verified steps
        merged_calculation = self._merge_verified_steps(prompt.value, verified_steps)
        
        # Step 5: Make final decision
        final_result = self._make_decision(calculation.value, merged_calculation)
        
        if self.logger:
            print("[V] Verification complete!")
        return Variable(final_result, requires_grad=True, role_description="verified calculation")
    
    def _generate_cot_steps(self, instance: str) -> List[str]:
        """Generate Chain of Thought steps from the instance."""
        if self.logger:
            print("INFO:textgrad:TextualVerifier:generate_cot_steps Generating CoT steps...")
        
        cot_prompt = COT_PROMPT.format(instance)
        
        response = self.engine(cot_prompt)
        steps = self._extract_steps_from_response(response)
        
        if self.logger:
            print(f"INFO:textgrad:TextualVerifier:generate_cot_steps Generated {len(steps)} steps")
        return steps
    
    def _extract_steps_from_response(self, response: str) -> List[str]:
        """Extract steps from LLM response."""
        # Look for <Step>...</Step> patterns
        step_pattern = r"<Step>(.*?)</Step>"
        steps = re.findall(step_pattern, response, re.DOTALL)
        
        # Clean up steps
        cleaned_steps = [step.strip() for step in steps if step.strip()]
        
        # Fallback if no tags found
        if not cleaned_steps:
            lines = response.split('\n')
            cleaned_steps = [line.strip() for line in lines 
                           if line.strip() and len(line.strip()) > 10]
        
        return cleaned_steps[:5]  # Limit to 5 steps max
    
    def _format_steps(self, steps: List[str]) -> List[str]:
        """Format steps for better processing."""
        if self.logger:
            print("INFO:textgrad:TextualVerifier:format_steps Formatting steps...")
        
        formatted = []
        for i, step in enumerate(steps):
            formatted_step = f"Step {i+1}: {step}"
            formatted.append(formatted_step)
        
        return formatted
    
    def _verify_each_step(self, instance: str, prompt: str, formatted_steps: List[str]) -> List[str]:
        """Verify each step by generating variants and voting."""
        if self.logger:
            print("INFO:textgrad:TextualVerifier:verify_each_step Verifying each step...")
        
        verified_steps = []
        
        for i, step in enumerate(formatted_steps):
            if self.logger:
                print(f"  Verifying step {i+1}/{len(formatted_steps)}")
            
            # Generate variants for this step
            variants = self._generate_step_variants(instance, prompt, step)
            
            # Vote on best variant
            best_variant = self._vote_on_variants(step, variants)
            
            verified_steps.append(best_variant)
        
        return verified_steps
    
    def _generate_step_variants(self, instance: str, prompt: str, step: str) -> List[str]:
        """Generate multiple variants of a calculation step."""
        variants = []
        
        for iteration in range(self.step_eval_iterations):
            variant_prompt = VARIANT_GENERATION_PROMPT.format(
                instance, prompt, step, iteration + 1
            )
            
            variant = self.engine(variant_prompt)
            variants.append(variant.strip())
        
        return variants
    
    def _vote_on_variants(self, original_step: str, variants: List[str]) -> str:
        """Vote on the most significant/correct variant."""
        variants_text = chr(10).join(f"{i+1}. {var}" for i, var in enumerate(variants))
        
        voting_prompt = VOTING_PROMPT.format(original_step, variants_text)
        
        best_step = self.engine(voting_prompt)
        return best_step.strip()
    
    def _merge_verified_steps(self, prompt: str, verified_steps: List[str]) -> str:
        """Merge all verified steps into one coherent calculation."""
        if self.logger:
            print("INFO:textgrad:TextualVerifier:merge_verified_steps Merging verified steps...")
        
        steps_text = chr(10).join(f"{i+1}. {step}" for i, step in enumerate(verified_steps))
        
        merge_prompt = MERGE_STEPS_PROMPT.format(prompt, steps_text)
        
        merged = self.engine(merge_prompt)
        return merged.strip()
    
    def _make_decision(self, original_calculation: str, merged_calculation: str) -> str:
        """Make final decision: update, merge, or pass."""
        if self.logger:
            print("INFO:textgrad:TextualVerifier:make_decision Making final decision...")
        
        decision_prompt = DECISION_PROMPT.format(original_calculation, merged_calculation)
        
        decision_response = self.engine(decision_prompt)
        
        # Parse decision
        if "REPLACE" in decision_response:
            if self.logger:
                print("[X] Original feedback insufficient - using process-focused version")
            return merged_calculation
        elif "ENHANCE" in decision_response:
            if self.logger:
                print("[-] Enhancing original feedback with process guidance")
            # Extract final feedback after the decision
            final_feedback = decision_response.split(":", 1)[-1].strip()
            return final_feedback if final_feedback else merged_calculation
        else:
            if self.logger:
                print("[V] Original feedback is already process-focused")
            return original_calculation
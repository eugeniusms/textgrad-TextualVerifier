import re
from textgrad.variable import Variable
from typing import Union, List
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .verifier import Verifier
from .verifier_prompts_experiment import (
    VARIANT_GENERATION_PROMPT_WITH_CONTEXT,
    VOTING_PROMPT_WITH_CONTEXT,
    DECISION_PROMPT,
)

"""
[Experiment Using TextualVerifier V2]

Due to step is exist step-by-step format, so skip:
1. _generate_cot_steps function -> initially formatted
2. _merge_verified_steps PROMPT phase -> for feedback building
3. _make_decision Enhance step -> due to specified output
"""
class TextualVerifierExperiment(Verifier):
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
        reasoning_steps = self._extract_steps_from_response(instance.value)

        # Step 2: Format steps properly
        formatted_steps = self._format_steps(reasoning_steps)

        print("FORMATTED_STEPS", formatted_steps)

        # Step 3: Verify each step iteratively
        verified_steps = self._verify_each_step_with_context(instance.value, prompt.value, formatted_steps)
        
        # Step 4: Merge all verified steps
        merged_calculation = self._merge_verified_steps(verified_steps)
        
        # Step 5: Make final decision
        final_result = self._make_decision(formatted_steps, merged_calculation)
        
        if self.logger:
            print("[V] Verification complete!")

        return Variable(final_result, requires_grad=True, role_description="verified calculation")

    def _extract_steps_from_response(self, response: str) -> List[str]:
        # Look for <Step>...</Step> patterns
        step_pattern = r"<Step>(.*?)</Step>"
        steps = re.findall(step_pattern, response, re.DOTALL)

        if steps:
            # Clean up extracted steps (remove leftover HTML or LaTeX artifacts if needed)
            return [step.strip() for step in steps if step.strip()]
        
        # Fallback: assume line-by-line reasoning
        lines = response.split('\n')
        return [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
    
    def _format_steps(self, steps: List[str]) -> List[str]:
        if self.logger:
            print("INFO:textgrad:TextualVerifier:format_steps Formatting steps...")

        return [f"Step {i+1}: {step}" for i, step in enumerate(steps)]

    
    def _verify_each_step_with_context(self, instance: str, prompt: str, formatted_steps: List[str]) -> List[str]:
        """
        ENHANCED: Verify each step by incorporating previous cumulative context.
        This strengthens step-by-step coherence and supports more informed verification.
        """
        if self.logger:
            print("INFO:textgrad:TextualVerifierV2Enhanced:verify_each_step_with_context Verifying with cumulative context...")
        
        verified_steps = []
        cumulative_context = ""
        
        for i, step in enumerate(formatted_steps):
            if self.logger:
                print(f"Verifying step {i+1}/{len(formatted_steps)} with context from {len(verified_steps)} previous steps")
            
            # Build cumulative context from all previous verified steps
            if verified_steps:
                cumulative_context = "\n".join([f"Previous Step {j+1}: {prev_step}" 
                                              for j, prev_step in enumerate(verified_steps)])
            
            is_last_step = False
            if i + 1 == len(formatted_steps):
                is_last_step = True

            # ENHANCEMENT: Generate variants with cumulative context
            variants = self._generate_step_variants_with_context(
                instance, prompt, step, cumulative_context, i+1, is_last_step
            )
            
            # Vote on best variant considering context
            best_variant = self._vote_on_variants_with_context(step, variants, cumulative_context, i)
            
            verified_steps.append(best_variant)
        
        return verified_steps
    
    def _generate_step_variants_with_context(self, 
                                            instance: str, 
                                            prompt: str, 
                                            current_step: str, 
                                            cumulative_context: str, 
                                            step_i: str,
                                            is_last_step: bool
                                            ) -> List[str]:
        """
        ENHANCED: Generate multiple variants of a step considering cumulative context.
        This ensures each step builds logically on previous verified steps.
        """
        variants = []
        
        for iteration in range(self.step_eval_iterations):
            enhanced_prompt = VARIANT_GENERATION_PROMPT_WITH_CONTEXT.format(
                instance,
                prompt,
                cumulative_context if cumulative_context else "None (this is the first step)",
                current_step,
                is_last_step,
                step_i,
                iteration + 1
            )
            
            variant = self.engine(enhanced_prompt)
            variants.append(variant.strip())
        
        return variants
    
    def _vote_on_variants_with_context(self, original_step: str, variants: List[str], 
                                     cumulative_context: str, step_i: str) -> str:
        """
        ENHANCED: Vote on variants considering cumulative context for consistency.
        """
        variants_text = "\n".join(f"{var}" for var in enumerate(variants))
        
        enhanced_voting_prompt = VOTING_PROMPT_WITH_CONTEXT.format(
            cumulative_context if cumulative_context else "None (this is the first step)",
            original_step,
            variants_text,
            step_i
        )
        
        best_step = self.engine(enhanced_voting_prompt)

        return best_step.strip()

    def _merge_verified_steps(self, verified_steps: List[str]) -> str:
        if self.logger:
            print("INFO:textgrad:TextualVerifier:merge_verified_steps Merging verified steps...")
        
        return "\n".join(verified_steps)  # No need for enumerate

    def _convert_steps_to_tags(self, text: str) -> str:
        """
        Convert 'Step X: ...' lines to <Step>...</Step> and include everything after
        the last step into the last <Step> block (e.g., # Answer and final value).
        """
        lines = text.strip().splitlines()
        step_pattern = re.compile(r"^Step \d+:\s*(.*)")
        output = []

        current_step_content = None
        for line in lines:
            match = step_pattern.match(line)
            if match:
                # Save previous step if exists
                if current_step_content is not None:
                    output.append(f"<Step>{current_step_content.strip()}</Step>")
                current_step_content = match.group(1)
            else:
                # Continuation of current step or post-step content
                if current_step_content is not None:
                    current_step_content += "\n" + line
                else:
                    # If no step yet, just keep accumulating
                    current_step_content = line

        # Append last step with post-content included
        if current_step_content:
            output.append(f"<Step>{current_step_content.strip()}</Step>")

        return "\n".join(output)

    def _make_decision(self, original_calculation: str, merged_calculation: str) -> str:
        """Make final decision: update or pass."""
        if self.logger:
            print("INFO:textgrad:TextualVerifier:make_decision Making final decision...")

        decision_prompt = DECISION_PROMPT.format(original_calculation, merged_calculation)

        decision_response = self.engine(decision_prompt)
        
        # Parse decision
        if "REPLACE" in decision_response:
            if self.logger:
                print("[X] Original is insufficient - using verified version")
            return self._convert_steps_to_tags(merged_calculation)
        else: # No Enhance -> Due to Specified Output
            if self.logger:
                print("[V] Original is already correct")
            return self._convert_steps_to_tags(original_calculation)
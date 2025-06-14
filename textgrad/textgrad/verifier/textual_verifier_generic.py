import re
from textgrad.variable import Variable
from typing import Union, List
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .verifier import Verifier
from .verifier_prompts_generic import (
    COT_PROMPT,
    VARIANT_GENERATION_PROMPT,
    VARIANT_GENERATION_PROMPT_STEP_BASED,
    MAJORITY_VOTING_PROMPT,
)

# OUTPUT: Variable("<VERIFIED></VERIFIED><VERIFIED></VERIFIED>")
class TextualVerifierGeneric(Verifier):
    def __init__(self,
                verifier_engine: Union[str, EngineLM], # LLM to verify
                use_cot_generation: bool = False, # True if instance need to use CoT Prompt Generation <Step>...</Step>
                use_step_breakdown: bool = True, # True if instance need to step breakdown -> Solution Optimization, Code Optimization
                                                 # False if don't need to step breakdown -> Prompt Optimization
                num_variants: int = 3, # Variant number
                enable_logging: bool = False): # If want to debug

        self.engine = validate_engine_or_get_default(verifier_engine)
        self.use_cot_generation = use_cot_generation
        self.use_step_breakdown = use_step_breakdown
        self.num_variants = num_variants
        self.enable_logging = enable_logging
        
    def verify(self, instance: Variable, instruction: Variable, calculation: Variable, verification_prompt: Variable) -> Variable:
        """
        [GENERIC VERIFICATION]
        Definition:
        - instance: input
        - instruction: instruction
        - calculation: output of instruction to input
        - verification_prompt: verification we want to use
        Formula:
        ```
        instance + instruction => calculation
        instance + calculation + verification_prompt => verified_calculation
        ```

        [CASE VERIFY LOSS VALUE]
        Definition:
        - instance = prediction
        - instruction = loss instruction
        - calculation = loss value
        Formula:
        ```
        prediction + loss instruction => loss value
        prediction + loss value + verification prompt => verified loss value
        ```

        [CASE VERIFY OPTIMIZER RESULT]
        Definition:
        - instance = (prediction & loss value)
        - instruction = optimization instruction
        - calculation = optimized prediction
        ```
        (prediction & loss value) + optimization instruction => optimized prediction
        (prediction & loss value) + optimized prediction + verification prompt => verified optimized prediction
        ```
        """
        
        instance = instance.value

        # 1. Breakdown to steps with CoT
        if self.use_cot_generation:
            updated_instance = self._generate_cot_steps(instance) # Format: "<Step> ... </Step>"
        else:
            updated_instance = instance

        # 2. Convert CoT to list
        if self.use_step_breakdown:
            step_breakdown = self._convert_cot_format_to_list(updated_instance) # Format: List[str]
        else:
            step_breakdown = [updated_instance] # Just only 1 step in no step breakdown flag
        
        voted_variant_list = []
        for step in step_breakdown:
            # 3. Generate variants
            generated_variants = self._generate_variants(step, instruction.value, calculation.value, verification_prompt.value)
            # 4. Vote variants
            voted_variant = self._majority_vote_variants(generated_variants)
            voted_variant_list.append(voted_variant)

        # 5. Merge voted variants
        verified_calculaton = "\n".join(f"<VERIFIED>{step}</VERIFIED>" for step in enumerate(voted_variant_list))

        return Variable(verified_calculaton, requires_grad=True, role_description="verified calculation")

    # For Instance
    def _generate_cot_steps(self, instance: str) -> List[str]:
        generate_cot_prompt = COT_PROMPT.format(
            instance=instance
        )
        generated_cot = self.engine(generate_cot_prompt)
        return generated_cot

    # For Instance
    def _convert_cot_format_to_list(self, cot_formatted: str) -> List[str]:
        step_pattern = r"<Step>(.*?)</Step>"
        steps = re.findall(step_pattern, cot_formatted, re.DOTALL)
        
        # Clean up steps
        cleaned_steps = [step.strip() for step in steps if step.strip()]
        
        # Fallback if no tags found
        if not cleaned_steps:
            lines = cot_formatted.split('\n')
            cleaned_steps = [line.strip() for line in lines 
                           if line.strip() and len(line.strip()) > 10]
        
        return cleaned_steps

    # 
    def _generate_variants(self, step: str, instruction: str, calculation: str, verification_prompt: str) -> List[str]:
        generated_variants = []
        for i in range(self.num_variants):
            variant_prompt = ""
            if self.use_step_breakdown:
                variant_prompt = VARIANT_GENERATION_PROMPT_STEP_BASED.format(
                    instance=step,

                    variant_no=i
                )
            else:
                variant_prompt = VARIANT_GENERATION_PROMPT.format(
                    instance=step,
                    instruction=instruction,
                    calculation=calculation,
                    verification_prompt=verification_prompt
                )
                
            new_variant = self.engine(variant_prompt)
            generated_variants.append(new_variant)
        return generated_variants

    def _majority_vote_variants(self, generated_variants: List[str]) -> str:
        voting_prompt = MAJORITY_VOTING_PROMPT.format(
            generated_variants=generated_variants
        )
        voted_variant = self.engine(voting_prompt)
        return voted_variant
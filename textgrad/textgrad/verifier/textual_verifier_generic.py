import re
from textgrad.variable import Variable
from typing import Union, List
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .verifier import Verifier
from .verifier_prompts_generic import (
    DEFAULT_VERIFICATION_TASK_PROMPTS,
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
                verification_task_prompts: List[str] = DEFAULT_VERIFICATION_TASK_PROMPTS,
                enable_logging: bool = False): # If want to debug

        self.engine = validate_engine_or_get_default(verifier_engine)
        self.use_cot_generation = use_cot_generation
        self.use_step_breakdown = use_step_breakdown
        self.verification_task_prompts = verification_task_prompts
        self.enable_logging = enable_logging
        
    def verify(self, instance: Variable, instruction: Variable, calculation: Variable) -> Variable:
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
        
        calculation_value = calculation.value

        # 1. Breakdown to steps with CoT
        if self.use_cot_generation:
            updated_calculation = self._generate_cot_steps(calculation_value) # Format: "<Step> ... </Step>"
        else:
            updated_calculation = calculation_value

        # 2. Convert CoT to list
        if self.use_step_breakdown:
            step_breakdown = self._convert_cot_format_to_list(updated_calculation) # Format: List[str]
        else:
            step_breakdown = [updated_calculation] # Just only 1 step in no step breakdown flag
        
        voted_variant_list = []
        for step in step_breakdown:
            # 3. Generate variants
            generated_variants = self._generate_variants(instance=instance.value, 
                                                        instruction=instruction.value, 
                                                        calculation=calculation.value)
            # 4. Vote variants
            voted_variant = self._majority_vote_variants(step, generated_variants)
            voted_variant_list.append(voted_variant)

        # 5. Merge voted variants
        verified_calculaton = "\n".join(f"<VERIFIED>{step}</VERIFIED>" for _, step in enumerate(voted_variant_list))

        return Variable(verified_calculaton, requires_grad=True, role_description="verified calculation")

    # For Calculation
    def _generate_cot_steps(self, calculation: str) -> List[str]:
        generate_cot_prompt = COT_PROMPT.format(
            calculation=calculation
        )
        generated_cot = self.engine(generate_cot_prompt)
        return generated_cot

    # For Calculation
    def _convert_cot_format_to_list(self, cot_formatted: str) -> List[str]:
        step_pattern = r"<STEP>(.*?)</STEP>"
        steps = re.findall(step_pattern, cot_formatted, re.DOTALL)
        
        # Clean up steps
        cleaned_steps = [step.strip() for step in steps if step.strip()]
        
        # Fallback if no tags found
        if not cleaned_steps:
            lines = cot_formatted.split('\n')
            cleaned_steps = [line.strip() for line in lines 
                           if line.strip() and len(line.strip()) > 10]
        
        return cleaned_steps

    # Generate Verified Variants of Calculation
    def _generate_variants(self, instance: str, instruction: str, calculation: str) -> List[str]:
        generated_variants = []
        for i in range(len(self.verification_task_prompts)):
            variant_prompt = ""
            if self.use_step_breakdown:
                variant_prompt = VARIANT_GENERATION_PROMPT_STEP_BASED.format(
                    instance=instance,
                    variant_no=i
                )
            else:
                variant_prompt = VARIANT_GENERATION_PROMPT.format(
                    instance=instance,
                    instruction=instruction,
                    calculation=calculation,
                    verification_task_prompt=self.verification_task_prompts[i]
                )
                
            new_variant = self.engine(variant_prompt)
            generated_variants.append(new_variant)

        print("GENERATED VARIANTS", generated_variants)
        return generated_variants

    def _majority_vote_variants(self, calculation: str, generated_variants: List[str]) -> str:
        print("CALCULATION", calculation)

        generated_variants_formatted = "\n".join(f"Variant {i+1}: ```{variant}```" for i, variant in enumerate(generated_variants))
        voting_prompt = MAJORITY_VOTING_PROMPT.format(
            calculation=calculation,
            generated_variants=generated_variants_formatted
        )
        voted_variant = self.engine(voting_prompt)
        return voted_variant
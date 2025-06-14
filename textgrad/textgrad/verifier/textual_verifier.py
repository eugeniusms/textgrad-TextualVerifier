import re
from textgrad.variable import Variable
from typing import Union, List
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .verifier import Verifier
from .verifier_prompts import (
    DEFAULT_VERIFICATION_TASK_PROMPTS,
    COT_PROMPT,
    VARIANT_GENERATION_PROMPT,
    MAJORITY_VOTING_PROMPT,
)

# OUTPUT: Variable("<VERIFIED></VERIFIED><VERIFIED></VERIFIED>")
class TextualVerifier(Verifier):
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

        if self.enable_logging:
            print("INFO:textgrad:TextualVerifier:verify Start verification ...")
        
        calculation_value = calculation.value

        # 1. Breakdown to steps with CoT
        if self.use_cot_generation:
            updated_calculation = self._generate_cot_steps(calculation_value) 
        else:
            updated_calculation = calculation_value
        
        if self.enable_logging:
            print("INFO:textgrad:TextualVerifier:verify Generate calculation list ...")

        # 2. Convert CoT to list
        if self.use_step_breakdown:
            step_breakdown = self._convert_cot_format_to_list(updated_calculation) # Format: List[str]
        else:
            step_breakdown = [updated_calculation] # Just only 1 step in no step breakdown flag
        
        if self.enable_logging:
            print("INFO:textgrad:TextualVerifier:verify Calculation list is ready ...")

        initial_context = f"These are previous context to help you verify calculation:\n{instance.value}\n"
        voted_variant_list = []
        for i, step in enumerate(step_breakdown):
            # 3. Generate variants
            if self.enable_logging:
                print(f"INFO:textgrad:TextualVerifier:verify Verify step {i+1}/{len(step_breakdown)} ...")

            context = initial_context.join(f"Step {i+1}: ```{voted_variant}```" for i, voted_variant in enumerate(voted_variant_list))
            generated_variants = self._generate_variants(instance=instance.value, 
                                                         instruction=instruction.value, 
                                                         previous_context=context,
                                                         calculation=step,
                                                         i=i+1)
            # 4. Vote variants
            voted_variant = self._majority_vote_variants(calculation=step, 
                                                         generated_variants=generated_variants,
                                                         i=i+1)
            
            voted_variant_list.append(voted_variant)

        # 5. Merge voted variants
        verified_calculaton = "\n".join(f"<VERIFIED>{step}</VERIFIED>" for _, step in enumerate(voted_variant_list))

        return Variable(verified_calculaton, requires_grad=True, role_description="verified calculation")

    # For Calculation
    def _get_ready_calculation(self, calculation: str) -> List[str]:
        updated_calculation = calculation
        if self.use_cot_generation: 
            generate_cot_prompt = COT_PROMPT.format(
                calculation=calculation
            )
            updated_calculation = self.engine(generate_cot_prompt)
        return updated_calculation

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
    def _generate_variants(self, instance: str, instruction: str, previous_context: str, calculation: str, i: str) -> List[str]:
        generated_variants = []
        for j in range(len(self.verification_task_prompts)):
            print(f"INFO:textgrad:TextualVerifier:generate_variants Generating step {i} variant {j+1}/{len(self.verification_task_prompts)} ...")

            if not self.use_step_breakdown: # If no use_step_breakdown -> ""
                previous_context = ""

            variant_prompt = VARIANT_GENERATION_PROMPT.format(
                instance=instance,
                instruction=instruction,
                previous_context=previous_context,
                calculation=calculation,
                verification_task_prompt=self.verification_task_prompts[j]
            )
                
            new_variant = self.engine(variant_prompt)
            generated_variants.append(new_variant)

        return generated_variants

    def _majority_vote_variants(self, calculation: str, generated_variants: List[str], i: str) -> str:
        if self.enable_logging:
            print(f"INFO:textgrad:TextualVerifier:majority_vote_variants Run majority voting for step {i}...")

        generated_variants_formatted = "\n".join(f"Variant {j+1}: ```{variant}```" for j, variant in enumerate(generated_variants))
        voting_prompt = MAJORITY_VOTING_PROMPT.format(
            calculation=calculation,
            generated_variants=generated_variants_formatted
        )
        voted_variant = self.engine(voting_prompt)

        return voted_variant
import re
from typing import Union, List
from textgrad.variable import Variable
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .verifier import Verifier
from .verifier_prompts import (
    DEFAULT_VERIFICATION_TASK_PROMPTS,
    COT_PROMPT,
    VARIANT_GENERATION_PROMPT,
    MAJORITY_VOTING_PROMPT,
)


class TextualVerifier(Verifier):
    """
    A verifier that uses textual reasoning to verify calculations through:
    1. Chain-of-Thought (CoT) generation
    2. Step breakdown
    3. Variant generation
    4. Majority voting
    
    The verification process outputs results in the format:
    "<VERIFIED>step1</VERIFIED><VERIFIED>step2</VERIFIED>..."
    """
    
    def __init__(
        self,
        verifier_engine: Union[str, EngineLM],
        use_cot_generation: bool = False,
        use_step_breakdown: bool = True,
        verification_task_prompts: List[str] = DEFAULT_VERIFICATION_TASK_PROMPTS,
        enable_logging: bool = False
    ):
        """
        Initialize the TextualVerifier.
        
        Args:
            verifier_engine: LLM engine to use for verification
            use_cot_generation: Whether to use Chain-of-Thought prompt generation
                               with <Step>...</Step> formatting
            use_step_breakdown: Whether to break down solutions into steps
                               - True: For solution/code optimization (multi-step)
                               - False: For prompt optimization (single-step)
            verification_task_prompts: List of verification prompts to use
            enable_logging: Whether to enable debug logging
        """
        self.engine = validate_engine_or_get_default(verifier_engine)
        self.use_cot_generation = use_cot_generation
        self.use_step_breakdown = use_step_breakdown
        self.verification_task_prompts = verification_task_prompts
        self.enable_logging = enable_logging
        
    def verify(
        self, 
        instance: Variable, 
        instruction: Variable, 
        calculation: Variable
    ) -> Variable:
        """
        Verify a calculation using the configured verification strategy.
        
        The verification follows this general formula:
        instance + instruction => calculation
        instance + calculation + verification_prompt => verified_calculation
        
        Use Cases:
        1. Loss Value Verification:
           - instance = prediction
           - instruction = loss instruction  
           - calculation = loss value
           
        2. Optimizer Result Verification:
           - instance = (prediction & loss value)
           - instruction = optimization instruction
           - calculation = optimized prediction
        
        Args:
            instance: Input data/context
            instruction: Instruction applied to the instance
            calculation: Output result from applying instruction to instance
            
        Returns:
            Variable containing verified calculation with <VERIFIED> tags
        """
        self._log("Start verification process...")
        
        # Phase 1: Generate Chain-of-Thought steps if enabled
        processed_calculation = self._process_calculation_with_cot(calculation.value)
        
        # Phase 2: Break down into verification steps
        verification_steps = self._create_verification_steps(processed_calculation)
        
        self._log(f"Ready to verify {len(verification_steps)} calculation steps...")
        
        # Phase 3: Verify each step through variant generation and voting
        verified_steps = self._verify_steps(
            instance=instance.value,
            instruction=instruction.value, 
            steps=verification_steps
        )
        
        # Phase 4: Format final result
        verified_result = self._format_verified_result(verified_steps)
        
        return Variable(
            verified_result, 
            requires_grad=True, 
            role_description="verified calculation"
        )

    def _process_calculation_with_cot(self, calculation_value: str) -> str:
        """
        Process calculation with Chain-of-Thought generation if enabled.
        
        Args:
            calculation_value: Raw calculation to process
            
        Returns:
            Processed calculation (with CoT steps if enabled)
        """
        if self.use_cot_generation:
            return self._generate_cot_steps(calculation_value)
        return calculation_value

    def _create_verification_steps(self, calculation: str) -> List[str]:
        """
        Create list of steps for verification.
        
        Args:
            calculation: Calculation to break down
            
        Returns:
            List of verification steps
        """
        if self.use_step_breakdown:
            return self._convert_cot_format_to_list(calculation)
        return [calculation]  # Single step if breakdown disabled

    def _verify_steps(
        self, 
        instance: str, 
        instruction: str, 
        steps: List[str]
    ) -> List[str]:
        """
        Verify each step through variant generation and majority voting.
        
        Args:
            instance: Original instance value
            instruction: Original instruction value
            steps: List of steps to verify
            
        Returns:
            List of verified steps
        """
        initial_context = f"Previous context for verification:\n{instance}\n"
        verified_steps = []
        
        for i, step in enumerate(steps):
            self._log(f"Verifying step {i+1}/{len(steps)}...")
            
            # Build context from previous verified steps
            context = self._build_step_context(initial_context, verified_steps)
            
            # Generate variants for this step
            variants = self._generate_variants(
                instance=instance,
                instruction=instruction,
                previous_context=context,
                calculation=step,
                step_number=i+1
            )
            
            # Vote on best variant
            if len(self.verification_task_prompts) > 1:
                best_variant = self._majority_vote_variants(
                    calculation=step,
                    generated_variants=variants,
                    step_number=i+1
                )
            else:
                best_variant = variants[0]
            
            verified_steps.append(best_variant)
            
        return verified_steps

    def _build_step_context(
        self, 
        initial_context: str, 
        verified_steps: List[str], 
    ) -> str:
        """
        Build context string from previous verified steps.
        
        Args:
            initial_context: Base context string
            verified_steps: Previously verified steps
            current_step: Current step index
            
        Returns:
            Formatted context string
        """
        if not self.use_step_breakdown:
            return ""
            
        step_contexts = [
            f"Step {i+1}: ```{step}```" 
            for i, step in enumerate(verified_steps)
        ]
        
        return initial_context + "\n".join(step_contexts)

    def _format_verified_result(self, verified_steps: List[str]) -> str:
        """
        Format verified steps into final result with <VERIFIED> tags.
        
        Args:
            verified_steps: List of verified calculation steps
            
        Returns:
            Formatted result string with <VERIFIED> tags
        """
        return "\n".join(f"<VERIFIED>{step}</VERIFIED>" for step in verified_steps)

    # Chain-of-Thought Processing Methods
    
    def _generate_cot_steps(self, calculation: str) -> str:
        """
        Generate Chain-of-Thought formatted steps for a calculation.
        
        Args:
            calculation: Raw calculation to convert to CoT format
            
        Returns:
            CoT formatted calculation with <STEP> tags
        """
        cot_prompt = COT_PROMPT.format(calculation=calculation)
        return self.engine(cot_prompt)

    def _convert_cot_format_to_list(self, cot_formatted: str) -> List[str]:
        """
        Extract individual steps from CoT formatted text.
        
        Args:
            cot_formatted: Text containing <STEP>...</STEP> tags
            
        Returns:
            List of individual step strings
        """
        # Extract steps using regex pattern
        step_pattern = r"<STEP>(.*?)</STEP>"
        steps = re.findall(step_pattern, cot_formatted, re.DOTALL)
        
        # Clean whitespace from extracted steps
        cleaned_steps = [step.strip() for step in steps if step.strip()]
        
        # Fallback: split by lines if no <STEP> tags found
        if not cleaned_steps:
            lines = cot_formatted.split('\n')
            cleaned_steps = [
                line.strip() for line in lines 
                if line.strip() and len(line.strip()) > 10
            ]
        
        return cleaned_steps

    # Variant Generation and Voting Methods
    
    def _generate_variants(
        self, 
        instance: str, 
        instruction: str, 
        previous_context: str, 
        calculation: str, 
        step_number: int
    ) -> List[str]:
        """
        Generate multiple verified variants of a calculation step.
        
        Args:
            instance: Original instance
            instruction: Original instruction  
            previous_context: Context from previous steps
            calculation: Current calculation step
            step_number: Current step number for logging
            
        Returns:
            List of generated variants
        """
        variants = []
        
        for j, verification_prompt in enumerate(self.verification_task_prompts):
            self._log(
                f"Generating step {step_number} variant "
                f"{j+1}/{len(self.verification_task_prompts)}..."
            )
            
            # Skip context if step breakdown is disabled
            context = previous_context if self.use_step_breakdown else ""
            
            # Generate variant using verification prompt
            variant_prompt = VARIANT_GENERATION_PROMPT.format(
                instance=instance,
                instruction=instruction,
                previous_context=context,
                calculation=calculation,
                verification_task_prompt=verification_prompt
            )
            
            variant = self.engine(variant_prompt)
            variants.append(variant)

        return variants

    def _majority_vote_variants(
        self, 
        calculation: str, 
        generated_variants: List[str], 
        step_number: int
    ) -> str:
        """
        Select best variant through majority voting.
        
        Args:
            calculation: Original calculation step
            generated_variants: List of generated variants
            step_number: Step number for logging
            
        Returns:
            Selected best variant
        """
        self._log(f"Running majority voting for step {step_number}...")
        
        # Format variants for voting prompt
        variants_text = "\n".join(
            f"Variant {j+1}: ```{variant}```" 
            for j, variant in enumerate(generated_variants)
        )
        
        # Perform majority voting
        voting_prompt = MAJORITY_VOTING_PROMPT.format(
            calculation=calculation,
            generated_variants=variants_text
        )
        
        return self.engine(voting_prompt)

    # Utility Methods
    
    def _log(self, message: str) -> None:
        """
        Log message if logging is enabled.
        
        Args:
            message: Message to log
        """
        if self.enable_logging:
            print(f"INFO:textgrad:TextualVerifier: {message}")